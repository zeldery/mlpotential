'''
Symmetry Function computation, similar to AEV in TorchANI
Mimic the aev.py file in TorchANI package
Link: https://aiqm.github.io/torchani/
'''

import torch
from torch import nn
from .utils import cumsum_from_zero, create_double_index, create_double_index_batch

def cut_off_function(x, cut_off):
    '''
    Assume all x is less than cut_off
    '''
    return 0.5*torch.cos(x*torch.pi/cut_off) + 0.5

def create_pairwise_encoder(n_elements):
    '''
    Return the encoder to change a pair of atomic index into index
    Example, if there are 3 type of atom, it will change
    (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (0,2) -> 2 etc ...
    '''
    index = torch.triu_indices(n_elements, n_elements, dtype=torch.long)
    pairwise_encoder = torch.full((100, 100), -1, dtype=torch.long)
    order = torch.arange(index.shape[1], dtype=torch.long)
    pairwise_encoder[index[0,:], index[1,:]] = order
    pairwise_encoder[index[1,:], index[0,:]] = order
    return pairwise_encoder

def create_triple_index(index):
    '''
    Create the group of pair_index to group the pair in index with common part
    Take in index as [2, n_pair] of interaction already less than cut_off
    Return central atoms, pair_index store the index of index, and flip of either the interaction need flip for element encoder
    Use for single structure, batch mode, with or without cell shift
    '''
    # Sort the index to see how many pairs each central atoms have
    x = index.view(-1)
    sort_index, sort_order = x.sort()
    unique_index, count = torch.unique_consecutive(sort_index, return_inverse=False, return_counts=True)
    pair_size = count * (count - 1) // 2

    # Get the central atom index
    pair_index = torch.repeat_interleave(pair_size)
    central_atom = unique_index.index_select(0, pair_index) # Use this to ensure if the index is not contains 0, 1 ...
    
    # Get all pairs of combinations, up to counts[i] set
    maximum_count = count.max().item()
    n_pair_size = pair_size.shape[0]
    all_pair_index = torch.tril_indices(maximum_count, maximum_count, -1, device=index.device).unsqueeze(1).expand(-1, n_pair_size, -1) # All combinations possible
                                                                                                                                        # Shape: [2, n_pair, n_internal_pair]
    mask = (torch.arange(all_pair_index.shape[2], device=index.device) < pair_size.unsqueeze(1)).flatten() # Eliminate for pair base on count
    reduced_pair_index = all_pair_index.flatten(1, 2)[:, mask]
    reduced_pair_index += cumsum_from_zero(count).index_select(0, pair_index) # Adjust the number of later interaction by counts
    
    # Reversed the sorting to original index order
    real_pair_index = sort_order[reduced_pair_index]

    # Check if the order of real_pair_index is [central_atom, other_atom] (False value) or [other_atom, central_atom] (True value)
    # Needed for compute the order in AEV
    n_index = index.shape[1]
    flip = real_pair_index >= n_index # Because all the index larger than n_index is from the row below

    return central_atom, real_pair_index % n_index, flip

def compute_radial_function(distance, eta, mu, cut_off):
    '''
    The symmetry function form: exp ( - eta * (distance - mu)**2 ) * cut_off_function
    '''
    result = torch.exp(-eta[None,:,None] * (distance[:, None, None] - mu[None, None,:])**2) * cut_off_function(distance, cut_off)[:, None, None]
    return result.flatten(start_dim=1)

def compute_angular_function(vector, distance, nu, zeta, eta, mu, cut_off):
    '''
    vector: [2, n_triple, 3]
    distance: [2, n_triple]
    The angular form: ( 1 + cos(angle - nu)) ** zeta / 2**zeta * exp(-eta * (average_distance - mu)**2) * cut_off(distance1) * cut_off(distance2)
    '''
    inner_product = vector.prod(dim=0).sum(dim=-1)
    cosine_of_angle = inner_product / torch.clamp(distance.prod(dim=0), min=1e-9)
    angle = torch.acos(0.99*cosine_of_angle)
    cut_off_part = cut_off_function(distance, cut_off).prod(dim=0)
    part1 = (1+torch.cos(angle[:, None, None, None, None] - nu[None, :, None, None, None]))**zeta[None, None, :, None, None] / 2**zeta[None, None, :, None, None]
    part2 = torch.exp(-eta[None, None, None, :, None] * ((distance.sum(dim=0))[:, None, None, None, None]/2 - mu[None, None, None, None, :])**2)
    result = part1 * part2 * cut_off_part[:, None, None, None, None]
    return result.flatten(start_dim=1)

class SymmetryFunction(nn.Module):
    '''
    The symmetry function module to compute the symmetry function with set of parameters stored
    Don't need exact element, but the index (from 0 to n_element-1) to compute
    Need to enter n_elements to determine the order of aev
    The symmetry function take the form of exp ( - eta * (distance - mu)**2 ) * cut_off_function 
    for radial function and ( 1 + cos(angle - nu)) ** zeta / 2**zeta * exp(-eta * (average_distance - mu)**2) * cut_off(distance1) * cut_off(distance2)
    for angular function
    '''
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def set(self, n_elements, radial_eta, radial_mu, radial_cut_off, angular_nu, angular_zeta, angular_eta, angular_mu, angular_cut_off):
        self.register_buffer('radial_eta', torch.tensor(radial_eta, dtype=torch.float32))
        self.register_buffer('radial_mu', torch.tensor(radial_mu, dtype=torch.float32))
        self.register_buffer('radial_cut_off', torch.tensor(radial_cut_off, dtype=torch.float32))
        self.register_buffer('angular_nu', torch.tensor(angular_nu, dtype=torch.float32))
        self.register_buffer('angular_zeta', torch.tensor(angular_zeta, dtype=torch.float32))
        self.register_buffer('angular_eta', torch.tensor(angular_eta, dtype=torch.float32))
        self.register_buffer('angular_mu', torch.tensor(angular_mu, dtype=torch.float32))
        self.register_buffer('angular_cut_off', torch.tensor(angular_cut_off, dtype=torch.float32))
        self.register_buffer('n_elements', torch.tensor(n_elements))
        self.initialized = True

    def read(self, file_name):
        try:
            state_dict = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        except:
            raise IOError(f'Cannot load the state dictionary from {file_name}')
        self.load(state_dict)
    
    def write(self, file_name):
        if self.radial_eta.device != torch.device('cpu'):
            temp = self.to(torch.device('cpu'))
            state_dict = temp.state_dict()
        else:
            state_dict = self.state_dict()
        torch.save(state_dict, file_name)

    def load(self, data):
        self.register_buffer('radial_eta', data['radial_eta'])
        self.register_buffer('radial_mu', data['radial_mu'])
        self.register_buffer('radial_cut_off', data['radial_cut_off'])
        self.register_buffer('angular_nu', data['angular_nu'])
        self.register_buffer('angular_zeta', data['angular_zeta'])
        self.register_buffer('angular_eta', data['angular_eta'])
        self.register_buffer('angular_mu', data['angular_mu'])
        self.register_buffer('angular_cut_off', data['angular_cut_off'])
        self.register_buffer('n_elements', data['n_elements'])
        self.initialized = True

    def dump(self):
        return self.state_dict()
    
    def compute(self, atomic_index, positions):
        '''
        Compute AEV from atomic index and positions
        '''
        n_element = self.n_elements.item()
        n_atoms = atomic_index.shape[0]

        # Radial part
        index = create_double_index(positions, self.radial_cut_off)
        vector = positions[index[1],:] - positions[index[0],:]
        distance = vector.norm(2, -1)
        radial_term = compute_radial_function(distance, self.radial_eta, self.radial_mu, self.radial_cut_off)
        radial_aev = radial_term.new_zeros((n_atoms * n_element, radial_term.shape[1]))
        # Re-index to add to the result, from n_interaction with information of elements to n_atoms*n_elements
        # the order of new_index: 0 : atom=0, ele=0; 1 : atom=0, ele=1; ...
        new_index = index * n_element + atomic_index[index].flip([0])
        radial_aev.index_add_(dim=0, index=new_index[0,:], source=radial_term)
        radial_aev.index_add_(dim=0, index=new_index[1,:], source=radial_term)

        # Prepare for angular part
        # Use nonzero and index_select because it's faster than mask
        # Reduce the number of pairs for create_triple_index
        closer_index = (distance.detach() < self.angular_cut_off).nonzero().flatten()
        index = index.index_select(1, closer_index)
        vector = vector.index_select(0, closer_index)
        distance = distance.index_select(0, closer_index)
        
        central_atom, pair_index, flip = create_triple_index(index)
        # Getting the vector by the pair_index
        pair_vector = vector.index_select(0, pair_index.view(-1)).view(2, -1, 3)
        pair_vector = pair_vector * torch.where(flip, -1.0, 1.0).unsqueeze(-1) # Make sure it has correct direction
        # Get the atomic numbers of NOT the central atoms in the triple
        # Use for new_pair_index
        closer_atomic_index = atomic_index[index].index_select(1, pair_index.view(-1)).view(2, -1)
        other_atomic_index = torch.where(flip.view(-1), closer_atomic_index[0], closer_atomic_index[1]).view(2, -1)

        # Angular terms
        angular_term = compute_angular_function(pair_vector, distance.index_select(0, pair_index.view(-1)).view(2, -1), 
                                                self.angular_nu, self.angular_zeta, self.angular_eta, self.angular_mu, self.angular_cut_off)
        n_element_pair = n_element * (n_element+1)//2
        angular_aev = angular_term.new_zeros((n_atoms*n_element_pair, angular_term.shape[1]))
        pairwise_encoder = create_pairwise_encoder(self.n_elements).to(positions.device)
        new_pair_index = central_atom * n_element_pair + pairwise_encoder[other_atomic_index[0], other_atomic_index[1]]
        angular_aev.index_add_(0, new_pair_index, angular_term)
        return torch.cat([radial_aev.view(n_atoms, -1), angular_aev.view(n_atoms, -1)], dim=-1)

    def batch_compute(self, atomic_index_, positions):
        '''
        Similar to compute, but for batch
        atomic_index_: [n_structure, n_atoms] with -1 for no-use data
        positions: [n_structure, n_atoms, 3]
        '''
        n_element = self.n_elements.item()
        n_structure, n_atoms = atomic_index_.shape
        atomic_index = atomic_index_.flatten() # Pre-flatten

        # Radial part
        index = create_double_index_batch(atomic_index_, positions, self.radial_cut_off)
        pos = positions.view(-1, 3) # Flatten positions too
        vector = pos[index[1],:] - pos[index[0],:]
        distance = vector.norm(2, -1)
        radial_term = compute_radial_function(distance, self.radial_eta, self.radial_mu, self.radial_cut_off)
        radial_aev = radial_term.new_zeros((n_structure * n_atoms * n_element, radial_term.shape[1]))
        # Re-index to add to the result, from n_interaction with information of elements to n_structure*n_atoms*n_elements
        # the order of new_index: 0 : atom=0, ele=0; 1 : atom=0, ele=1; ...
        new_index = index * n_element + atomic_index[index].flip([0])
        radial_aev.index_add_(dim=0, index=new_index[0,:], source=radial_term)
        radial_aev.index_add_(dim=0, index=new_index[1,:], source=radial_term)

        # Prepare for angular part
        # Everything is the same for angular, with the flatten first dimension of reduced n_structure * n_atoms
        closer_index = (distance.detach() < self.angular_cut_off).nonzero().flatten()
        index = index.index_select(1, closer_index)
        vector = vector.index_select(0, closer_index)
        distance = distance.index_select(0, closer_index)
        
        central_atom, pair_index, flip = create_triple_index(index)
        # Getting the vector by the pair_index
        pair_vector = vector.index_select(0, pair_index.view(-1)).view(2, -1, 3)
        pair_vector = pair_vector * torch.where(flip, -1.0, 1.0).unsqueeze(-1) # Make sure it has correct direction
        # Get the atomic numbers of NOT the central atoms in the triple
        # Use for new_pair_index
        closer_atomic_index = atomic_index[index].index_select(1, pair_index.view(-1)).view(2, -1)
        other_atomic_index = torch.where(flip.view(-1), closer_atomic_index[0], closer_atomic_index[1]).view(2, -1)

        # Angular terms
        angular_term = compute_angular_function(pair_vector, distance.index_select(0, pair_index.view(-1)).view(2, -1), 
                                                self.angular_nu, self.angular_zeta, self.angular_eta, self.angular_mu, self.angular_cut_off)
        n_element_pair = n_element * (n_element+1)//2
        angular_aev = angular_term.new_zeros((n_structure*n_atoms*n_element_pair, angular_term.shape[1]))
        pairwise_encoder = create_pairwise_encoder(self.n_elements).to(positions.device)
        new_pair_index = central_atom * n_element_pair + pairwise_encoder[other_atomic_index[0], other_atomic_index[1]]
        angular_aev.index_add_(0, new_pair_index, angular_term)
        # Return the aev, reshape to n_structure x n_atoms x n_aev
        return torch.cat([radial_aev.view(n_structure, n_atoms, -1), angular_aev.view(n_structure, n_atoms, -1)], dim=-1)

    def compute_pbc(self, atomic_index, positions, cell):
        '''
        Not implemented yet
        '''
        pass

    def batch_compute_pbc(self, atomic_index, positions, cell):
        '''
        Not implemented yet
        '''
        pass

