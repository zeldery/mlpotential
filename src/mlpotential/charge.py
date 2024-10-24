'''
Charge equilibration procedure
'''

import torch
from torch import nn
from .net import IndexNetwork, IndexValue
from .utils import cumsum_from_zero, BOHR_TO_ANGSTROM

def compute_charge_equilibration(positions, hardness, electronegativity, sigma, total_charge):
    '''
    Compute charge-equilibration scheme for single structure
    hardness: the term in front of Q^2 (Angstrom ** (-1) unit)
    sigma: the standard deviation of the Gaussian distribution for the charge (Angstrom)
    Return tuple of charge and energy (converted to float64)
    '''
    device = positions.device
    n_atoms = positions.shape[0]
    index = torch.triu_indices(n_atoms, n_atoms, 1, device=device)
    distance = (positions[index[1,:],:] - positions[index[0,:],:]).norm(dim=1)
    gamma = 1/(2 * (sigma[index[1,:]]**2 + sigma[index[0,:]]**2))**0.5 # Include 2 to simplify the cross-term expression
    a = torch.ones((n_atoms+1, n_atoms+1), dtype=torch.float32, device=device)
    seq = torch.arange(0, n_atoms, device=device)
    a[seq, seq] = (hardness + 1 / (sigma * torch.pi**0.5)) * BOHR_TO_ANGSTROM # Change the unit, should be 1/ANSTROM_TO_BOHR
    a[index[0,:], index[1,:]] = torch.erf(distance * gamma) / distance * BOHR_TO_ANGSTROM # Similar to above
    a[index[1,:], index[0,:]] = torch.erf(distance * gamma) / distance * BOHR_TO_ANGSTROM # The unit of distance and gamma cancel each other,
                                                                                          # just correct for 1/distance
    a[n_atoms, n_atoms] = 0.0
    b = torch.zeros((n_atoms+1, 1), dtype=torch.float32, device=device)
    b[seq, 0] = - electronegativity
    b[n_atoms, 0] = total_charge
    charge = torch.linalg.solve(a, b)[:n_atoms,0]
    energy = (0.5 / sigma / torch.pi**0.5 * BOHR_TO_ANGSTROM * charge**2).sum() \
             + (charge[index[0,]] * charge[index[1,]] * torch.erf(distance * gamma) / distance * BOHR_TO_ANGSTROM).sum() # do not over count here
    return charge, energy.to(torch.float64)

def compute_charge_equilibration_batch(index, positions, hardness, electronegativity, sigma, total_charge):
    '''
    Compute the charge equilibration for a batch of structure
    positions, sigma has the unit of Angstrom, hardness has unit of 1/Angstrom
    The matrix is constructed to reproduce the charge in order, followed by the Lagrange multiplier and 0
    Return charge (float32) and energy (converted to float64)
    '''

    # Initialize
    n_batch, n_mask = index.shape
    device = positions.device
    a = torch.zeros((n_batch, n_mask+1, n_mask+1), dtype=torch.float32, device=device).flatten()
    b = torch.zeros((n_batch, n_mask+1), dtype=torch.float32, device=device).flatten()
    
    # Get number of atom each batch
    number_atoms = (index != -1).sum(dim=1)

    # Get batch correction for flatten the data
    batch_index = torch.repeat_interleave(number_atoms)
    batch_correct = batch_index * n_mask
    batch_correct_1d = batch_index * (n_mask+1)
    batch_correct_2d = batch_correct_1d * (n_mask+1)

    # a: diagonal hardness + sigma / pi**0.5

    run_index = torch.arange(n_mask, device=device).unsqueeze(0).repeat(n_batch, 1)
    mask = (run_index < number_atoms.unsqueeze(1))
    run_index = run_index[mask]
    
    selected_hardness = hardness.flatten()[batch_correct + run_index]
    selected_sigma = sigma.flatten()[batch_correct + run_index]
    diagonal_element = (selected_hardness + 1/(selected_sigma * torch.pi**0.5)) * BOHR_TO_ANGSTROM # For unit, watch non-batch version
    a[batch_correct_2d + run_index*(n_mask+1) + run_index] = diagonal_element
    diagonal_element_for_energy = 0.5 / selected_sigma / torch.pi**0.5 * BOHR_TO_ANGSTROM

    # b: electronegativy

    selected_electronegativity = electronegativity.flatten()[batch_correct + run_index]
    b[batch_correct_1d + run_index] = - selected_electronegativity

    # a: list of 1

    fix_index = torch.repeat_interleave(number_atoms, number_atoms)

    a[batch_correct_2d + fix_index * (n_mask + 1) + run_index] = 1.0
    a[batch_correct_2d + run_index * (n_mask + 1) + fix_index] = 1.0

    # a: off diagoal: torch.erf(distance * gamma) / distance

    pair_index = torch.tril_indices(n_mask, n_mask, -1, device=device).unsqueeze(1).repeat(1, n_batch, 1)
    number_pairs = number_atoms * (number_atoms -1) // 2
    mask = (torch.arange(pair_index.shape[2], device=device) < number_pairs.unsqueeze(1))
    pair_index = pair_index[:, mask]

    # Redefine batch correct for number of pairs
    pair_batch = torch.repeat_interleave(number_pairs) # show which batch the pair interaction in
    pair_correct = pair_batch * n_mask
    pair_correct_2d = pair_batch * (n_mask+1) * (n_mask+1)

    pos_index = pair_correct.unsqueeze(0).expand(2, -1) + pair_index

    vector = positions.flatten(end_dim=1).index_select(0, pos_index.view(-1)).view(2, -1, 3)
    distance = (vector[1,:,:] - vector[0,:,:]).norm(dim=1)
    selected_sigma = sigma.view(-1).index_select(0, pos_index.view(-1)).view(2, -1)
    gamma = 1 / (2 * (selected_sigma[0,:]**2 + selected_sigma[1,:]**2)) **0.5
    pair_value = torch.erf(distance * gamma) / distance * BOHR_TO_ANGSTROM
    a[pair_correct_2d + pair_index[0] * (n_mask+1) + pair_index[1]] = pair_value
    a[pair_correct_2d + pair_index[1] * (n_mask+1) + pair_index[0]] = pair_value

    # a: diagonal: 1 for larger

    n_extra = n_mask - number_atoms
    extra_correct = torch.repeat_interleave(n_extra) * (n_mask+1) * (n_mask+1)
    extra_run_index = torch.arange(n_mask+1, device=device).unsqueeze(0).repeat(n_batch, 1)
    mask = (extra_run_index >= number_atoms.unsqueeze(1)+1)
    extra_run_index = extra_run_index[mask]

    a[extra_correct + extra_run_index * (n_mask+1) + extra_run_index] = 1.0

    # b: Q total
    
    b[number_atoms + torch.arange(n_batch, device=device)*(n_mask+1)] = total_charge
    
    # Reshape and solve linear equation
    a = a.view((n_batch, n_mask+1, n_mask+1))
    b = b.view((n_batch, n_mask+1))
    x = torch.linalg.solve(a, b)
    
    # Return the charge
    final_charge = torch.zeros((n_batch*n_mask,), dtype=torch.float32, device=device)
    selected_charge = x.flatten()[batch_correct_1d + run_index]
    final_charge[batch_correct + run_index] = selected_charge

    # Calculate the energy

    energy = torch.zeros([n_batch,], dtype=torch.float32, device=device)
    energy.index_add_(0, batch_index, diagonal_element_for_energy * selected_charge**2) # Component of diagonal
    
    # The pair-wise interaction
    batch_pair_correct = cumsum_from_zero(number_atoms).index_select(0, pair_batch) # Atomic correct for batch of pair
    charge_index = pair_index + batch_pair_correct # Correct index order
    pair_charge = selected_charge.index_select(0, charge_index.flatten()).view(2,-1)
    energy.index_add_(0, pair_batch, pair_value * pair_charge.prod(dim=0))

    return final_charge.view((n_batch, n_mask)), energy.to(torch.float64)

class ChargeEquilibration(nn.Module):
    '''
    Charge equilibration model
    Need the neural network to predict the electronegativity, sigma and hardness
    Can perform the calculation with index, aev input, and total charge
    Can store the neural network needed (remember last layer to float32) and other variables
    '''
    def __init__(self):
        super().__init__()

    def set(self, neuralnetworks, hardness, sigma):
        self.neuralnetworks = neuralnetworks
        self.hardness = hardness
        self.sigma = sigma

    def compute(self, index, aev, positions, total_charge):
        hardness = self.hardness.compute(index)
        sigma = self.sigma.compute(index)
        electronegativity = self.neuralnetworks.compute(index, aev)
        return compute_charge_equilibration(positions, hardness, electronegativity, sigma, total_charge)
    
    def batch_compute(self, index, aev, positions, total_charge):
        hardness = self.hardness.batch_compute(index)
        sigma = self.sigma.batch_compute(index)
        electronegativity = self.neuralnetworks.batch_compute(index, aev)
        return compute_charge_equilibration_batch(index, positions, hardness, electronegativity, sigma, total_charge)

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)

    def load(self, data):
        self.neuralnetworks = IndexNetwork()
        self.neuralnetworks.load(data['neuralnetworks'])
        self.hardness = IndexValue(data['hardness'], True)
        self.sigma = IndexValue(data['sigma'], False)

    def dump(self):
        data = {}
        data['neuralnetworks'] = self.neuralnetworks.dump()
        data['hardness'] = self.hardness.values.detach().cpu()
        data['sigma'] = self.sigma.values.detach().cpu()
        return data
    