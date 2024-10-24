'''
Contains commonly used function, shared function or unit conversion
'''

import torch

# Conversion

PLANCK_CONSTANT = 6.62607015e-34
ELEMENTARY_CHARGE = 1.602176634e-19
BOLTZMANN_CONSTANT = 1.380649e-23
AVOGADRO_CONSTANT = 6.02214076e23
SPEED_OF_LIGHT = 2.99792458e8

KCAL_TO_JOULE = 4.184e3
PI = torch.pi

HARTREE_TO_KCALMOL = 4.3597447222060e-18 * AVOGADRO_CONSTANT / KCAL_TO_JOULE
BOHR_TO_ANGSTROM = 0.529177210545
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
AMU_TO_KG = 1.660539040e-27

# Commonly used function

def compute_shift_expansion(cell, cut_off):
    '''
    Return the number of expansion times to cover all the interaction inside of cut_off
    '''
    cell_inv = torch.linalg.pinv(cell)
    surface_dist_inv = cell_inv.norm(2,dim=-2)
    n_expansion = torch.ceil(cut_off * surface_dist_inv).to(torch.long)
    return n_expansion

def compute_half_shift(cell, cut_off):
    '''
    Compute all the shift possible given the cell and cut_off
    Similar to TorchANI, except eliminate the pbc
    For symmetry function computation
    '''
    n_expansion = compute_shift_expansion(cell, cut_off)
    r1 = torch.arange(1, n_expansion[0].item() + 1, device=cell.device)
    r2 = torch.arange(1, n_expansion[1].item() + 1, device=cell.device)
    r3 = torch.arange(1, n_expansion[2].item() + 1, device=cell.device)
    o = torch.zeros(1, dtype=torch.long, device=cell.device)
    return torch.cat([
        torch.cartesian_prod(r1, r2, r3),
        torch.cartesian_prod(r1, r2, o),
        torch.cartesian_prod(r1, r2, -r3),
        torch.cartesian_prod(r1, o, r3),
        torch.cartesian_prod(r1, o, o),
        torch.cartesian_prod(r1, o, -r3),
        torch.cartesian_prod(r1, -r2, r3),
        torch.cartesian_prod(r1, -r2, o),
        torch.cartesian_prod(r1, -r2, -r3),
        torch.cartesian_prod(o, r2, r3),
        torch.cartesian_prod(o, r2, o),
        torch.cartesian_prod(o, r2, -r3),
        torch.cartesian_prod(o, o, r3),
    ])

def compute_shift(cell, cut_off):
    '''
    Similar to compute shift half, but expanded for all the image cell
    For the energy interaction
    '''
    n_expansion = compute_shift_expansion(cell, cut_off)
    r1 = torch.arange(1, n_expansion[0].item() + 1, device=cell.device)
    r2 = torch.arange(1, n_expansion[1].item() + 1, device=cell.device)
    r3 = torch.arange(1, n_expansion[2].item() + 1, device=cell.device)
    o = torch.zeros(1, dtype=torch.long, device=cell.device)
    # Return the expansion from any direction
    return torch.cat([
        torch.cartesian_prod(r1, r2, r3),
        torch.cartesian_prod(r1, r2, o),
        torch.cartesian_prod(r1, r2, -r3),
        torch.cartesian_prod(r1, o, r3),
        torch.cartesian_prod(r1, o, o),
        torch.cartesian_prod(r1, o, -r3),
        torch.cartesian_prod(r1, -r2, r3),
        torch.cartesian_prod(r1, -r2, o),
        torch.cartesian_prod(r1, -r2, -r3),

        torch.cartesian_prod(o, r2, r3),
        torch.cartesian_prod(o, r2, o),
        torch.cartesian_prod(o, r2, -r3),
        torch.cartesian_prod(o, o, r3),
        # torch.cartesian_prod(o, o, o), # Not self-reflect
        torch.cartesian_prod(o, o, -r3),
        torch.cartesian_prod(o, -r2, r3),
        torch.cartesian_prod(o, -r2, o),
        torch.cartesian_prod(o, -r2, -r3),

        torch.cartesian_prod(-r1, r2, r3),
        torch.cartesian_prod(-r1, r2, o),
        torch.cartesian_prod(-r1, r2, -r3),
        torch.cartesian_prod(-r1, o, r3),
        torch.cartesian_prod(-r1, o, o),
        torch.cartesian_prod(-r1, o, -r3),
        torch.cartesian_prod(-r1, -r2, r3),
        torch.cartesian_prod(-r1, -r2, o),
        torch.cartesian_prod(-r1, -r2, -r3)
    ])


def cumsum_from_zero(input_):
    '''
    Create a cumsum, but start from 0 and omit the last
    Use in create_triple_index of symmetryfunction and charge
    '''
    cumsum = torch.zeros_like(input_)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum

def create_element_encoder(atomic_number_list, device=None):
    '''
    Return the encoder to change the atomic_number to index
    atomic_number_list should be list or tensor
    encoder[-1] will results in -1 (last member)
    '''
    if device is None:
        device = torch.device('cpu')
    element_encoder = torch.full((100,), -1, dtype=torch.long, device=device)
    for index, atomic_number in enumerate(atomic_number_list):
        element_encoder[atomic_number] = index
    return element_encoder

def create_double_index(positions, cut_off):
    '''
    Create all pair of index up to the number of atoms in positions
    The function eliminate the pair outside of the cut_off
    The index does not create the backward to positions
    '''
    pos = positions.detach()
    n = positions.shape[0]
    index = torch.triu_indices(n, n, 1, device=positions.device)
    # Use index_select to reduce number of kernel
    selected_positions = pos.index_select(0, index.view(-1)).view(2, -1, 3)
    distance = (selected_positions[1] - selected_positions[0]).norm(2, -1)
    # Use nonzero and index_select instead of index[:,good_mask] because it's faster
    good_index = (distance < cut_off).nonzero().flatten()
    return index.index_select(1, good_index)

def create_double_index_batch(atomic_index, positions, cut_off):
    '''
    Create all pair of index up to the number of atoms in atomic_index
    The function eliminate the pair outside of the cut_off
    atomic_index has to have -1 to indicate the mask location
    '''
    # Set the positions masking position to NaN to be eliminate in nonzero
    pos = positions.detach().masked_fill(atomic_index.unsqueeze(-1)==-1, torch.nan)
    device =  positions.device
    n_structure, n_atoms, _ = positions.shape
    index = torch.triu_indices(n_atoms, n_atoms, 1, device=device)
    selected_positions = pos.index_select(1, index.view(-1)).view(n_structure, 2, -1, 3)
    distance = (selected_positions[:,1,:,:] - selected_positions[:,0,:,:]).norm(2, -1)
    # Get all the index where distance is less than cut_off
    in_index = (distance < cut_off).nonzero()
    # Split the structure (row) and which pair index it is
    structure_index, pair_index = in_index.unbind(1)
    # Scale up the structure index, so that it works with flatten positions [n_structure*n_atoms, 3]
    # Return the index correspondance to pair_index
    return structure_index * n_atoms + index[:, pair_index]
