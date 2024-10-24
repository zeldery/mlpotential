'''
The file contains final model for multiple neural network potential, depends on what is included in
Current support:
 + traditional short-range neural network potential
 + Charge equilibration neural network potential
'''

from .net import IndexNetwork, NetworkEnsemble
from .sf import SymmetryFunction
from .charge import ChargeEquilibration
from .dispersion import ExchangeHoleDispersion
from .utils import create_element_encoder
import torch
from torch import nn

class ShortRangeModel(nn.Module):
    '''
    The model contain short-range neural network potential
    '''
    def __init__(self):
        super().__init__()

    def set(self, element_list, symmetry_function, neural_network):
        self.element_list = element_list.copy()
        self.symmetry_function = symmetry_function
        self.neural_network = neural_network

    def read(self, file_name):
        data = torch.load(file_name, map_location='cpu', weights_only=True)
        self.load(data)

    def write(self, file_name):
        torch.save(self.dump(), file_name)

    def load(self, data):
        self.symmetry_function = SymmetryFunction()
        self.symmetry_function.load(data['symmetry_function'])
        self.neural_network = IndexNetwork()
        self.neural_network.load(data['neural_network'])
        self.element_list = data['element_list'].copy()

    def dump(self):
        return {'element_list': self.element_list, 'symmetry_function': self.symmetry_function.dump(), 
                'neural_network': self.neural_network.dump()}
    
    def compute(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        return self.neural_network.compute(atomic_index, aev)
    
    def batch_compute(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder.index_select(0, atomic_numbers.view(-1)).view(atomic_numbers.shape[0], atomic_numbers.shape[1])
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        return self.neural_network.batch_compute(atomic_index, aev)

    def compute_pbc(self, atomic_numbers, positions, cell):
        pass

    def batch_compute_pbc(self, atomic_numbers, positions, cell):
        pass

class ShortRangeEnsembleModel(ShortRangeModel):
    '''
    
    '''
    def __init__(self):
        super().__init__()

    def load(self, data):
        self.symmetry_function = SymmetryFunction()
        self.symmetry_function.load(data['symmetry_function'])
        self.neural_network = NetworkEnsemble()
        self.neural_network.load(data['network_ensemble'])
        self.element_list = data['element_list'].copy()

    def dump(self):
        return {'element_list': self.element_list, 'symmetry_function': self.symmetry_function.dump(), 
                'network_ensemble': self.neural_network.dump()}

class ChargeModel(nn.Module):
    '''
    The model contain 4th generation neural network potential
    '''
    def __init__(self):
        super().__init__()

    def set(self, element_list, symmetry_function, charge_model, short_network):
        self.element_list = element_list.copy()
        self.symmetry_function = symmetry_function
        self.charge_model = charge_model
        self.short_network = short_network

    def compute_charge(self, atomic_numbers, positions, total_charge=None):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        if total_charge is None:
            total_charge = 0.0
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        charge, _ = self.charge_model.compute(atomic_index, aev, positions, total_charge)
        return charge

    def batch_compute_charge(self, atomic_numbers, positions, total_charge=None):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        if total_charge is None:
            total_charge = torch.zeros(atomic_numbers.shape[0], dtype=torch.float32, device=positions.device)
        atomic_index = encoder.index_select(0, atomic_numbers.view(-1)).view(atomic_numbers.shape[0], atomic_numbers.shape[1])
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        charge, _ = self.charge_model.batch_compute(atomic_index, aev, positions, total_charge)
        return charge

    def compute_energy(self, atomic_numbers, positions, total_charge=None):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        if total_charge is None:
            total_charge = 0.0
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        _ , energy = self.charge_model.compute(atomic_index, aev, positions, total_charge)
        return energy

    def batch_compute_energy(self, atomic_numbers, positions, total_charge=None):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        if total_charge is None:
            total_charge = torch.zeros(atomic_numbers.shape[0], dtype=torch.float32, device=positions.device)
        atomic_index = encoder.index_select(0, atomic_numbers.view(-1)).view(atomic_numbers.shape[0], atomic_numbers.shape[1])
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        _ , energies = self.charge_model.batch_compute(atomic_index, aev, positions, total_charge)
        return energies

    def compute(self, atomic_numbers, positions, total_charge=None):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        if total_charge is None:
            total_charge = 0.0
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        _ , charge_energy = self.charge_model.compute(atomic_index, aev, positions, total_charge)
        short_energy = self.short_network.compute(atomic_index, aev)
        return charge_energy + short_energy

    def batch_compute(self, atomic_numbers, positions, total_charge=None):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        if total_charge is None:
            total_charge = torch.zeros(atomic_numbers.shape[0], dtype=torch.float32, device=positions.device)
        atomic_index = encoder.index_select(0, atomic_numbers.view(-1)).view(atomic_numbers.shape[0], atomic_numbers.shape[1])
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        _ , charge_energy = self.charge_model.batch_compute(atomic_index, aev, positions, total_charge)
        short_energy = self.short_network.batch_compute(atomic_index, aev)
        return charge_energy + short_energy

    def write(self, file_name):
        torch.save(self.dump(), file_name)

    def read(self, file_name):
        data = torch.load(file_name, map_location='cpu', weights_only=True)
        self.load(data)

    def load(self, data):
        self.element_list = data['element_list'].copy()
        self.symmetry_function = SymmetryFunction()
        self.symmetry_function.load(data['symmetry_function'])
        self.charge_model = ChargeEquilibration()
        self.charge_model.load(data['charge_model'])
        self.short_network = IndexNetwork()
        self.short_network.load(data['short_network'])

    def dump(self):
        return {'element_list': self.element_list.copy(), 'symmetry_function': self.symmetry_function.dump(),
                'charge_model': self.charge_model.dump(), 'short_network': self.short_network.dump()}

class ChargeEnsembleModel(ChargeModel):
    '''
    
    '''
    def __init__(self):
        super().__init__()

    def load(self, data):
        self.element_list = data['element_list'].copy()
        self.symmetry_function = SymmetryFunction()
        self.symmetry_function.load(data['symmetry_function'])
        self.charge_model = ChargeEquilibration()
        self.charge_model.load(data['charge_model'])
        self.short_network = NetworkEnsemble()
        self.short_network.load(data['short_ensemble'])

    def dump(self):
        return {'element_list': self.element_list.copy(), 'symmetry_function': self.symmetry_function.dump(),
                'charge_model': self.charge_model.dump(), 'short_ensemble': self.short_network.dump()}

class DispersionModel(nn.Module):
    '''
    
    '''
    def __init__(self):
        super().__init__()

    def set(self, element_list, symmetry_function, dispersion_model, short_network):
        self.element_list = element_list.copy()
        self.symmetry_function = symmetry_function
        self.dispersion_model = dispersion_model
        self.short_network = short_network

    def compute(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        return self.dispersion_model.compute(atomic_index, aev, positions) \
                + self.short_network.compute(atomic_index, aev)
    
    def batch_compute(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        return self.dispersion_model.batch_compute(atomic_index, aev, positions) \
                + self.short_network.batch_compute(atomic_index, aev)

    def compute_pbc(self, atomic_numbers, positions, cell):
        pass

    def batch_compute_pbc(self, atomic_numbers, positions, cell):
        pass

    def compute_short(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        return self.short_network.compute(atomic_index, aev)
    
    def batch_compute_short(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        return self.short_network.batch_compute(atomic_index, aev)

    def compute_dispersion(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        return self.dispersion_model.compute(atomic_index, aev, positions)
    
    def batch_compute_dispersion(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        return self.dispersion_model.batch_compute(atomic_index, aev, positions)

    def compute_m1(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        return self.dispersion_model.m1_net.compute(atomic_index, aev)

    def compute_m2(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        return self.dispersion_model.m2_net.compute(atomic_index, aev)

    def compute_m3(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        return self.dispersion_model.m3_net.compute(atomic_index, aev)

    def compute_v(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        return self.dispersion_model.v_net.compute(atomic_index, aev)

    def batch_compute_m1(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        return self.dispersion_model.m1_net.batch_compute(atomic_index, aev)

    def batch_compute_m2(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        return self.dispersion_model.m2_net.batch_compute(atomic_index, aev)

    def batch_compute_m3(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        return self.dispersion_model.m3_net.batch_compute(atomic_index, aev)

    def batch_compute_v(self, atomic_numbers, positions):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        return self.dispersion_model.v_net.batch_compute(atomic_index, aev)

    def load(self, data):
        self.element_list = data['element_list'].copy()
        self.symmetry_function = SymmetryFunction()
        self.symmetry_function.load(data['symmetry_function'])
        self.dispersion_model = ExchangeHoleDispersion()
        self.dispersion_model.load(data['dispersion_model'])
        self.short_network = IndexNetwork()
        self.short_network.load(data['short_network'])

    def dump(self):
        return {'element_list': self.element_list.copy(), 'symmetry_function': self.symmetry_function.dump(),
                'dispersion_model': self.dispersion_model.dump(), 'short_network': self.short_network.dump()}

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)

class DispersionEnsembleModel(DispersionModel):
    '''
    
    '''
    def __init__(self):
        super().__init__()

    def load(self, data):
        self.element_list = data['element_list'].copy()
        self.symmetry_function = SymmetryFunction()
        self.symmetry_function.load(data['symmetry_function'])
        self.dispersion_model = ExchangeHoleDispersion()
        self.dispersion_model.load(data['dispersion_model'])
        self.short_network = NetworkEnsemble()
        self.short_network.load(data['short_ensemble'])

    def dump(self):
        return {'element_list': self.element_list.copy(), 'symmetry_function': self.symmetry_function.dump(),
                'dispersion_model': self.dispersion_model.dump(), 'short_ensemble': self.short_network.dump()}

class ChargeDispersionModel(nn.Module):
    '''
    
    '''
    def __init__(self):
        super().__init__()

    def set(self, element_list, symmetry_function, charge_model, dispersion_model, short_network):
        self.element_list = element_list.copy()
        self.symmetry_function = symmetry_function
        self.charge_model = charge_model
        self.dispersion_model = dispersion_model
        self.short_network = short_network

    def compute(self, atomic_numbers, positions, total_charge=None):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        if total_charge is None:
            total_charge = 0.0
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.compute(atomic_index, positions)
        _ , charge_energy = self.charge_model.compute(atomic_index, aev, positions, total_charge)
        dispersion_energy = self.dispersion_model.compute(atomic_index, aev, positions)
        short_energy = self.short_network.compute(atomic_index, aev)
        return short_energy + dispersion_energy + charge_energy
    
    def batch_compute(self, atomic_numbers, positions, total_charge=None):
        encoder = create_element_encoder(self.element_list, device=positions.device)
        if total_charge is None:
            total_charge = torch.zeros(atomic_numbers.shape[0], dtype=torch.float32, device=positions.device)
        atomic_index = encoder[atomic_numbers]
        aev = self.symmetry_function.batch_compute(atomic_index, positions)
        _ , charge_energy = self.charge_model.batch_compute(atomic_index, aev, positions, total_charge)
        dispersion_energy = self.dispersion_model.batch_compute(atomic_index, aev, positions)
        short_energy = self.short_network.batch_compute(atomic_index, aev)
        return short_energy + dispersion_energy + charge_energy

    def compute_pbc(self, atomic_numbers, positions, cell):
        pass

    def batch_compute_pbc(self, atomic_numbers, positions, cell):
        pass

    def load(self, data):
        self.element_list = data['element_list'].copy()
        self.symmetry_function = SymmetryFunction()
        self.symmetry_function.load(data['symmetry_function'])
        self.dispersion_model = ExchangeHoleDispersion()
        self.dispersion_model.load(data['dispersion_model'])
        self.charge_model = ChargeEquilibration()
        self.charge_model.load(data['charge_model'])
        self.short_network = IndexNetwork()
        self.short_network.load(data['short_network'])

    def dump(self):
        return {'element_list': self.element_list.copy(), 'symmetry_function': self.symmetry_function.dump(),
                'charge_model': self.charge_model.dump(), 'dispersion_model': self.dispersion_model.dump(),
                'short_network': self.short_network.dump()}

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)
        
class ChargeDispersionEnsembleModel(ChargeDispersionModel):
    '''
    
    '''
    def __init__(self):
        super().__init__()

    def load(self, data):
        self.element_list = data['element_list'].copy()
        self.symmetry_function = SymmetryFunction()
        self.symmetry_function.load(data['symmetry_function'])
        self.dispersion_model = ExchangeHoleDispersion()
        self.dispersion_model.load(data['dispersion_model'])
        self.charge_model = ChargeEquilibration()
        self.charge_model.load(data['charge_model'])
        self.short_network = NetworkEnsemble()
        self.short_network.load(data['short_ensemble'])

    def dump(self):
        return {'element_list': self.element_list.copy(), 'symmetry_function': self.symmetry_function.dump(),
                'charge_model': self.charge_model.dump(), 'dispersion_model': self.dispersion_model.dump(),
                'short_ensemble': self.short_network.dump()}
    