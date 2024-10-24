'''
Exchange-hole Dispersion Model
'''

import torch
from torch import nn
from .net import IndexValue, IndexNetwork
from .utils import create_double_index, create_double_index_batch, BOHR_TO_ANGSTROM

def dispersion_cut_off(distance, cut_off):
    '''
    Cut-off function for long-range
    The distance has to lower than cut_off distance
    The effect start at 0.66 cut_off distance by default
    '''
    rc = cut_off ** 2
    ro = 0.66**2 * rc
    r = distance **2
    return torch.where(
        r < ro,
        torch.ones(distance.shape, device=distance.device, dtype=distance.dtype),
        (rc - r)**2 * (rc + 2*r - 3*ro) / (rc-ro)**3
    )

class ExchangeHoleDispersion(nn.Module):
    '''
    Dispersion model following exchange-hole dispersion model
    '''
    def __init__(self):
        super().__init__()

    def set(self, m1_net, m2_net, m3_net, v_net, v_free, polar_free, critical_values, cut_off):
        self.m1_net = m1_net
        self.m2_net = m2_net
        self.m3_net = m3_net
        self.v_net = v_net 
        self.v_free = v_free
        self.polar_free = polar_free
        self.critical_values = critical_values.copy()
        self.cut_off = cut_off

    def load(self, data):
        self.m1_net = IndexNetwork()
        self.m1_net.load(data['m1_net'])
        self.m2_net = IndexNetwork()
        self.m2_net.load(data['m2_net'])
        self.m3_net = IndexNetwork()
        self.m3_net.load(data['m3_net'])
        self.v_net = IndexNetwork()
        self.v_net.load(data['v_net'])
        self.v_free = IndexValue(data['v_free'], False)
        self.polar_free = IndexValue(data['polar_free'], False)
        self.critical_values = data['critical_values'].copy()
        self.cut_off = data['cut_off']

    def dump(self):
        data = {}
        data['m1_net'] = self.m1_net.dump()
        data['m2_net'] = self.m2_net.dump()
        data['m3_net'] = self.m3_net.dump()
        data['v_net'] = self.v_net.dump()
        data['v_free'] = self.v_free.values.detach().cpu()
        data['polar_free'] = self.polar_free.values.detach().cpu()
        data['critical_values'] = self.critical_values.copy()
        data['cut_off'] = self.cut_off
        return data

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)

    def compute(self, atomic_index, aev, positions):
        m1 = self.m1_net.compute(atomic_index, aev)
        m2 = self.m2_net.compute(atomic_index, aev)
        m3 = self.m3_net.compute(atomic_index, aev)
        v = self.v_net.compute(atomic_index, aev)
        v_free = self.v_free.compute(atomic_index)
        polar_free = self.polar_free.compute(atomic_index)
        polar = polar_free * v / v_free
        index = create_double_index(positions, self.cut_off)
        vector = positions.index_select(0, index.view(-1)).view(2, -1, 3)
        distance = (vector[1,:,:] - vector[0,:,:]).norm(2, 1)
        m1_pair = m1.index_select(0, index.view(-1)).view(2, -1)
        m2_pair = m2.index_select(0, index.view(-1)).view(2, -1)
        m3_pair = m3.index_select(0, index.view(-1)).view(2, -1)
        polar_pair = polar.index_select(0, index.view(-1)).view(2, -1)
        scaled_m1 = m1_pair[0,:] / polar_pair[0,:] + m1_pair[1,:] / polar_pair[1,:]
        c6 = m1_pair[0,:] * m1_pair[1,:] / scaled_m1
        c8 = 1.5 * (m1_pair[0,:]*m2_pair[1,:] + m2_pair[0,:]*m1_pair[1,:]) / scaled_m1
        c10 = 2 * (m1_pair[0,:] * m3_pair[1,:] + m3_pair[0,:] * m1_pair[1,:] + \
                   2.1 * m2_pair[0,:] * m2_pair[1,:]) / scaled_m1
        r_critical = ((c8/c6)**0.5 + (c10/c6)**0.25 + (c10/c8)**0.5) / 3
        r_vdw = self.critical_values[0] + self.critical_values[1] * r_critical * BOHR_TO_ANGSTROM
                                                # Critical value is listed in Angstrom, but r_critical is bohr, the rvdw is in Angstrom
        cut_off_value = dispersion_cut_off(distance, self.cut_off)
        # The energy has the inclusion of damping function already
        energy_6 = -c6 / (distance ** 6 + r_vdw ** 6) * cut_off_value * BOHR_TO_ANGSTROM**6 # Change the distance unit from angstrom to bohr
        energy_8 = -c8 / (distance ** 8 + r_vdw ** 6) * cut_off_value * BOHR_TO_ANGSTROM**8
        energy_10 = -c10 / (distance ** 10 + r_vdw ** 10) * cut_off_value * BOHR_TO_ANGSTROM**10
        return energy_6.sum() + energy_8.sum() + energy_10.sum()

    def batch_compute(self, atomic_index, aev, positions):
        n_batch = positions.shape[0]
        m1 = self.m1_net.batch_compute(atomic_index, aev)
        m2 = self.m2_net.batch_compute(atomic_index, aev)
        m3 = self.m3_net.batch_compute(atomic_index, aev)
        v = self.v_net.batch_compute(atomic_index, aev)
        v_free = self.v_free.batch_compute(atomic_index)
        polar_free = self.polar_free.batch_compute(atomic_index)
        polar = polar_free * v / v_free
        index = create_double_index_batch(atomic_index, positions, self.cut_off)
        vector = positions.view(-1, 3).index_select(0, index.view(-1)).view(2, -1, 3)
        distance = (vector[1,:,:] - vector[0,:,:]).norm(2,1)
        m1_pair = m1.view(-1).index_select(0, index.view(-1)).view(2, -1)
        m2_pair = m2.view(-1).index_select(0, index.view(-1)).view(2, -1)
        m3_pair = m3.view(-1).index_select(0, index.view(-1)).view(2, -1)
        polar_pair = polar.view(-1).index_select(0, index.view(-1)).view(2, -1)
        scaled_m1 = m1_pair[0,:] / polar_pair[0,:] + m1_pair[1,:] / polar_pair[1,:]
        c6 = m1_pair[0,:] * m1_pair[1,:] / scaled_m1
        c8 = 1.5 * (m1_pair[0,:]*m2_pair[1,:] + m2_pair[0,:]*m1_pair[1,:]) / scaled_m1
        c10 = 2 * (m1_pair[0,:] * m3_pair[1,:] + m3_pair[0,:] * m1_pair[1,:] + \
                   2.1 * m2_pair[0,:] * m2_pair[1,:]) / scaled_m1
        r_critical = ((c8/c6)**0.5 + (c10/c6)**0.25 + (c10/c8)**0.5) / 3
        r_vdw = self.critical_values[0] + self.critical_values[1] * r_critical * BOHR_TO_ANGSTROM # See non-batch version
        cut_off_value = dispersion_cut_off(distance, self.cut_off)
        # The energy has the inclusion of damping function already
        energy_6 = -c6 / (distance ** 6 + r_vdw ** 6) * cut_off_value * BOHR_TO_ANGSTROM**6 # See non-batch version
        energy_8 = -c8 / (distance ** 8 + r_vdw ** 6) * cut_off_value * BOHR_TO_ANGSTROM**8
        energy_10 = -c10 / (distance ** 10 + r_vdw ** 10) * cut_off_value * BOHR_TO_ANGSTROM**10
        number_atoms = (index != -1).sum(dim=1)
        batch_index = torch.repeat_interleave(number_atoms)
        energy_6_batch = torch.zeros((n_batch,), dtype=torch.float32, device=positions.device)
        energy_6_batch.index_add_(dim=0, index=batch_index, source=energy_6)
        energy_8_batch = torch.zeros((n_batch,), dtype=torch.float32, device=positions.device)
        energy_8_batch.index_add_(dim=0, index=batch_index, source=energy_8)
        energy_10_batch = torch.zeros((n_batch,), dtype=torch.float32, device=positions.device)
        energy_10_batch.index_add_(dim=0, index=batch_index, source=energy_10)
        return energy_6_batch + energy_8_batch + energy_10_batch

    def compute_pbc(self, atomic_index, aev, positions, cell):
        pass

    def batch_compute_pbc(self, atomic_index, aev, positions, cell):
        pass

    

