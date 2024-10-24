'''
Handle the neural network needed in the package
+ IndexValue: for the look-up table with learnable or constant value
+ ShiftNetwork: a normal feed-forward network with last layer linear scaled by constant value
+ IndexNetwork: a look-up set of ShiftNetwork that depends on the index
'''

import torch
from torch import nn

class IndexValue(nn.Module):
    '''
    Create a selection style, where the index (element) turns into look-up value stored
    Have two version, with learnable=True for learning, and learnable=False for constant
    '''
    def __init__(self, values, learnable=False):
        super().__init__()
        self.learnable = learnable
        if self.learnable:
            if isinstance(values, torch.Tensor):
                self.values = nn.parameter.Parameter(values)
            else:
                self.values = nn.parameter.Parameter(torch.tensor(values))
        else:
            if isinstance(values, torch.Tensor):
                self.register_buffer('values', values)
            else:
                self.register_buffer('values', torch.tensor(values))

    def compute(self, index):
        n = index.shape[0]
        output = torch.zeros((n,), dtype=self.values.dtype, device=index.device)
        for i in range(self.values.shape[0]):
            mask = index == i
            output.masked_fill_(mask, self.values[i])
        return output

    def batch_compute(self, index):
        n_structure, n_atoms = index.shape
        output = torch.zeros((n_structure*n_atoms,), dtype=self.values.dtype, device=index.device)
        index_flatten = index.flatten()
        for i in range(self.values.shape[0]):
            mask = index_flatten == i
            output.masked_fill_(mask, self.values[i])
        return output.view(n_structure, n_atoms)

class ShiftedNetwork(nn.Module):
    '''
    The feed-forward neural network with output scaled and shifted by a pre-determined value
    Generally take in single-precision input, and output float32 or float64
    Do not sum up at the end
    Only support CELU, ELU and Tanh as the activation function, with linear at the last
    '''
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.layers = nn.Sequential()

    def init(self, dimensions, activations, shift, alpha, output_dtype=torch.float32):
        if self.initialized:
            raise ValueError('Already initialize')
        n = len(dimensions)
        assert len(activations) == n-1
        self.output_dtype = output_dtype
        self.dimensions = dimensions.copy()
        self.activations = activations.copy()
        for i in range(n-2):
            self.layers.append(nn.Linear(self.dimensions[i], self.dimensions[i+1]))
            if self.activations[i] == 'celu':
                self.layers.append(nn.CELU())
            elif self.activations[i] == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activations[i] == 'elu':
                self.layers.append(nn.ELU())
            else:
                raise ValueError(f'{self.activations[i]} is not supported')
        self.layers.append(nn.Linear(self.dimensions[-2], self.dimensions[-1]))
        self.register_buffer('shift', torch.tensor(shift, dtype=output_dtype))
        self.register_buffer('alpha', torch.tensor(alpha, dtype=output_dtype))
        self.initialized = True
    
    def forward(self, x):
        if not self.initialized:
            raise ValueError('The network has not been initialized')
        y = self.layers(x)
        y = y.to(self.output_dtype)
        return y * self.alpha + self.shift

class IndexNetwork(nn.Module):
    '''
    The set of shifted neural network with index to dictate which network to use
    '''
    def __init__(self):
        super().__init__()
    
    def init(self, dimensions, activations, shifts, alphas, output_dtype=torch.float32, sum_up=False):
        self.dimensions = dimensions.copy()
        self.activations = activations.copy()
        self.shifts = shifts.copy()
        self.alphas = alphas.copy()
        self.output_dtype = output_dtype
        self.sum_up = sum_up
        n = len(self.dimensions)
        self.networks = nn.ModuleList()
        for i in range(n):
            net = ShiftedNetwork()
            net.init(dimensions[i], activations[i], shifts[i], alphas[i], output_dtype)
            self.networks.append(net)

    def compute(self, index, aev):
        n = index.shape[0]
        if self.sum_up:
            output = 0.0
            for i in range(len(self.dimensions)):
                mask = index == i
                output = output + self.networks[i](aev[mask,:]).sum()
            return output
        else:
            output = torch.zeros((n,), dtype=self.output_dtype, device=aev.device)
            for i in range(len(self.dimensions)):
                mask = index == i
                tmp = self.networks[i](aev[mask,:])
                output.masked_scatter_(mask, tmp)
            return output

    def batch_compute(self, index, aev):
        n_structure, _ , n_aev = aev.shape
        index_flat = index.view(-1)
        aev_flat = aev.view(-1, n_aev)
        output = torch.zeros((index_flat.shape[0],), dtype=self.output_dtype, device=aev.device)
        for i in range(len(self.dimensions)):
            mask = index_flat == i
            tmp = self.networks[i](aev_flat[mask,:])
            output.masked_scatter_(mask, tmp)
        if self.sum_up:
            return output.view(n_structure, -1).sum(dim=1)
        else:
            return output.view(n_structure, -1)

    def write(self, file_name):
        params = {'dimensions': self.dimensions, 'activations': self.activations, 'shifts': self.shifts, 
                  'alphas': self.alphas, 'output_dtype': self.output_dtype, 'sum_up': self.sum_up}
        weights = self.networks.state_dict()
        data = {'params': params, 'weights': weights}
        torch.save(data, file_name)

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        params = data['params']
        self.init(params['dimensions'], params['activations'], params['shifts'], params['alphas'], params['output_dtype'], params['sum_up'])
        self.networks.load_state_dict(data['weights'])

    def load(self, data):
        params = data['params']
        self.init(params['dimensions'], params['activations'], params['shifts'], params['alphas'], params['output_dtype'], params['sum_up'])
        self.networks.load_state_dict(data['weights'])

    def dump(self):
        params = {'dimensions': self.dimensions, 'activations': self.activations, 'shifts': self.shifts, 
                  'alphas': self.alphas, 'output_dtype': self.output_dtype, 'sum_up': self.sum_up}
        weights = self.networks.state_dict()
        data = {'params': params, 'weights': weights}
        return data
    

class NetworkEnsemble(nn.Module):
    '''
    
    '''
    def __init__(self):
        super().__init__()

    def set(self, network_list):
        self.network_list = nn.ModuleList()
        for network in network_list:
            self.network_list.append(network)

    def compute(self, index, aev, module_index=-1):
        if module_index == -1:
            y = [net.compute(index, aev) for net in self.network_list]
            y = torch.stack(y, dim=0).mean(dim=0)
            return y
        else:
            return self.network_list[module_index].compute(index, aev)

    def std(self, index, aev, correction=0):
        y = [net.compute(index, aev) for net in self.network_list]
        y = torch.stack(y, dim=0).std(dim=0, correction=correction)
        return y

    def batch_compute(self, index, aev, module_index=-1):
        if module_index == -1:
            y = [net.batch_compute(index, aev) for net in self.network_list]
            y = torch.stack(y, dim=0).mean(dim=0)
            return y
        else:
            return self.network_list[module_index].batch_compute(index, aev)

    def batch_std(self, index, aev, correction=0):
        y = [net.batch_compute(index, aev) for net in self.network_list]
        y = torch.stack(y, dim=0).std(dim=0, correction=correction)
        return y

    def load(self, data):
        n = len(data)
        self.network_list = nn.ModuleList()
        for i in range(n):
            net = IndexNetwork()
            net.load(data[i])
            self.network_list.append(net)

    def dump(self):
        data = []
        for network in self.network_list:
            data.append(network.dump())
        return data

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)
        
