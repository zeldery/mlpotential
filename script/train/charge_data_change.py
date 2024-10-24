import torch
from mlpotential.combine import ChargeModel, ChargeEnsembleModel
from sklearn.linear_model import LinearRegression
import argparse
import numpy as np
import h5py
from tqdm import tqdm

def get_argument():
    parser = argparse.ArgumentParser('charge_adjust', 
                                     description='Adjust the data for charge training')
    parser.add_argument('-m', '--model', default='model.pt')
    parser.add_argument('-t', '--type', default='charge')
    parser.add_argument('-i', '--input', default='data.hdf5')
    parser.add_argument('-o', '--output', default='output.hdf5')
    parser.add_argument('-g', '--gpu', default='0')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
    if args.type == 'charge':
        model = ChargeModel()
    elif args.type == 'charge_ensemble':
        model = ChargeEnsembleModel()
    model.read(args.model)
    inp = h5py.File(args.input, 'r')
    if args.gpu =='0':
        device = torch.device('cpu')
    elif args.gpu == '1':
        device = torch.device('cuda')
    else:
        raise ValueError(f'Unvalid value for gpu argument of {args.gpu}')
    model = model.to(device)
    outp = h5py.File(args.output, 'w')
    n_train = len(inp['train'])
    n_test = len(inp['test'])
    outp.create_group('train')
    for i in tqdm(range(n_train), desc='Train'):
        outp.create_group(f'train/{i}')
        sub = inp[f'train/{i}']
        outp.copy(source=sub['atomic_numbers'], dest=outp[f'train/{i}'], name='atomic_numbers')
        outp.copy(source=sub['coordinates'], dest=outp[f'train/{i}'], name='coordinates')
        atomic_numbers = torch.tensor(np.array(sub['atomic_numbers']), dtype=torch.int64, device=device)
        positions = torch.tensor(np.array(sub['coordinates']), dtype=torch.float32, requires_grad=True, device=device)
        energies = model.batch_compute_energy(atomic_numbers, positions)
        forces = -torch.autograd.grad(energies, positions, torch.ones_like(energies))[0]
        e = np.array(sub['energies'], dtype=np.float64)
        f = np.array(sub['forces'], dtype=np.float32)
        f -= forces.detach().cpu().numpy()
        e -= energies.detach().cpu().numpy()
        outp.create_dataset(name=f'train/{i}/energies', data=e)
        outp.create_dataset(name=f'train/{i}/forces', data=f)
    outp.create_group('test')
    for i in tqdm(range(n_test), desc='Test'):
        outp.create_group(f'test/{i}')
        sub = inp[f'test/{i}']
        outp.copy(source=sub['atomic_numbers'], dest=outp[f'test/{i}'], name='atomic_numbers')
        outp.copy(source=sub['coordinates'], dest=outp[f'test/{i}'], name='coordinates')
        atomic_numbers = torch.tensor(np.array(sub['atomic_numbers']), dtype=torch.int64, device=device)
        positions = torch.tensor(np.array(sub['coordinates']), dtype=torch.float32, requires_grad=True, device=device)
        energies = model.batch_compute_energy(atomic_numbers, positions)
        forces = -torch.autograd.grad(energies, positions, torch.ones_like(energies))[0]
        e = np.array(sub['energies'], dtype=np.float64)
        f = np.array(sub['forces'], dtype=np.float32)
        f -= forces.detach().cpu().numpy()
        e -= energies.detach().cpu().numpy()
        outp.create_dataset(name=f'test/{i}/energies', data=e)
        outp.create_dataset(name=f'test/{i}/forces', data=f)

if __name__ == '__main__':
    main()
