import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from mlpotential.dataloader import DataIterator
from mlpotential.combine import *

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='model.pt')
    parser.add_argument('-t', '--type', default='charge')
    parser.add_argument('-e', '--element')
    parser.add_argument('-d', '--data', default='data.h5')
    parser.add_argument('-g', '--gpu', default='0')
    parser.add_argument('-o', '--output', default='output.csv')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
    scanner = DataIterator(args.data, ['atomic_numbers', 'coordinates', 'q'])
    loader = scanner.dataloader(shuffle=True)
    if args.type == 'charge':
        model = ChargeModel()
    elif args.type == 'charge_ensemble':
        model = ChargeEnsembleModel()
    elif args.type == 'charge_dispersion':
        model = ChargeDispersionModel()
    elif args.type == 'charge_dispersion_ensemble':
        model = ChargeDispersionEnsembleModel()
    else:
        raise ValueError(f'Incorrect type {args.type}')
    model.read(args.model)
    element = int(args.element)
    if args.gpu == '0':
        device = torch.device('cpu')
    elif args.gpu == '1':
        device = torch.device('cuda')
    else:
        raise ValueError(f'Invalid gpu argument {args.gpu}')
    model = model.to(device)
    scanner.mode = 'all'
    total_atoms = 0
    with torch.no_grad():
        for data in loader:
            total_atoms += (data['atomic_numbers'] == element).sum().item()
        output = pd.DataFrame(columns=['train', 'reference', 'predict'], index=np.arange(total_atoms))
        current = 0
        scanner.mode = 'train'
        for data in tqdm(loader, desc='Train'):
            # WARNING: loc in pandas includes the last index
            atomic_numbers = data['atomic_numbers'].to(torch.int64).to(device)
            positions = data['coordinates'].to(torch.float32).to(device)
            n = (data['atomic_numbers'] == element).sum().item()
            ref = data['q']
            pred = model.batch_compute_charge(atomic_numbers, positions)
            index = (atomic_numbers.flatten() == element)
            output.loc[current:(current+n-1), 'train'] = 1
            output.loc[current:(current+n-1), 'reference'] = ref.flatten()[index.cpu()].detach().numpy()
            output.loc[current:(current+n-1), 'predict'] = pred.flatten()[index].detach().cpu().numpy()
            current += n
        scanner.mode = 'test'
        for data in tqdm(loader, desc='Test'):
            atomic_numbers = data['atomic_numbers'].to(torch.int64).to(device)
            positions = data['coordinates'].to(torch.float32).to(device)
            n = (data['atomic_numbers'] == element).sum().item()
            ref = data['q']
            pred = model.batch_compute_charge(atomic_numbers, positions)
            index = (atomic_numbers.flatten() == element)
            output.loc[current:(current+n-1), 'train'] = 0
            output.loc[current:(current+n-1), 'reference'] = ref.flatten()[index.cpu()].detach().numpy()
            output.loc[current:(current+n-1), 'predict'] = pred.flatten()[index].detach().cpu().numpy()
            current += n
    output.to_csv(args.output)


if __name__ == '__main__':
    main()