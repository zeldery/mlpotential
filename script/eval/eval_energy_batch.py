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
    parser.add_argument('-t', '--type')
    parser.add_argument('-d', '--data', default='data.h5')
    parser.add_argument('-g', '--gpu', default='0')
    parser.add_argument('-o', '--output', default='output.csv')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
    scanner = DataIterator(args.data, ['atomic_numbers', 'coordinates', 'energies'])
    loader = scanner.dataloader(shuffle=True)
    if args.type == 'short':
        model = ShortRangeModel()
    elif args.type == 'short_ensemble':
        model = ShortRangeEnsembleModel()
    elif args.type == 'charge':
        model = ChargeModel()
    elif args.type == 'charge_ensemble':
        model = ChargeEnsembleModel()
    elif args.type == 'dispersion':
        model = DispersionModel()
    elif args.type == 'dispersion_ensemble':
        model = DispersionEnsembleModel()
    elif args.type == 'charge_dispersion':
        model = ChargeDispersionModel()
    elif args.type == 'charge_dispersion_ensemble':
        model = ChargeDispersionEnsembleModel()
    else:
        raise ValueError(f'Incorrect type {args.type}')
    model.read(args.model)
    if args.gpu == '0':
        device = torch.device('cpu')
    elif args.gpu == '1':
        device = torch.device('cuda')
    else:
        raise ValueError(f'Invalid gpu argument {args.gpu}')
    model = model.to(device)
    scanner.mode = 'all'
    total_structure = 0
    with torch.no_grad():
        for data in loader:
            total_structure += data['energies'].shape[0]
        output = pd.DataFrame(columns=['train', 'reference', 'predict'], index=np.arange(total_structure))
        current = 0
        scanner.mode = 'train'
        for data in tqdm(loader, desc='Train'):
            n = data['energies'].shape[0]
            # WARNING: loc in pandas includes the last index
            output.loc[current:(current+n-1), 'train'] = 1
            output.loc[current:(current+n-1), 'reference'] = data['energies'].detach().cpu().squeeze().numpy()
            atomic_numbers = data['atomic_numbers'].to(torch.int64).to(device)
            positions = data['coordinates'].to(torch.float32).to(device)
            predicted = model.batch_compute(atomic_numbers, positions).detach().cpu().squeeze().numpy()
            output.loc[current:(current+n-1), 'predict'] = predicted
            current += n
        scanner.mode = 'test'
        for data in tqdm(loader, desc='Test'):
            n = data['energies'].shape[0]
            output.loc[current:(current+n-1), 'train'] = 0
            output.loc[current:(current+n-1), 'reference'] = data['energies'].detach().cpu().squeeze().numpy()
            atomic_numbers = data['atomic_numbers'].to(torch.int64).to(device)
            positions = data['coordinates'].to(torch.float32).to(device)
            predicted = model.batch_compute(atomic_numbers, positions).detach().cpu().squeeze().numpy()
            output.loc[current:(current+n-1), 'predict'] = predicted
            current += n
    output.to_csv(args.output)


if __name__ == '__main__':
    main()