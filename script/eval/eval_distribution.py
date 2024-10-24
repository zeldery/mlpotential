import argparse
import torch
from tqdm import tqdm
from mlpotential.combine import *
from mlpotential.dataloader import DataIterator

def get_argument():
    parser = argparse.ArgumentParser('distribution')
    parser.add_argument('-m', '--model', default='model.pt')
    parser.add_argument('-d', '--data', default='data.pt')
    parser.add_argument('-t', '--type', default='short')
    parser.add_argument('-o', '--output', default='output.txt')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
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
    model.neural_network.sum_up = False
    scanner = DataIterator(args.data, ['atomic_numbers', 'coordinates'])
    loader = scanner.dataloader()
    f = open(args.output, 'w')
    for data in tqdm(loader):
        atomic_numbers = data['atomic_numbers'].to(torch.int64)
        positions = data['coordinates'].to(torch.float32)
        energies = model.batch_compute(atomic_numbers, positions)
        atomic_temp = atomic_numbers.flatten()
        energies_temp = energies.flatten()
        for i in [1, 6, 7, 8]:
            f.write(f'**{i}\n')
            tmp = energies_temp[atomic_temp == i].detach().numpy()
            for x in tmp:
                f.write(f'{x}\n')
    f.close()

if __name__ == '__main__':
    main()
