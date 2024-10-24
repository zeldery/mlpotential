import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from mlpotential.combine import *
from mlpotential.utils import HARTREE_TO_KCALMOL

def get_argument():
    parser = argparse.ArgumentParser('evaluation')
    parser.add_argument('-m', '--model')
    parser.add_argument('-t', '--type', default='charge')
    parser.add_argument('-d', '--data')
    parser.add_argument('-n', '--chargename', default='charge')
    parser.add_argument('-e', '--energyname', default='energy')
    parser.add_argument('-o', '--output', default='output.csv')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
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
    dat = pd.read_csv(args.data, index_col=0)
    result = pd.DataFrame(index=dat.index, columns=[args.energyname, args.chargename])
    for index in tqdm(dat.index):
        atomic_numbers = dat.loc[index, 'atomic_numbers']
        atomic_numbers = np.array([int(x) for x in atomic_numbers.split()])
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
        positions = dat.loc[index, 'coordinates']
        positions = np.array([float(x) for x in positions.split()]).reshape((-1, 3))
        positions = torch.tensor(positions, dtype=torch.float32)
        charge = model.compute_charge(atomic_numbers, positions).detach().cpu().numpy()
        charge = [str(x) for x in charge]
        result.loc[index, args.chargename] = ' '.join(charge)
        result.loc[index, args.energyname] = model.compute_energy(atomic_numbers, positions).item() * HARTREE_TO_KCALMOL
    result.to_csv(args.output)

if __name__ == '__main__':
    main()
    