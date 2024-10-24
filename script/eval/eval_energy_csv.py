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
    parser.add_argument('-t', '--type', default='short')
    parser.add_argument('-d', '--data')
    parser.add_argument('-n', '--fieldname', default='energy')
    parser.add_argument('-o', '--output', default='output.csv')
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
    dat = pd.read_csv(args.data, index_col=0)
    result = pd.DataFrame(index=dat.index, columns=[args.fieldname])
    for index in tqdm(dat.index):
        atomic_numbers = dat.loc[index, 'atomic_numbers']
        atomic_numbers = np.array([int(x) for x in atomic_numbers.split()])
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
        positions = dat.loc[index, 'coordinates']
        positions = np.array([float(x) for x in positions.split()]).reshape((-1, 3))
        positions = torch.tensor(positions, dtype=torch.float32)
        result.loc[index, args.fieldname] = model.compute(atomic_numbers, positions).item() * HARTREE_TO_KCALMOL
    result.to_csv(args.output)

if __name__ == '__main__':
    main()