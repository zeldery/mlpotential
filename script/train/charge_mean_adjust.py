import torch
from mlpotential.dataloader import DataIterator
from mlpotential.combine import ChargeModel
from sklearn.linear_model import LinearRegression
import argparse
import numpy as np

def get_argument():
    parser = argparse.ArgumentParser('charge_adjust', 
                                     description='Adjust the mean for the charge')
    parser.add_argument('-m', '--model', default='model.pt')
    parser.add_argument('-d', '--data', default='data.hdf5')
    parser.add_argument('-o', '--output', default='output_model.pt')
    parser.add_argument('-g', '--gpu', default='0')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
    model = ChargeModel()
    model.read(args.model)
    data = DataIterator(args.data, ['atomic_numbers', 'coordinates', 'energies'])
    loader = data.dataloader(shuffle=True)
    if args.gpu =='0':
        device = torch.device('cpu')
    elif args.gpu == '1':
        device = torch.device('cuda')
    else:
        raise ValueError(f'Unvalid value for gpu argument of {args.gpu}')
    model = model.to(device)
    e_lst = []
    total_lst = []
    element_list = [1,6,7,8]
    count = {}
    for element in element_list:
        count[element] = []
    for batch_data in loader:
        atomic_numbers = batch_data['atomic_numbers'].to(torch.int64).to(device)
        positions = batch_data['coordinates'].to(torch.float32).to(device)
        total_energies = batch_data['energies'].to(torch.float64).to(device)
        q_energies = model.batch_compute_energy(atomic_numbers, positions)
        energies = (total_energies - q_energies).detach().cpu().numpy()
        atomic = atomic_numbers.detach().cpu().numpy()
        e_lst += energies.tolist()
        total_lst += total_energies.detach().cpu().numpy().tolist()
        for element in element_list:
            count[element] += (atomic == element).sum(axis=1).tolist()
    tmp = []
    for element in element_list:
        tmp.append(np.array(count[element]))
    x = np.stack(tmp, 1)
    energies_model = LinearRegression(fit_intercept=False)
    energies_model.fit(x, e_lst)
    for coef in energies_model.coef_:
        print(coef, end=' ')
    print()
    # Change the model here
    tmp = model.dump()
    tmp['short_network']['params']['shifts'] = [x for x in energies_model.coef_]
    new_model = ChargeModel()
    new_model.load(tmp)
    new_model.write(args.output)
    # For reference
    total_model = LinearRegression(fit_intercept=False)
    total_model.fit(x, total_lst)
    for coef in total_model.coef_:
        print(coef, end=' ')
    print()

if __name__ == '__main__':
    main()
