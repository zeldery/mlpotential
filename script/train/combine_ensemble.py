from mlpotential.combine import ShortRangeModel, ShortRangeEnsembleModel
from mlpotential.net import IndexNetwork, NetworkEnsemble
import torch


def main():
    lst_checkpoint = [f'../energy_ensemble/{x}/checkpoint.pt' for x in range(8)]
    net_list = []
    for name in lst_checkpoint:
        chk = torch.load(name, map_location='cpu')
        model = ShortRangeModel()
        model.load(chk['best_model'])
        net_list.append(model.neural_network)
    ensemble = NetworkEnsemble()
    ensemble.set(net_list)
    new_model = ShortRangeEnsembleModel()
    new_model.set(model.element_list, model.symmetry_function, ensemble)
    new_model.write('energy_ensemble_model.pt')


if __name__ == '__main__':
    main()