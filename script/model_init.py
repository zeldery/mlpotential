# Short-range

import torch
from net import IndexNetwork
from sf import SymmetryFunction
from combine import ShortRangeModel

symfunc = SymmetryFunction()
symfunc.set(4, [16.0], [0.9, 1.16875, 1.4375, 1.70625, 1.975, 2.24375,
            2.5125, 2.78125, 3.05, 3.31875, 3.5875, 3.85625,
            4.125, 4.39375, 4.6625, 4.93125], 5.2,
            [0.19634954, 0.58904862, 0.9817477, 1.3744468,
            1.7671459, 2.1598449, 2.552544, 2.9452431],
            [32.0],  [8.0], [0.9, 1.55, 2.2, 2.85], 3.5)

neuralnet = IndexNetwork()
neuralnet.init([[384, 160, 128, 96, 1], [384, 144, 112, 96, 1], [384, 128, 112, 96, 1], [384, 128, 112, 96, 1]], \
         [['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu']], \
         [-0.5973188989593174, -38.06347076062943, -54.68947219639185, -75.16868155337862], \
         [1.0, 1.0, 1.0, 1.0], torch.float64, True)

model = ShortRangeModel()
model.set([1, 6, 7, 8], symfunc, neuralnet)
model.write('ani_model.pt')

# Short-range with charge equilibration

from sf import SymmetryFunction
from net import IndexValue, IndexNetwork
from charge import ChargeEquilibration
from combine import ChargeModel

symfunc = SymmetryFunction()
symfunc.set(4, [16.0], [0.9, 1.16875, 1.4375, 1.70625, 1.975, 2.24375,
            2.5125, 2.78125, 3.05, 3.31875, 3.5875, 3.85625,
            4.125, 4.39375, 4.6625, 4.93125], 5.2,
            [0.19634954, 0.58904862, 0.9817477, 1.3744468,
            1.7671459, 2.1598449, 2.552544, 2.9452431],
            [32.0],  [8.0], [0.9, 1.55, 2.2, 2.85], 3.5)

short_net = IndexNetwork()
short_net.init([[384, 160, 128, 96, 1], [384, 144, 112, 96, 1], [384, 128, 112, 96, 1], [384, 128, 112, 96, 1]], \
                [['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu']], \
                [0.0, 0.0, 0.0, 0.0], \
                [1.0, 1.0, 1.0, 1.0], torch.float64, True) # Need to replace with better energy shift

charge_net = IndexNetwork()
charge_net.init([[384, 160, 96, 1], [384, 160, 96, 1], [384, 160, 96, 1], [384, 160, 96, 1]], \
                [['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu']], \
                [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], torch.float32, False)

charge_equil = ChargeEquilibration()
sigma = [0.31, 0.76, 0.71, 0.66] # Angstrom
sigma = IndexValue(sigma, False)
hardness = [6.3, 2.9, 3.4, 3.9]
hardness = IndexValue(hardness, True)
charge_equil.set(charge_net, hardness, sigma)

model = ChargeModel()
model.set([1, 6, 7, 8], symfunc, charge_equil, short_net)
model.write('runner_model.pt')

# MLXDM model

from sf import SymmetryFunction
from net import IndexNetwork, IndexValue
from dispersion import ExchangeHoleDispersion
from combine import DispersionModel

symfunc = SymmetryFunction()
symfunc.set(4, [16.0], [0.9, 1.16875, 1.4375, 1.70625, 1.975, 2.24375,
            2.5125, 2.78125, 3.05, 3.31875, 3.5875, 3.85625,
            4.125, 4.39375, 4.6625, 4.93125], 5.2,
            [0.19634954, 0.58904862, 0.9817477, 1.3744468,
            1.7671459, 2.1598449, 2.552544, 2.9452431],
            [32.0],  [8.0], [0.9, 1.55, 2.2, 2.85], 3.5)

short_net = IndexNetwork()
short_net.init([[384, 160, 128, 96, 1], [384, 144, 112, 96, 1], [384, 128, 112, 96, 1], [384, 128, 112, 96, 1]], \
                [['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu']], \
                [0.0, 0.0, 0.0, 0.0], \
                [1.0, 1.0, 1.0, 1.0], torch.float64, True)

m1_net = IndexNetwork()
m1_net.init([[384, 144, 96, 1], [384, 144, 96, 1], [384, 144, 96, 1], [384, 144, 96, 1]], \
            [['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu']], \
            [1.5476255223478212, 4.322816284605242, 4.659640969237404, 4.841779646534], \
            [0.12243342204400418, 0.40875807265603364, 0.5465825133744416, 0.404868962115141], torch.float32, False)

m2_net = IndexNetwork()
m2_net.init([[384, 144, 96, 1], [384, 144, 96, 1], [384, 144, 96, 1], [384, 144, 96, 1]], \
            [['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu']], \
            [12.553692350335549, 54.948165624034274, 45.28593689285579, 37.3829639825235], \
            [1.1532345620529276, 4.962789635070926, 5.692156022885587, 3.787773384647434], torch.float32, False)

m3_net = IndexNetwork()
m3_net.init([[384, 144, 96, 1], [384, 144, 96, 1], [384, 144, 96, 1], [384, 144, 96, 1]], \
            [['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu']], \
            [209.422031043635, 987.0110684459416, 596.760233961183, 382.59123549642015], \
            [26.974767172514678, 105.90113739290175, 75.77879922230893, 43.76511962041938], torch.float32, False)

v_net = IndexNetwork()
v_net.init([[384, 144, 96, 1], [384, 144, 96, 1], [384, 144, 96, 1], [384, 144, 96, 1]], \
            [['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu']], \
            [6.050941007535366, 31.570349038828038, 26.203838979591527, 21.844255175123205], \
            [0.38810715373608695, 1.0409411278052347, 1.0743814233166031, 0.8374817382718157], torch.float32, False)

v_free = IndexValue([8.2794385587, 35.403450375, 26.774856263, 22.577665436], False)
polar_free = IndexValue([4.4997895, 11.87706886, 7.423168043, 5.412164335], False)

xdm_model = ExchangeHoleDispersion()
xdm_model.set(m1_net, m2_net, m3_net, v_net, v_free, polar_free, [2.6791, 0.4186], 14.0)

# xdm_model.write('xdm_model.pt')

model = DispersionModel()
model.set([1,6,7,8], symfunc, xdm_model, short_net)
model.write('mlxdm_model.pt')

# Short-network ensemble

import torch
from net import IndexNetwork, NetworkEnsemble
from sf import SymmetryFunction
from combine import ShortRangeEnsembleModel

symfunc = SymmetryFunction()
symfunc.set(4, [16.0], [0.9, 1.16875, 1.4375, 1.70625, 1.975, 2.24375,
            2.5125, 2.78125, 3.05, 3.31875, 3.5875, 3.85625,
            4.125, 4.39375, 4.6625, 4.93125], 5.2,
            [0.19634954, 0.58904862, 0.9817477, 1.3744468,
            1.7671459, 2.1598449, 2.552544, 2.9452431],
            [32.0],  [8.0], [0.9, 1.55, 2.2, 2.85], 3.5)

neural_list = []
for i in range(8):
    neuralnet = IndexNetwork()
    neuralnet.init([[384, 160, 128, 96, 1], [384, 144, 112, 96, 1], [384, 128, 112, 96, 1], [384, 128, 112, 96, 1]], \
            [['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu']], \
            [-0.6010229764812536, -38.058753273282775, -54.68547936106545, -75.16323296795024], \
            [1.0, 1.0, 1.0, 1.0], torch.float64, True)
    neural_list.append(neuralnet)

ensemble = NetworkEnsemble()
ensemble.set(neural_list)

model = ShortRangeEnsembleModel()
model.set([1, 6, 7, 8], symfunc, ensemble)
model.write('ani_ensemble.pt')

# Charge neural network with ensemble

from sf import SymmetryFunction
from net import IndexValue, IndexNetwork, NetworkEnsemble
from charge import ChargeEquilibration
from combine import ChargeEnsembleModel

symfunc = SymmetryFunction()
symfunc.set(4, [16.0], [0.9, 1.16875, 1.4375, 1.70625, 1.975, 2.24375,
            2.5125, 2.78125, 3.05, 3.31875, 3.5875, 3.85625,
            4.125, 4.39375, 4.6625, 4.93125], 5.2,
            [0.19634954, 0.58904862, 0.9817477, 1.3744468,
            1.7671459, 2.1598449, 2.552544, 2.9452431],
            [32.0],  [8.0], [0.9, 1.55, 2.2, 2.85], 3.5)

neural_list = []
for i in range(8):
    neuralnet = IndexNetwork()
    neuralnet.init([[384, 160, 128, 96, 1], [384, 144, 112, 96, 1], [384, 128, 112, 96, 1], [384, 128, 112, 96, 1]], \
            [['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu']], \
            [0.0, 0.0, 0.0, 0.0], \
            [1.0, 1.0, 1.0, 1.0], torch.float64, True)
    neural_list.append(neuralnet)

ensemble = NetworkEnsemble()
ensemble.set(neural_list)

charge_net = IndexNetwork()
charge_net.init([[384, 128, 64, 1], [384, 128, 64, 1], [384, 128, 64, 1], [384, 128, 64, 1]], \
                [['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu'], ['celu', 'celu', 'celu']], \
                [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], torch.float32, False)

charge_equil = ChargeEquilibration()
sigma = [0.31, 0.76, 0.71, 0.66] # Angstrom
sigma = IndexValue(sigma, False)
hardness = [6.3, 2.9, 3.4, 3.9]
hardness = IndexValue(hardness, True)
charge_equil.set(charge_net, hardness, sigma)

model = ChargeEnsembleModel()
model.set([1, 6, 7, 8], symfunc, charge_equil, ensemble)
model.write('runner_ensemble.pt')
