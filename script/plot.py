import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import HARTREE_TO_KCALMOL
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':11})
rc('mathtext',**{'default':'regular'})

####################################################################
# Analyze

dat = pd.read_csv('charge_force_ensemble_batch.csv', index_col=0)
tmp = dat.loc[dat['train'] == 1,:]
x = (tmp['predict'] - tmp['reference']).abs().mean() * HARTREE_TO_KCALMOL
print(f'Train: {x} kcal/mol')
tmp = dat.loc[dat['train'] == 0,:]
x = (tmp['predict'] - tmp['reference']).abs().mean() * HARTREE_TO_KCALMOL
print(f'Test : {x} kcal/mol')


ref = pd.read_csv('deshaw_370K_combined.csv', index_col=0)
dat = pd.read_csv('energy_ensemble_deshaw.csv', index_col=0)
x = (ref['PBE0'] - dat['delta_E']).abs().mean()
print(f'DeShaw: {x} kcal/mol')

# DESHAW

plt.clf()
plt.cla()

ref = pd.read_csv('deshaw_370K_combined.csv', index_col=0)
dat = pd.read_csv('charge_force_ensemble_deshaw.csv', index_col=0)

fig, ax = plt.subplots(1, 1, figsize=(3.25, 3))

x = ref['PBE0']
y = dat['delta_E']
ax.hist2d(x, y, range=[[-10, 10], [-10, 10]], bins=[100, 100], norm=mpl.colors.LogNorm(), cmap=plt.colormaps['YlOrRd'])
ax.set_xlabel('PBE0 (kcal/mol)')
ax.set_ylabel('Charge Force ensemble (kcal/mol)')

fig.tight_layout()

fig.savefig('charge_force_ensemble_deshaw.pdf')

# LINE

plt.clf()
plt.cla()

dat = pd.read_csv('charge_force_acetone.csv', index_col=0)
ref = pd.read_csv('acetone_dimer.csv', index_col=0)

fig, ax = plt.subplots(1, 1, figsize=(3.25, 3))

x = np.arange(171)
y = ref.loc[:170, 'PBE0_energy'] - ref.loc[170, 'PBE0_energy']
ax.plot(x,y, color='r')
y = dat.loc[:170, 'energy'] - 2 * dat.loc[171, 'energy']
ax.plot(x,y, color='b')
ax.set_ylim([-5, 5])

ax.set_xlabel('Distance')
ax.set_ylabel('Energy (kcal/mol')

fig.tight_layout()
fig.savefig('charge_force_acetone.pdf')


plt.clf()
plt.cla()

dat = pd.read_csv('charge_force_nma.csv', index_col=0)
ref = pd.read_csv('nma_dimer.csv', index_col=0)

fig, ax = plt.subplots(1, 1, figsize=(3.25, 3))

x = np.arange(171)
y = ref.loc[:170, 'PBE0_energy'] - ref.loc[170, 'PBE0_energy']
ax.plot(x,y, color='r')
y = dat.loc[:170, 'energy'] - 2 * dat.loc[171, 'energy']
ax.plot(x,y, color='b')
ax.set_ylim([-7, 5])

ax.set_xlabel('Distance')
ax.set_ylabel('Energy (kcal/mol')

fig.tight_layout()
fig.savefig('charge_force_nma.pdf')


