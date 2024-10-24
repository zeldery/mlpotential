==================================
Machine Learning Potential package
==================================

The package is used to create and test the machine learning potential (MLP), especially 
2nd and 4th generation neural network potential proposed by Behler.

The package includes the base of the code, with useful scripting files in script folder

Requirements
============

The package requires the following packages:

* ``numpy`` for array manipulation
* ``pandas`` for result storage in script
* ``torch`` for machine learning task
* ``h5py`` for large data manipulation
* ``scikit-learn`` for machine learning solving
* ``tqdm`` for display progress

Installation
============

One can install the package using ::

    pip install .

The script in the folder can be used separated with the package available.

Usage
=====

The package supports the short-range neural network potential (NNP) with the 
addition of dispersion correction MLXDM and Charge-Equilibration scheme. All 
models are included in combine files, and can be imported as ::

    from mlpotential.combine import *

The list of supported models is:

* ``ShortRangeModel`` traditional short-range NNP
* ``ShortRangeEnsembleModel`` traditional short-range NNP with Ensemble
* ``ChargeModel`` Charge-equilibration scheme with NNP
* ``ChargeEnsembleModel`` Charge-equilibration scheme with Ensemble of NNP
* ``DispersionModel`` NNP with MLXDM
* ``DispersionEnsembleModel`` Ensemble of NNP with MLXDM
* ``ChargeDispersionModel`` Charge-equilibration scheme with NNP and MLXDM
* ``ChargeDispersionEnsembleModel`` Charge-equilibration scheme with Ensemble NNP and MLXDM

The model can be initialized with the explicit declaration of hyper-parameters. See example 
in ``model_init.py`` script.

Citations
=========

When using the package or script, please cite:

1. Tu, N. T. P.; Rezajooei, N.; Johnson, E. R.; Rowley, C. N. A Neural Network Potential with 
Rigorous Treatment of Long-Range Dispersion. Digital Discovery 2023, 2 (3), 718–727. 
https://doi.org/10.1039/D2DD00150K.

2. Tu, N. T. P.; Williamson, S.; Johnson, E. R.; Rowley, C. N. Modeling Intermolecular 
Interactions with Exchange-Hole Dipole Moment Dispersion Corrections to Neural Network Potentials. 
J. Phys. Chem. B 2024, 128 (35), 8290–8302. https://doi.org/10.1021/acs.jpcb.4c02882.

