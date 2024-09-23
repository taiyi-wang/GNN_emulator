## Introduction
This is the software accompanying the publication of Graph Neural Network based elastic deformation emulators for magmatic reservoirs of complex geometries, on the journal Volcanica. The software is comprised of two emulators, one for spheroidal magma reservoirs and one for general-shape reservoirs parameterized with spherical harmonics.

## Usage
Usage of the emulators for forward modeling is demonstrated via prediction.py. Follow instructions within prediction.py to set model parameters and run prediction.py.

To see the dependence of surface displacements on the parameters specifying the spheroidal magma chamber, run test_spheroids.py.

## Installation
conda create -n GNN python=3.8 numpy matplotlib scipy pip ipython
conda activate GNN
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install torch_geometric
pip install -U scikit-learn
pip install torch_scatter
pip install h5py
pip install torch_cluster
pip install shapely
