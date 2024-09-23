#!/usr/bin/env python
# coding: utf-8
# Examine predicted surface displacement profiles associated with spheroids of various orientations, aspect ratios, depths etc.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import time as Ti
import torch
import copy
import h5py

parent_dir = os.path.split(os.getcwd())[0]
sys.path.insert(0, parent_dir+'/Functions')
sys.path.insert(0, parent_dir+'/spheroidEmulator')
sys.path.insert(0, parent_dir+'/generalEmulator')
sys.path.insert(0, parent_dir+'/data')

import GNN_functions as GNN_f
from GNN_module import GNN_Network_spheroid, GNN_Network_Norm_spheroid, GNN_Network_Lh_and_Lv_spheroid 
from shape_functions import cmp_SP_shape, set_axes_equal
from mpl_toolkits.mplot3d import Axes3D

device = torch.device('cpu') # 'cpu' or 'cuda'

n_ver_load = 2 
n_ver_load_norm = 2
n_ver_load_grid = 2
n_ver_load_hv = 2

# 1. Set up default parameters-----------------------------------------------------------------------------

# default parameters
dx = 0e3; dy = 0e3; dz = -3e3
Ra = 1e3; Rb = 1e3; 
theta_x = 0; theta_y = 0; theta_z = 0; dp2mu = 1e-3 

# model domain on which to compute surface displacement
X_min = -10e3; X_max = 10e3
Y_min = -10e3; Y_max = 10e3

alpha = Ra/Rb
md = np.array([Ra, Rb, alpha, dx, dy, dz, theta_x, theta_z, dp2mu]) 

# 2. Load trained GNN-----------------------------------------------------------------------------------------

# Load GNN prediction of normalized displacement field
m = GNN_Network_spheroid().to(device)
m.load_state_dict(torch.load(parent_dir + '/spheroidEmulator/TrainedModels/trained_model_spheroids_displacement_prediction_ver_%d.h5'%n_ver_load,  map_location=device)) 

# Load GNN prediction of displacement amplitude
m_norm = GNN_Network_Norm_spheroid().to(device)
m_norm.load_state_dict(torch.load(parent_dir+ '/spheroidEmulator/TrainedModels/trained_model_spheroids_norm_prediction_ver_%d.h5'%n_ver_load_norm,  map_location=device)) 

# Load Lh and Lv prediction
m_hv = GNN_Network_Lh_and_Lv_spheroid().to(device)
m_hv.load_state_dict(torch.load(parent_dir + '/spheroidEmulator/TrainedModels/trained_model_spheroids_Lh_and_Lv_prediction_ver_%d.h5'%n_ver_load_hv,  map_location=device)) 

# Load grids
z = np.load(parent_dir + '/spheroidEmulator/Grids/spatial_grid_apply_ver_%d.npz'%n_ver_load_grid, allow_pickle = True)
A_edges_c = torch.Tensor(z['A_edges_c']).long().to(device)
scale_val = z['scale_val'][0]
z.close()

z = np.load(parent_dir + '/spheroidEmulator/Grids/spatial_grid_spheroids_ver_1.npz', allow_pickle = True)

pos_grid_l = z['pos_grid_l']
pos_grid_l = [torch.Tensor(pos_grid_l[i]).to(device) for i in range(len(pos_grid_l))]
A_edges_l = [GNN_f.make_spatial_graph(pos_grid_l[i]) for i in range(len(pos_grid_l))]
z.close()

z = np.load(parent_dir + '/spheroidEmulator/Grids/spheroid_norm_values.npz')
norm_vals = z['norm_vals']
z.close()

# 3. Plot transects----------------------------------------------------------------------
titles = ['$u_x (m)$', '$u_y (m)$', '$u_z (m)$', '$u_r (m)$']

n_transect = 200
x1 = np.linspace(X_min, X_max, n_transect)
x2 = np.linspace(Y_min, Y_max, n_transect)
x_q1 = np.zeros((n_transect,3))
x_q1[:,0] = x1
x_q2 = np.zeros((n_transect,3))
x_q2[:,1] = x2

alphas = [0.2, 0.4, 0.6, 0.8, 1]

# Vary depths
dz_s = [-10e3, -8e3, -6e3, -4e3, -2e3]
plt.figure()
for i, dz in enumerate(dz_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 

        md_test = copy.deepcopy(md)
        md_test[5] = dz
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x1/1e3, pred[:,0], 'k-', alpha = alphas[i])
plt.xlabel('x (km)')
plt.title('$u_x (m)$')
plt.show()


plt.figure()
for i, dz in enumerate(dz_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 

        md_test = copy.deepcopy(md)
        md_test[5] = dz
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x1/1e3, pred[:,2], 'k-', alpha = alphas[i])
plt.xlabel('x (km)')
plt.title('$u_z (m)$')
plt.show()

# Vary dx
dx_s = [-1e3, -2e3, -6e3, -10e3, -20e3]
plt.figure()
for i, dx in enumerate(dx_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 

        md_test = copy.deepcopy(md)
        md_test[3] = dx
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x1/1e3, pred[:,0], 'k-', alpha = alphas[i])
plt.xlabel('x (km)')
plt.title('$u_x (m)$')
plt.show()

plt.figure()
for i, dx in enumerate(dx_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 

        md_test = copy.deepcopy(md)
        md_test[3] = dx
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x1/1e3, pred[:,2], 'k-', alpha = alphas[i])
plt.xlabel('x (km)')
plt.title('$u_z (m)$')
plt.show()

# Vary volume
R_s = [0.5e3, 1e3, 1.5e3, 2e3, 2.5e3] # radius for spherical chamber
plt.figure()
for i, R in enumerate(R_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 

        md_test = copy.deepcopy(md)
        md_test[0] = R; md_test[1] = R
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x1/1e3, pred[:,0], 'k-', alpha = alphas[i])
plt.xlabel('x (km)')
plt.title('$u_x (m)$')
plt.show()

plt.figure()
for i, R in enumerate(R_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 

        md_test = copy.deepcopy(md)
        md_test[0] = R; md_test[1] = R
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x1/1e3, pred[:,2], 'k-', alpha = alphas[i])
plt.xlabel('x (km)')
plt.title('$u_z (m)$')
plt.show()

# Vary aspect ratio
aspect_s = [0.1, 0.5, 1, 2, 5] # radius for spherical chamber
plt.figure()
for i, aspect in enumerate(aspect_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 
        
        md_test = copy.deepcopy(md)
        md_test[0] = Rb*aspect; md_test[1] = Rb
        md_test[5] = -6e3# make the source deeper
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x1/1e3, pred[:,0], 'k-', alpha = alphas[i])
plt.xlabel('x (km)')
plt.title('$u_x$')
plt.show()

plt.figure()
for i, aspect in enumerate(aspect_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 
        
        md_test = copy.deepcopy(md)
        md_test[0] = Rb*aspect; md_test[1] = Rb
        md_test[5] = -6e3# make the source deeper
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x1/1e3, pred[:,2], 'k-', alpha = alphas[i])
plt.xlabel('x (km)')
plt.title('$u_z$')
plt.show()

# Vary counterclockwise rotation around x-axis (degrees)
theta_x_s = [0, 30, 45, 50, 90] 
plt.figure()
for i, theta_x in enumerate(theta_x_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 
        
        md_test = copy.deepcopy(md)
        md_test[0] = Rb*1.5; md_test[1] = Rb # fix the aspect ratio to be 1.5
        md_test[5] = -2.5e3
        md_test[6] = theta_x                  
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q2, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 
    
    plt.plot(x2/1e3, np.sqrt(pred[:,0]**2+pred[:,1]**2), 'k-', alpha = alphas[i])
plt.xlabel('y (km)')
plt.title('$u_r$')
plt.show()

plt.figure()
for i, theta_x in enumerate(theta_x_s):
    preds  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 
        
        md_test = copy.deepcopy(md)
        md_test[0] = Rb*1.5; md_test[1] = Rb # fix the aspect ratio to be 1.5
        md_test[5] = -2.5e3
        md_test[6] = theta_x                  
    
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md_test, x_q2, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds.append(np.expand_dims(pred, axis = 0)); 

    pred = np.concatenate(preds, axis = 0).mean(0); 

    plt.plot(x2/1e3, pred[:,2], 'k-', alpha = alphas[i])
plt.xlabel('y (km)')
plt.title('$u_z$')
plt.show()

