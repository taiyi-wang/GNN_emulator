#!/usr/bin/env python
# coding: utf-8
# Predict surface deformation for either spheroidal or general-shape magma reservoirs

# Usage:
# 1. Set hyper-parameters
# 2. Optional: set parameters specifying chamber geometry. Only do so when not predicting deformation for target data sets where parameters are known.
# 3. Set receiver locations
# 4. In command line, type "python3 prediction.py"

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import time as Ti
import torch
import h5py

parent_dir = os.path.split(os.getcwd())[0]
sys.path.insert(0, parent_dir+'/Functions')
sys.path.insert(0, parent_dir+'/spheroidEmulator')
sys.path.insert(0, parent_dir+'/generalEmulator')
sys.path.insert(0, parent_dir+'/Data')

import GNN_functions as GNN_f
from GNN_module import GNN_Network_spheroid, GNN_Network_Norm_spheroid, GNN_Network_Lh_and_Lv_spheroid
from shape_functions import cmp_SP_shape, cmp_SH_shape_with_norm, generateDegreeOrder, set_axes_equal
from load_model import load_model_mesh_enhanced
from mpl_toolkits.mplot3d import Axes3D

# 1. Set hyper-parameters-------------------------------------------------------------------------------
device = torch.device('cpu') # 'cpu' or 'cuda'

caseFlag = 1          # 0 for spheroid, 1 for general shapes parameterized with spherical harmonics
loadTrgt = 0          # whether to load a target data set (computed using BEM) for comparison
avg_grid = 1          # 0 for using one spatial grid; 1 for averaging over grids (which result in spatially smoother displacements)
iselect = 0           # select which spatial graph (0 - 4 indices) to use; only matters if use_one_grid = True

# If load target data set, choose a source shape and uncomment one of the following paths

# Spheroids
trgt_path  = parent_dir + '/data/asp0d3_61.mat'          # spheroid demo 1
#trgt_path = parent_dir + '/data/asp1d0_277.mat'        # spheroid demo 2
#trgt_path = parent_dir + '/data/asp5d0_1001.mat'       # spheroid demo 3
#trgt_path = parent_dir + '/data/asp2d0_8265.mat'       # spheroid demo 4

# General shapes
#trgt_path = parent_dir + '/data/synthetic_deformation_6.mat' # general demo 1, three-fold axi-symmetry
#trgt_path = parent_dir + '/data/sh_100000.mat'               # general demo 2, superposition of randomly sampled spherical harmonic modes 
#trgt_path = parent_dir + '/data/sh_100201.mat'               # general demo 3, superposition of randomly sampled spherical harmonic modes
#trgt_path = parent_dir + '/data/sh_304420.mat'               # general demo 4, superposition of spherical harmonic modes to resemble spheroids
#trgt_path = parent_dir + '/data/sh_335.mat'                  # general demo 5, same as demo 4, but with additional random perturbations in spherical harmonic modes

# 2. Set up input parameters-----------------------------------------------------------------------------

if caseFlag == 0:

    n_ver_load = 2 
    n_ver_load_norm = 2
    n_ver_load_grid = 2
    n_ver_load_hv = 2

    if loadTrgt == 0:

        ################################## USER INPUT ############################

        # For spheroids, model parameters are:
        # dx, dy, dz: x, y, z coordinates of chamber centroid, respectively (right handed Cartesian coordinate system, where dz in the half-space is negative)
        # Ra: semi-major axis length (m) 
        # Rb: semi-minor axis length (m) 
        # thetax: Counter-clockwise rotation (degrees) with regard to x-axis 
        # thetaz: Counter-clockwise rotation (degrees) with regard to z-axis
        # dp2mu: dimensionless ratio between pressure change and shear modulus
        # Both thetax and thetaz assume semi-major axis aligned with z axis and semi-minor axes aligned with x and y axes when both rotation angles are zero.

        dx = 0; dy = 0; dz = -2.5e3
        Ra = 1.5e3; Rb = 1e3; 
        theta_x = 45; theta_y = 0; theta_z = 0; dp2mu = 1e-3 

        #########################################################################
        
    else:
        z = h5py.File(trgt_path, 'r')
        dx = float(z['input/dx'][:][0][0])
        dy = float(z['input/dy'][:][0][0])
        dz = float(z['input/dz'][:][0][0]) 
        Ra = float(z['input/ra'][:][0][0]); Rb = float(z['input/rb'][:][0][0]); dp2mu = float(z['input/dp2mu'][:][0][0])
        theta_x = float(z['input/thetax'][:][0][0]); theta_y = float(z['input/thetay'][:][0][0]); theta_z = float(z['input/thetaz'][:][0][0])

    # Print values of input parameters
    print('thetax, thetay, thetaz', theta_x, theta_y, theta_z)
    print('Ra, Rb', Ra, Rb)
    print('dx, dy, dz', dx, dy, dz)
    print('dp2mu', dp2mu)

else:

    if loadTrgt == 0:
        ################################## USER INPUT ########################

        # For general shapes, model parameters are:
        # dx, dy, dz: x, y, z coordinates of chamber centroid, respectively (right handed Cartesian coordinate system, where dz in the half-space is negative)
        # Rmax: maximum radius of the chamber relative to its centroid 
        # dp2mu: dimensionless ratio between pressure change and shear modulus
        # f_Re, f_Im: real and imaginery components of spherical harmonic coefficients defining chamber geometry. There are 21 of each of these, corresponding to
        #             spherical harmonics up to degree = 5. To ensure that the result chamber shape is real (not complex), all coefficients corresponding to order = 0
        #             must be identically zero

        dx = 0; dy = 0; dz = -1500; Rmax = 1000; dp2mu = 10**(-3.45)
        f_Re = np.array([4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        f_Im = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        #######################################################################

        lmax = 5                              # maximum degree used to train emulator
        ls, ms = generateDegreeOrder(lmax)
        ls = ls[np.where(ms >= 0)]            # only use non-negative orders for constructing real shapes
        ms = ms[np.where(ms >= 0)]
        
        # By the requirement that the shape is real, these modes do not have imaginery components
        zero_idc = np.where(ms == 0)[0]
        f_Im[zero_idc] = 0

        fs = np.squeeze(np.concatenate((f_Re, f_Im), axis = 0))
    else:
        z = h5py.File(trgt_path, 'r')
        dz = float(z['input/dz'][:][0][0]) # depth
        dx = 0; dy = 0
        fs = np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1).reshape(-1)
        dp2mu = float(z['input/dp2mu'][:][0][0])

        normF_trgt = float(z['input/normF'][:]) # normF from training
        Rmax = float(z['input/rmax'][:]) # 
    
     # Print values of input parameters
    print('fs', fs)
    print('Rmax', Rmax)
    print('dx, dy, dz', dx, dy, dz)
    print('dp2mu', dp2mu)
    
# 3. Set receiver locations----------------------------------------------------------------------------------
if loadTrgt == 0:
    ################################## USER INPUT ########################
    
    X = np.array([-5000, 5000])
    Y = np.array([-5000, 5000])
    Z = np.zeros(np.size(X))
    #####################################################################
else:
    n_samples = 2000
    Ux = z['output/Ux'][0,0:n_samples] # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
    Uy = z['output/Uy'][0,0:n_samples] # [1 x 1440]
    Uz = z['output/Uz'][0,0:n_samples] # [1 x 1440]
    X = z['output/X'][0,0:n_samples] # [1 x 1440]
    Y = z['output/Y'][0,0:n_samples] # [1 x 1440]
    Z = z['output/Z'][0,0:n_samples] # [1 x 1440]

queries = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)), axis = 1) # receiver locations


######################################### DO NOT MODIFY THE FOLLOWING ##########################################################################################


# 4. Load trained GNN-----------------------------------------------------------------------------------------

if caseFlag == 0:   
    # Spheroid
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

else:
    path_to_file = parent_dir + '/generalEmulator'
    m = load_model_mesh_enhanced(path_to_file)


# 1. Plot shape of chamber---------------------------------------------------------------------

tta = np.linspace(0, np.pi, 70)
phi = np.linspace(0, 2*np.pi, 70)
PHI, TTA = np.meshgrid(phi, tta)

if caseFlag == 0:

    alpha = -theta_x/180*np.pi; beta = -theta_y/180*np.pi; gamma = -theta_z/180*np.pi # negative signs are needed to be consistent with input rotation angles, which are counter-clockwise with regard to axes
    X_s, Y_s, Z_s = cmp_SP_shape(Ra, Rb, alpha, beta, gamma, dz, TTA, PHI)

elif caseFlag == 1:

    lmax = 5 # maximum degree used to train emulator
    ls, ms = generateDegreeOrder(lmax)

    # only use non-negative orders for constructing real shapes
    ls = ls[np.where(ms >= 0)]
    ms = ms[np.where(ms >= 0)]

    fs_plt = fs[0:np.size(ls)] + 1j*fs[np.size(ls):]# re-create complex fs vector for plotting

    X_s, Y_s, Z_s, normF_pred = cmp_SH_shape_with_norm(ls, ms, fs_plt, Rmax, dz, TTA, PHI) # the normF0 here is ignored

    if loadTrgt == 1:
        print('Target normF is', normF_trgt)
    print('Predicted normF is', normF_pred)


# plotting 
fig = plt.figure()
ax = fig.add_subplot(111 , projection='3d')
    
light = LightSource(azdeg=0, altdeg=65) # create a light source
illuminated_surface = light.shade(Z_s/1e3, cmap=cm.hot)
surf = ax.plot_surface((X_s+dx)/1e3, (Y_s+dy)/1e3, Z_s/1e3, linewidth = 0.5, facecolors=illuminated_surface, cmap=cm.hot)
surf.set_edgecolor('k')
ax.set_box_aspect([1,1,1])
set_axes_equal(ax)
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
plt.show()

# Compute surface deformation ---------------------------------------------------------------------------------------------------
if caseFlag == 0:

    z = np.load(parent_dir + '/spheroidEmulator/Grids/spheroid_norm_values.npz')
    norm_vals = z['norm_vals']
    z.close()
    alpha = Ra/Rb
    md = np.array([Ra, Rb, alpha, dx, dy, dz, theta_x, theta_z, dp2mu]) 

    if avg_grid == 0:
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect]
        pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md, queries, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
    else:
        preds  = []
        t1 = Ti.time()
        for iselect in range(5):
            pos_grid = pos_grid_l[iselect]
            A_edges = A_edges_l[iselect] 
            pred = GNN_f.apply_models_make_displacement_prediction_spheroid(md, queries, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
            
            preds.append(np.expand_dims(pred, axis = 0))
        pred = np.concatenate(preds, axis = 0).mean(0)
        t2 = Ti.time()

else:

    t1 = Ti.time()
    pred = m.predict(fs, dx, dy, dz, Rmax, dp2mu, X, Y, Z, avg_grid)
    t2 = Ti.time()

print('Forward calculation took', (t2-t1), 'seconds.')

# 2. Plot surface displacements comparisons with target-------------------------------------
titles = ['$u_x (m)$', '$u_y (m)$', '$u_z (m)$', '$u_r (m)$']

if loadTrgt == 1:
    trgt = np.concatenate((Ux.reshape(-1,1), Uy.reshape(-1,1), Uz.reshape(-1,1)), axis = 1)
    d_min = np.min(trgt.reshape(-1))
    d_max = np.max(trgt.reshape(-1))
    d_limit = np.max([np.abs(d_min), np.abs(d_max)]) 

    resid = pred-trgt

    fig, ax = plt.subplots(3,4, figsize = [8.97, 5.93], sharex = True, sharey = True)
    for i in range(4):
        if i < 3:
            h1 = ax[0, i].scatter(queries[:, 0]/1e3, queries[:, 1]/1e3, s=5,c = trgt[:, i], cmap='bwr', vmin=-d_limit, vmax=d_limit)
            h2 = ax[1, i].scatter(queries[:, 0]/1e3, queries[:, 1]/1e3, s=5,c = pred[:, i], cmap='bwr', vmin=-d_limit, vmax=d_limit)
            h3 = ax[2, i].scatter(queries[:, 0]/1e3, queries[:, 1]/1e3, s=5,c = resid[:, i], cmap='bwr', vmin=-d_limit, vmax=d_limit)
        else:
            # compute radial component
            trgt_r = np.sqrt(trgt[:, 0]**2 + trgt[:, 1]**2)
            pred_r = np.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
            resid_r = pred_r - trgt_r
            d_min_r = np.min(trgt_r.reshape(-1)); d_max_r = np.max(trgt_r.reshape(-1))

            # plot radial component
            print('trgt_r', trgt_r)
            h1 = ax[0, i].scatter(queries[:, 0]/1e3, queries[:, 1]/1e3, s=5,c = trgt_r, cmap='turbo', vmin=d_min_r, vmax=d_max_r)
            h2 = ax[1, i].scatter(queries[:, 0]/1e3, queries[:, 1]/1e3, s=5,c = pred_r, cmap='turbo', vmin=d_min_r,vmax=d_max_r)
            h3 = ax[2, i].scatter(queries[:, 0]/1e3, queries[:, 1]/1e3, s=5,c = resid_r, cmap='turbo', vmin=d_min_r, vmax=d_max_r)

        ax[0, i].set_aspect('equal'); ax[1, i].set_aspect('equal'); ax[2, i].set_aspect('equal')
        ax[2, i].set_xlabel('x (km)')
        ax[0, i].set_title(titles[i])
        if i == 2 or i == 3:
            plt.colorbar(h1)
            plt.colorbar(h2)
            plt.colorbar(h3)
    ax[0, 0].set_ylabel('y (km)'); ax[1, 0].set_ylabel('y (km)'); ax[2, 0].set_ylabel('y (km)')
    plt.show()


# 3.  Plot transects----------------------------------------------------------------------
n_transect = 200
x1 = np.linspace(queries[:,0].min(), queries[:,0].max(), n_transect)
x2 = np.linspace(queries[:,1].min(), queries[:,1].max(), n_transect)
x_q1 = np.zeros((n_transect,3))
x_q1[:,0] = x1
x_q2 = np.zeros((n_transect,3))
x_q2[:,1] = x2

if caseFlag == 0:
    preds1  = []; preds2  = []
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 
        pred1 = GNN_f.apply_models_make_displacement_prediction_spheroid(md, x_q1, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        pred2 = GNN_f.apply_models_make_displacement_prediction_spheroid(md, x_q2, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds1.append(np.expand_dims(pred1, axis = 0)); preds2.append(np.expand_dims(pred2, axis = 0))

    pred1 = np.concatenate(preds1, axis = 0).mean(0); pred2 = np.concatenate(preds2, axis = 0).mean(0)
else:

    pred1 = m.predict(fs, dx, dy, dz, Rmax, dp2mu, x_q1[:,0], x_q1[:,1], x_q1[:,2], avg_grid)
    pred2 = m.predict(fs, dx, dy, dz, Rmax, dp2mu, x_q1[:,0], x_q2[:,1], x_q2[:,2], avg_grid)


fig, ax = plt.subplots(2,3, figsize = [8.97, 5.93], sharex = True, sharey = True)
for i in range(3):
    ax[0,i].plot(x1/1e3, pred1[:,i], 'k-')
    ax[0,i].set_xlabel('x (km)')
    ax[0,i].set_title(titles[i])
    ax[1,i].plot(x2/1e3, pred2[:,i], 'k-')
    ax[1,i].set_xlabel('y (km)')

# to show radial asymmetry in deformaiton
ax[0,0].plot(x2/1e3, pred2[:,1], 'k--')
ax[1,1].plot(x1/1e3, pred1[:,0], 'k--')
ax[0,1].plot(x2/1e3, pred2[:,0], 'k--')
ax[1,0].plot(x1/1e3, pred1[:,1], 'k--')
ax[0,2].plot(x2/1e3, pred2[:,2], 'k--')
ax[1,2].plot(x1/1e3, pred1[:,2], 'k--')
plt.show()

# 4. Plot deformation on regular field-------------------------------------------------
X1, X2 = np.meshgrid(x1, x2)
X_q = np.zeros((n_transect**2,3))
X_q[:,0] = X1.reshape(-1)
X_q[:,1] = X2.reshape(-1)
if caseFlag == 0:
    preds3  = []; 
    for iselect in range(5):
        pos_grid = pos_grid_l[iselect]
        A_edges = A_edges_l[iselect] 
        pred3 = GNN_f.apply_models_make_displacement_prediction_spheroid(md, X_q, m, m_norm, m_hv, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu')
        preds3.append(np.expand_dims(pred3, axis = 0))

    pred3 = np.concatenate(preds3, axis = 0).mean(0); 
else:
    pred3 = m.predict(fs, dx, dy, dz, Rmax, dp2mu, X_q[:,0], X_q[:,1], X_q[:,2], avg_grid)
    

pred3_r = np.sqrt(pred3[:, 0]**2 + pred3[:, 1]**2)

d_min = np.min(pred3.reshape(-1)); d_max = np.max(pred3.reshape(-1))
d_limit = np.max([np.abs(d_min), np.abs(d_max)]) 
d_min_r = np.min(pred3_r.reshape(-1)); d_max_r = np.max(pred3_r.reshape(-1))
d_limit_r = np.max([np.abs(d_min_r), np.abs(d_max_r)]) 

fig, ax = plt.subplots(1,4, figsize = [8.97, 5.93], sharex = True, sharey = True)
for i in range(4):
    if i > 2:
        h = ax[i].scatter(X_q[:, 0]/1e3, X_q[:, 1]/1e3, s=5,c = pred3_r, cmap='turbo', vmin=d_min_r, vmax=d_max_r)
    else:
        h = ax[i].scatter(X_q[:, 0]/1e3, X_q[:, 1]/1e3, s=5,c = pred3[:, i], cmap='bwr', vmin=-d_limit, vmax=d_limit)
    ax[i].set_aspect('equal')
    ax[i].set_xlabel('x (km)')
    ax[i].set_title(titles[i])
    if i >= 2:
        plt.colorbar(h)
ax[0].set_ylabel('y (km)')
plt.show()

