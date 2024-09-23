
import numpy as np
import torch
from load_model import *
import h5py
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

## Set path to file
path_to_file = str(pathlib.Path().absolute())

m = load_model_mesh_enhanced(path_to_file)

# z = h5py.File('/scratch/users/taiyi/ForIan/complex_08162023/sph_spheroid_approx/sh_304420.mat', 'r')
# z = h5py.File('/scratch/users/taiyi/ForIan/complex_08162023/sph_complex/sh_100000.mat', 'r')
# z = h5py.File('D:/Projects/Laplace/ApplyHeterogenous/sh_304420.mat', 'r')
# z = h5py.File('D:/Projects/Laplace/ApplyHeterogenous/sh_36940.mat', 'r')

z = h5py.File('/Users/taiyiwang/Desktop/SHMC/Output/complex/synthetic_deformation_6.mat', 'r')
#z = h5py.File('/Users/taiyiwang/Desktop/SHMC/Output/complex/cmplx_rand/sh_100000.mat', 'r')
#z = h5py.File('/Users/taiyiwang/Desktop/SHMC/Output/complex/cmplx_rand/sh_100201.mat', 'r')
#z = h5py.File('/Users/taiyiwang/Desktop/SHMC/Output/complex/cmplx_spheroid/sh_304420.mat', 'r')
#z = h5py.File('/Users/taiyiwang/Desktop/SHMC/Output/complex/cmplx_spheroid_perturb/sh_335.mat', 'r')


# z = h5py.File('D:/Projects/Laplace/ApplyHeterogenous/sh_100000.mat', 'r')
dz = float(z['input/dz'][:][0][0]) # depth? (scalar)
Fs = np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1).reshape(-1)
NormF = float(z['input/normF'][:]) # .reshape(-1)
RMax = float(z['input/rmax'][:]) # .reshape(-1)

n_samples = 2000
Ux = z['output/Ux'][0,0:n_samples] # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
Uy = z['output/Uy'][0,0:n_samples] # [1 x 1440]
Uz = z['output/Uz'][0,0:n_samples] # [1 x 1440]
X = z['output/X'][0,0:n_samples] # [1 x 1440]
Y = z['output/Y'][0,0:n_samples] # [1 x 1440]
Z = z['output/Z'][0,0:n_samples] # [1 x 1440]
z.close()

inpt = np.concatenate((np.array([dz]), Fs, np.array([NormF]), np.array([RMax])), axis = 0)
queries = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)), axis = 1)
trgt = np.concatenate((Ux.reshape(-1,1), Uy.reshape(-1,1), Uz.reshape(-1,1)), axis = 1)
grid_ind = 0

## Make predictions
# pred = m.prediction(inpt, queries, grid_ind)

## Average predictions over grid
pred_avg = m.prediction_average_grids(inpt, queries).mean(0)

pred = np.copy(pred_avg)

# 2. Plot surface displacements comparisons with target-------------------------------------
titles = ['$u_x (m)$', '$u_y (m)$', '$u_z (m)$', '$u_r (m)$']

loadTrgt = True
if loadTrgt == 1:
    # trgt = np.concatenate((Ux.reshape(-1,1), Uy.reshape(-1,1), Uz.reshape(-1,1)), axis = 1)
    d_min = np.min(trgt.reshape(-1))
    d_max = np.max(trgt.reshape(-1))
    # d_limit = np.min([np.abs(d_min), np.abs(d_max)]) 
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
            d_min_r = np.min(trgt_r.reshape(-1))
            d_max_r = np.max(trgt_r.reshape(-1))

            # plot radial component
            #d_min_r = 0.016
            #d_max_r = 0.02
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

    # for i in range(3):
    #     for j in range(4):
    #         ax[i,j].set_xlim(-20, 20)
    #         ax[i,j].set_ylim(-20, 20)

    plt.show()

    moi 

    plt.show(block = False)
























################### Extra plotting ####################


import numpy as np
import torch
from load_model import *
import h5py
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

## Set path to file
path_to_file = str(pathlib.Path().absolute())

m = load_model_mesh_enhanced(path_to_file)

# z = h5py.File('/scratch/users/taiyi/ForIan/complex_08162023/sph_spheroid_approx/sh_304420.mat', 'r')
# z = h5py.File('/scratch/users/taiyi/ForIan/complex_08162023/sph_complex/sh_100000.mat', 'r')

file_list = ['D:/Projects/Laplace/ApplyHeterogenous/sh_304420.mat', 'D:/Projects/Laplace/ApplyHeterogenous/sh_36940.mat', 'D:/Projects/Laplace/ApplyHeterogenous/sh_100000.mat', 'D:/Projects/Laplace/ApplyHeterogenous/sh_45270.mat']

for inc, s in enumerate(file_list):

    z = h5py.File(s, 'r')
    # z = h5py.File('D:/Projects/Laplace/ApplyHeterogenous/sh_304420.mat', 'r')
    # z = h5py.File('D:/Projects/Laplace/ApplyHeterogenous/sh_36940.mat', 'r')
    # z = h5py.File('D:/Projects/Laplace/ApplyHeterogenous/sh_100000.mat', 'r')
    dz = float(z['input/dz'][:][0][0]) # depth? (scalar)
    Fs = np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1).reshape(-1)
    NormF = float(z['input/normF'][:]) # .reshape(-1)
    RMax = float(z['input/rmax'][:]) # .reshape(-1)

    n_samples = 2000
    Ux = z['output/Ux'][0,0:n_samples] # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
    Uy = z['output/Uy'][0,0:n_samples] # [1 x 1440]
    Uz = z['output/Uz'][0,0:n_samples] # [1 x 1440]
    X = z['output/X'][0,0:n_samples] # [1 x 1440]
    Y = z['output/Y'][0,0:n_samples] # [1 x 1440]
    Z = z['output/Z'][0,0:n_samples] # [1 x 1440]
    z.close()

    inpt = np.concatenate((np.array([dz]), Fs, np.array([NormF]), np.array([RMax])), axis = 0)
    queries = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)), axis = 1)
    trgt = np.concatenate((Ux.reshape(-1,1), Uy.reshape(-1,1), Uz.reshape(-1,1)), axis = 1)
    grid_ind = 0

    ## Make predictions
    pred = m.prediction(inpt, queries, grid_ind)

    ## Average predictions over grid
    pred_avg = m.prediction_average_grids(inpt, queries).mean(0)

    pred = np.copy(pred_avg)

    # 2. Plot surface displacements comparisons with target-------------------------------------
    titles = ['$u_x (m)$', '$u_y (m)$', '$u_z (m)$', '$u_r (m)$']

    loadTrgt = True
    if loadTrgt == 1:
        # trgt = np.concatenate((Ux.reshape(-1,1), Uy.reshape(-1,1), Uz.reshape(-1,1)), axis = 1)
        d_min = np.min(trgt.reshape(-1))
        d_max = np.max(trgt.reshape(-1))
        # d_limit = np.min([np.abs(d_min), np.abs(d_max)]) 
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
                d_min_r = np.min(trgt_r.reshape(-1))
                d_max_r = np.max(trgt_r.reshape(-1))

                print('Median')
                print(np.median(np.linalg.norm(trgt - pred, axis = 1)/np.linalg.norm(trgt, axis = 1).max()))

                # plot radial component
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

        fig.set_size_inches([15, 12])

        print(np.linalg.norm(pred - trgt)/np.linalg.norm(trgt))

        fig.savefig('D:/Projects/Laplace/ApplyHeterogenous/example_prediction_%d_ver_9.png'%inc, bbox_inches = 'tight', pad_inches = 0.2, dpi = 300)

        # for i in range(3):
        #     for j in range(4):
        #         ax[i,j].set_xlim(-20, 20)
        #         ax[i,j].set_ylim(-20, 20)

        # plt.show(block = False)
