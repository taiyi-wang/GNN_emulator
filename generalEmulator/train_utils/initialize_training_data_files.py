
import yaml
from utils import load_config
import glob
import numpy as np
import h5py

### Run this file to loop over the dataset and find all training samples within max norm values, and create the "training" and "validation" indices  ###

config = load_config('config.yaml')

ext_data = config['path_to_data']

n_samples = config['n_samples'] # 2000

scale_val = config['scale_val'] # 50e3

norm_version = config['norm_version'] # 2

max_val = config['max_val'] # 0.1
min_val = config['min_val'] # -3.5

use_only_complex = config['use_only_complex']

## Load data, save maximums
Lh = []
Lv = []

Ns = []
dp2mu = []
dx = [] # position ? (scalar). It is 0
dy = [] # position ? (scalar). It is 0
dz = [] # depth? (scalar)
mu = [] # shear modulus (scalar)
nu = [] # Poissons ratio
ra2d = []
thetax = []
thetaz = []
Fs = []
NormF = []
RMax = []

Ra = []
Rb = []

C = [] # C [3 x 3196] # Positions of mesh?
P = [] # P [3 x 1600] ## P is the positions of the mesh boundaries                                                    Are P the normal vectors of each face?
T = [] # T [3 x 3196] ## All ints (must be the connectivity matrix)

## T is base 1 indexed, and max T is 1600

Ux = [] # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
Uy = [] # [1 x 1440]
Uz = [] # [1 x 1440]
X = [] # [1 x 1440]
Y = [] # [1 x 1440]
Z = [] # [1 x 1440]

dhat = []
nhat = []
that = []

Normals = []
Normals_face = []

Norm_values = []

if use_only_complex == True:

	st = glob.glob(ext_data + 'sph_complex/*.mat')

else:

	st = glob.glob(ext_data + 'sph_complex/*.mat')
	st1 = glob.glob(ext_data + 'sph_spheroid_approx/*.mat')
	st2 = glob.glob(ext_data + 'sph_spheroid_perturb/*.mat')
	st3 = glob.glob(ext_data + 'sph_mode_approx/*.mat')
	st = np.concatenate((st, st1, st2, st3), axis = 0)

n_files = len(st)

iwhere = []
ifail = []
ilarge = []


for i in range(n_files):

	try:
		z = h5py.File(st[i], 'r')
	except:
		ifail.append(i)
		print('Failed on %d (%d)'%(i, len(ifail)))
		continue

	if norm_version == 1:
		norm_val = np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]]))
	elif norm_version == 2:
		norm_val = np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max()
	
	Lh = z['input/Lh'][:][0][0]/scale_val
	Lv = z['input/Lv'][:][0][0]/scale_val

	trgt = np.array([Lh, Lv])

	z.close()

	if (np.log10(norm_val) < max_val)*(np.log10(norm_val) > min_val)*(trgt.max() < 20):

		iwhere.append(i)

	if trgt.max() > 20:
		ilarge.append(i)
		print('Large value on %d (%d)'%(i, trgt.max()))

	if np.mod(i, 100) == 0:
		print(i)

st = [st[iwhere[i]] for i in range(len(iwhere))]

itrain = np.sort(np.random.choice(len(st), size = int(0.9*len(st)), replace = False))
ivald = np.delete(np.arange(len(st)), itrain, axis = 0)

np.savez_compressed(ext_data + 'training_files_within_norm_bounds.npz', st = st, itrain = itrain, ivald = ivald)


n_files = len(st)



norm_vals_max = -1.0*np.inf*np.ones((1,45))

X_max = 0.0
Y_max = 0.0

for i in range(n_files):

	z = h5py.File(st[i], 'r')

	dz = z['input/dz'][:][0][0]/scale_val # depth? (scalar)
	Fs = np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1)
	NormF = z['input/normF'][:].reshape(-1)
	RMax = z['input/rmax'][:].reshape(-1)


	norm_vals_slice = np.concatenate([np.array([np.abs(dz)]), np.abs(Fs).reshape(-1), np.abs(NormF), np.abs(RMax)], axis = 0).reshape(1,-1)

	norm_vals_max = np.concatenate((norm_vals_max, norm_vals_slice), axis = 0).max(0, keepdims = True)

	if np.mod(i, 1000) == 0:
		print(i)

	z.close()

norm_vals_max[norm_vals_max <= 0] = 1.0

np.savez_compressed(ext_data + 'training_files_complex_max_values.npz', st = st, itrain = itrain, ivald = ivald, norm_vals_max = norm_vals_max)
