# Train the model that predicts the spatial scale of deformation field
import yaml
from utils import *
from module import *
from shape_function import *
import pathlib

path_to_file = str(pathlib.Path().absolute())
seperator = '\\' if '\\' in path_to_file else '/'
path_to_model = path_to_file + seperator + 'TrainedModels' + seperator

## Load parameters
config = load_config('config.yaml')
n_grids = config['number_of_grids']
n_nodes_grid = config['number_of_spatial_nodes']
scale_val = config['scale_val']
k_spc_edges = config['k_spc_edges']
norm_version = config['norm_version']
n_samples = config['n_samples']
n_cayley = config['n_cayley']
use_only_complex = config['use_only_complex']
path_to_data = config['path_to_data']
min_val = config['min_val']
max_val = config['max_val']
n_ver_save = config['n_ver_save']
n_epochs = config['n_epochs']
n_batch = config['n_batch']
device = config['device']
n_features = config['n_features']

## Choose grid version
n_ver_grid = config['n_ver_grid']
n_ver_expander_grid = config['n_ver_expander_grid']

## Load files
z = np.load(path_to_data + 'training_files_within_norm_bounds.npz')
st, itrain, ivald = z['st'], z['itrain'], z['ivald']
z.close()

z = np.load(path_to_data + 'training_files_complex_max_values.npz')
norm_vals_complex_max = z['norm_vals_max']
z.close()


ls, ms, tta, phi, PHI, TTA, lmax = load_spherical_harmonic_parameters()

norm_vals = np.copy(norm_vals_complex_max)

n_nodes_mesh = PHI.shape[0]*PHI.shape[1]
n_nodes_grid = PHI.shape[0]*PHI.shape[1]


## Load spatial graphs
pos_grid_l = np.load(path_to_file + seperator + 'Grids' + seperator + 'spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver_grid)['pos_grid_l']
A_edges_l = [make_spatial_graph(torch.Tensor(pos_grid_l[i]).to(device), k_pos = k_spc_edges, device = device) for i in range(n_grids)]
print('Note: for training non Lh and Lv model, must make the spatial graphs after saling by predicted Lh and Lv')

## Build cayley graph
# A_edges_c = make_cayleigh_graph(n_cayley)
# A_edges_c = subgraph(torch.arange(n_nodes_grid), torch.Tensor(A_edges_c.T).long().flip(0).contiguous())[0].to(device)

A_edges_c = torch.Tensor(np.load(path_to_file + seperator + 'Grids' + seperator + 'cayley_grid_heterogenous_ver_%d.npz'%n_ver_expander_grid)['A_edges_c']).to(device).long()

## Make batch vectors
batch_index = torch.hstack([torch.ones(n_nodes_grid)*j for j in range(n_batch)]).long().to(device)
batch_index_query = torch.hstack([torch.ones(n_samples)*j for j in range(n_batch)]).long().to(device)
batch_zero = torch.zeros(n_nodes_grid).long().to(device)
batch_query_zero = torch.zeros(n_samples).long().to(device)

## Set parameters vector
params = [scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device]
shape_vals = [ls, ms, TTA, PHI]

## Train
device = torch.device(device)

m = GNN_Network_Lh_and_Lv_Mesh_Enhanced(device = device).to(device)

optimizer = optim.Adam(m.parameters(), lr = 0.001)

schedular = StepLR(optimizer, 25000, gamma = 0.9)

loss_func = nn.MSELoss()

n_restart = None
if n_restart is not None:
	n_begin = n_restart

	m.load_state_dict(torch.load(path_to_model + 'trained_heterogenous_Lh_and_Lv_model_step_%d_ver_%d.h5'%(n_restart, n_ver_save)))
	optimizer.load_state_dict(torch.load(path_to_model + 'trained_heterogenous_Lh_and_Lv_model_optimizer_step_%d_ver_%d.h5'%(n_restart, n_ver_save)))

else:
	n_begin = 0


n_train = len(itrain)
n_vald = len(ivald)

losses = []
losses_vald = []

n_vald_steps = 10
n_save_steps = 1000


for i in range(n_begin, n_epochs):

	optimizer.zero_grad()

	isample = np.sort(np.random.choice(itrain, size = n_batch))

	st_files = [st[isample[j]] for j in range(n_batch)]

	# grid_ind = np.random.choice(n_grids, size = n_batch)

	pos_slice, signal_slice, edges_slice, trgt_slice = load_batch_data_Lh_and_Lv_mesh_enhanced(st_files, shape_vals, params)
	# n_nodes_mesh = pos_slice[0].shape[0]

	inpt_batch = torch.vstack(signal_slice) # .to(device)
	mask_batch = torch.vstack(signal_slice) # Only select non-position points for mask
	pos_batch = torch.vstack(pos_slice)
	edges_batch = torch.cat([edges_slice[j] + j*n_nodes_mesh for j in range(len(edges_slice))], dim = 1) # .to(device)
	trgt_batch = torch.vstack(trgt_slice) # .to(device)

	# inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch = batch_inputs_norm_mesh(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid)

	pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index, n_nodes_mesh)

	loss = loss_func(pred/5.0, trgt_batch/5.0)

	if loss.item() > 100:
		break

	loss.backward()

	optimizer.step()

	schedular.step()

	losses.append(loss.item())

	print('%d %0.8f'%(i, loss.item()))

	if np.mod(i, n_vald_steps) == 0:

		with torch.no_grad():

			isample = np.sort(np.random.choice(ivald, size = n_batch))

			st_files = [st[isample[j]] for j in range(n_batch)]

			pos_slice, signal_slice, edges_slice, trgt_slice = load_batch_data_Lh_and_Lv_mesh_enhanced(st_files, shape_vals, params)
			# n_nodes_mesh = pos_slice[0].shape[0]

			inpt_batch = torch.vstack(signal_slice) # .to(device)
			mask_batch = torch.vstack(signal_slice) # Only select non-position points for mask
			pos_batch = torch.vstack(pos_slice)
			edges_batch = torch.cat([edges_slice[j] + j*n_nodes_mesh for j in range(len(edges_slice))], dim = 1) # .to(device)
			trgt_batch = torch.vstack(trgt_slice) # .to(device)

			# inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch = batch_inputs_norm_mesh(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid)

			pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index, n_nodes_mesh)

			loss = loss_func(pred/5.0, trgt_batch/5.0)

			losses_vald.append(loss.item())

			print('%d %0.4f (Vald)'%(i, loss.item()), flush = True)

		# with open(path_to_model + 'output_Lh_and_Lv_%d.txt'%n_ver_save, 'a') as text_file:
		# 	text_file.write('%d loss %0.9f, %0.9f \n'%(i, losses[-1], losses_vald[-1]))

	if np.mod(i, n_save_steps) == 0:

		torch.save(m.state_dict(), path_to_model + 'trained_heterogenous_Lh_and_Lv_model_step_%d_ver_%d.h5'%(i, n_ver_save))
		torch.save(optimizer.state_dict(), path_to_model + 'trained_heterogenous_Lh_and_Lv_model_optimizer_step_%d_ver_%d.h5'%(i, n_ver_save))
		np.savez_compressed(path_to_model + 'trained_heterogenous_Lh_and_Lv_model_prediction_step_%d_ver_%d.npz'%(i, n_ver_save), pred = pred.cpu().detach().numpy(), trgt = trgt_batch.cpu().detach().numpy(), losses = losses, losses_vald = losses_vald, itrain = itrain, ivald = ivald)