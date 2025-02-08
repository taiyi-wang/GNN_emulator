# Train the model that predicts the dimensionless displacement field
import yaml
from utils import *
from module_extended import *
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

## Choose Lh_and_Lv version
n_ver_load_lh_and_lv = config['n_ver_load_lh_and_lv_for_training_displacement']
n_step_load_lh_and_lv = config['n_step_load_lh_and_lv_for_training_displacement']

## Load files
z = np.load(path_to_data + 'training_files_within_norm_bounds.npz')
st, itrain, ivald = z['st'], z['itrain'], z['ivald']
z.close()

z = np.load(path_to_data + 'training_files_complex_max_values.npz')
norm_vals_complex_max = z['norm_vals_max']
z.close()

z = np.load(path_to_data + 'training_files_spheroid_max_values.npz')
norm_vals_spheroid_max = z['norm_vals_max']
z.close()

ls, ms, tta, phi, PHI, TTA, lmax = load_spherical_harmonic_parameters()

norm_vals = np.copy(norm_vals_complex_max)

n_nodes_mesh = PHI.shape[0]*PHI.shape[1]
# n_nodes_grid = PHI.shape[0]*PHI.shape[1]


## Load spatial graphs
pos_grid_l = np.load(path_to_file + seperator + 'Grids' + seperator + 'spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver_grid)['pos_grid_l']
A_edges_l = [make_spatial_graph(torch.Tensor(pos_grid_l[i]).to(device), k_pos = k_spc_edges, device = device) for i in range(n_grids)]
print('Note: for training non Lh and Lv model, must make the spatial graphs after saling by predicted Lh and Lv', flush = True)
n_nodes_grid = pos_grid_l[0].shape[0]



A_edges_c = torch.Tensor(np.load(path_to_file + seperator + 'Grids' + seperator + 'cayley_grid_heterogenous_ver_%d.npz'%n_ver_expander_grid)['A_edges_c']).to(device).long()
A_edges_c_mesh = torch.Tensor(np.load(path_to_file + seperator + 'Grids' + seperator + 'cayley_grid_heterogenous_mesh_ver_%d.npz'%n_ver_expander_grid)['A_edges_c']).to(device).long()



## Make batch vectors lh and lv
batch_index_lh_lv = torch.hstack([torch.ones(n_nodes_mesh)*j for j in range(n_batch)]).long().to(device)
batch_index_mesh_lh_lv = torch.hstack([torch.ones(n_nodes_mesh + n_nodes_mesh)*j for j in range(n_batch)]).long().to(device)
batch_index_query_lh_lv = torch.hstack([torch.ones(n_samples)*j for j in range(n_batch)]).long().to(device)
batch_zero_lh_lv = torch.zeros(n_nodes_mesh).long().to(device)
batch_zero_mesh_lh_lv = torch.zeros(n_nodes_mesh + n_nodes_mesh).long().to(device)
batch_query_zero_lh_lv = torch.zeros(n_samples).long().to(device)


## Make batch vectors
batch_index = torch.hstack([torch.ones(n_nodes_grid)*j for j in range(n_batch)]).long().to(device)
batch_index_mesh = torch.hstack([torch.ones(n_nodes_grid + n_nodes_mesh)*j for j in range(n_batch)]).long().to(device)
batch_index_query = torch.hstack([torch.ones(n_samples)*j for j in range(n_batch)]).long().to(device)
batch_zero = torch.zeros(n_nodes_grid).long().to(device)
batch_zero_mesh = torch.zeros(n_nodes_grid + n_nodes_mesh).long().to(device)
batch_query_zero = torch.zeros(n_samples).long().to(device)

## Make offset vectors for batch, so that x positions are not queried when predicting queries.
subset_indices = torch.hstack([torch.arange(n_nodes_grid) + (n_nodes_grid + n_nodes_mesh)*j for j in range(n_batch)]).long().to(device)
## Every n_grid + 2500 nodes are given batch of "bigger than all queries", so that these nodes arn't referenced when querying nearest neighbors

## Set parameters vectors
params = [scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device]
shape_vals = [ls, ms, TTA, PHI]

## Train
device = torch.device(device)

m = GNN_Network_Mesh_Enhanced(device = device).to(device)

optimizer = optim.Adam(m.parameters(), lr = 0.001)

schedular = StepLR(optimizer, 50000, gamma = 0.9)

loss_func = nn.MSELoss()

n_restart = None
if n_restart is not None:
	n_begin = n_restart

	m.load_state_dict(torch.load(path_to_model + 'trained_heterogenous_displacement_model_step_%d_ver_%d.h5'%(n_restart, n_ver_save)))
	optimizer.load_state_dict(torch.load(path_to_model + 'trained_heterogenous_displacement_model_optimizer_step_%d_ver_%d.h5'%(n_restart, n_ver_save)))

	for i in range(n_restart):
		schedular.step()

else:
	n_begin = 0

## Load pre-trained Lh and Lv prediction model
m_lh_and_lv = GNN_Network_Lh_and_Lv_Mesh_Enhanced(device = device).to(device)
m_lh_and_lv.load_state_dict(torch.load(path_to_model + 'trained_heterogenous_Lh_and_Lv_model_step_%d_ver_%d.h5'%(n_step_load_lh_and_lv, n_ver_load_lh_and_lv), map_location = device))
m_lh_and_lv.eval()

## Load pre-trained norm prediction model
m_norm = GNN_Network_Norm_Mesh_Enhanced(device = device).to(device)
m_norm.load_state_dict(torch.load(path_to_model + 'trained_heterogenous_norm_model_step_%d_ver_%d.h5'%(n_step_load_lh_and_lv, n_ver_load_lh_and_lv), map_location = device))
m_norm.eval()

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

	grid_ind = np.random.choice(n_grids, size = n_batch)


	## Make Norm Prediction

	pos_slice, signal_slice, edges_slice, trgt_norm = load_batch_data_norm_mesh_enhanced(st_files, shape_vals, params)
	# n_nodes_mesh = pos_slice[0].shape[0]

	inpt_batch = torch.vstack(signal_slice) # .to(device)
	mask_batch = torch.vstack(signal_slice) # Only select non-position points for mask
	pos_batch = torch.vstack(pos_slice)
	edges_batch = torch.cat([edges_slice[j] + j*n_nodes_mesh for j in range(len(edges_slice))], dim = 1) # .to(device)
	trgt_norm = torch.vstack(trgt_norm) # .to(device)

	# inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch = batch_inputs_norm_mesh(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid)

	pred_norm = m_norm(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index_lh_lv, n_nodes_mesh)


	pos_slice, signal_slice, edges_slice, trgt_slice = load_batch_data_Lh_and_Lv_mesh_enhanced(st_files, shape_vals, params)
	# n_nodes_mesh = pos_slice[0].shape[0]

	inpt_batch = torch.vstack(signal_slice) # .to(device)
	mask_batch = torch.vstack(signal_slice) # Only select non-position points for mask
	pos_batch = torch.vstack(pos_slice)
	edges_batch = torch.cat([edges_slice[j] + j*n_nodes_mesh for j in range(len(edges_slice))], dim = 1) # .to(device)
	trgt_batch = torch.vstack(trgt_slice) # .to(device)


	pred_lh_and_lv = m_lh_and_lv(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index_lh_lv, n_nodes_mesh)

	pred_lh_and_lv = pred_lh_and_lv.detach().cpu().numpy()

	pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, trgt_slice = load_batch_data_displacement_mesh_enhanced_both_edges(st_files, grid_ind, pred_lh_and_lv, pos_grid_l, A_edges_c, A_edges_c_mesh, shape_vals, params)

	inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_feature_batch, edges_batch_c, trgt_batch = batch_inputs_mesh(signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid + n_nodes_mesh, device = device)

	pred_norm_repeat = pred_norm.repeat_interleave(len(pos_grid_l[0]) + n_nodes_mesh, dim = 0)

	pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), pred_norm_repeat, query_batch, edges_batch, edges_feature_batch, edges_batch_c, pos_batch, batch_index_mesh, batch_index_query, subset_indices, n_nodes_grid + n_nodes_mesh)


	## Scaled loss outside prediction
	pred_norm_val = torch.pow(torch.Tensor([10.0]).to(device), pred_norm).repeat_interleave(n_samples, dim = 0)
	trgt_norm_val = torch.pow(torch.Tensor([10.0]).to(device), trgt_norm).repeat_interleave(n_samples, dim = 0)
	# loss = (((1/trgt_norm.repeat_interleave(n_samples, dim = 0))**2)*((pred_norm.repeat_interleave(n_samples, dim = 0)*pred - trgt_batch*trgt_norm.repeat_interleave(n_samples, dim = 0))**2)).mean()
	loss = (((1/trgt_norm_val)**2)*((pred_norm_val*pred - trgt_batch*trgt_norm_val)**2)).mean()


	if loss.item() > 100:
		break

	loss.backward()

	optimizer.step()

	schedular.step()

	losses.append(loss.item())

	print('%d %0.8f'%(i, loss.item()), flush = True)

	if np.mod(i, n_vald_steps) == 0:

		with torch.no_grad():

			isample = np.sort(np.random.choice(ivald, size = n_batch))

			st_files = [st[isample[j]] for j in range(n_batch)]

			grid_ind = np.random.choice(n_grids, size = n_batch)

			pos_slice, signal_slice, edges_slice, trgt_norm = load_batch_data_norm_mesh_enhanced(st_files, shape_vals, params)
			# n_nodes_mesh = pos_slice[0].shape[0]

			inpt_batch = torch.vstack(signal_slice) # .to(device)
			mask_batch = torch.vstack(signal_slice) # Only select non-position points for mask
			pos_batch = torch.vstack(pos_slice)
			edges_batch = torch.cat([edges_slice[j] + j*n_nodes_mesh for j in range(len(edges_slice))], dim = 1) # .to(device)
			trgt_norm = torch.vstack(trgt_norm) # .to(device)

			# inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch = batch_inputs_norm_mesh(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid)

			pred_norm = m_norm(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index_lh_lv, n_nodes_mesh)

			pos_slice, signal_slice, edges_slice, trgt_slice = load_batch_data_Lh_and_Lv_mesh_enhanced(st_files, shape_vals, params)
			# n_nodes_mesh = pos_slice[0].shape[0]

			inpt_batch = torch.vstack(signal_slice) # .to(device)
			mask_batch = torch.vstack(signal_slice) # Only select non-position points for mask
			pos_batch = torch.vstack(pos_slice)
			edges_batch = torch.cat([edges_slice[j] + j*n_nodes_mesh for j in range(len(edges_slice))], dim = 1) # .to(device)
			trgt_batch = torch.vstack(trgt_slice) # .to(device)

			# inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch = batch_inputs_norm_mesh(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid)

			pred_lh_and_lv = m_lh_and_lv(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index_lh_lv, n_nodes_mesh)

			pred_lh_and_lv = pred_lh_and_lv.detach().cpu().numpy()

			## Now make displacement prediction (with predicted Lh and Lv spatial graph size scaling values)

			pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, trgt_slice = load_batch_data_displacement_mesh_enhanced_both_edges(st_files, grid_ind, pred_lh_and_lv, pos_grid_l, A_edges_c, A_edges_c_mesh, shape_vals, params)


			inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_feature_batch, edges_batch_c, trgt_batch = batch_inputs_mesh(signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, pos_slice, trgt_slice, n_nodes_grid + n_nodes_mesh, device = device)

			pred_norm_repeat = pred_norm.repeat_interleave(len(pos_grid_l[0]) + n_nodes_mesh, dim = 0)

			pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), pred_norm_repeat, query_batch, edges_batch, edges_feature_batch, edges_batch_c, pos_batch, batch_index_mesh, batch_index_query, subset_indices, n_nodes_grid + n_nodes_mesh)


			## Scaled loss outside prediction
			pred_norm_val = torch.pow(torch.Tensor([10.0]).to(device), pred_norm).repeat_interleave(n_samples, dim = 0)
			trgt_norm_val = torch.pow(torch.Tensor([10.0]).to(device), trgt_norm).repeat_interleave(n_samples, dim = 0)
			# loss = (((1/trgt_norm.repeat_interleave(n_samples, dim = 0))**2)*((pred_norm.repeat_interleave(n_samples, dim = 0)*pred - trgt_batch*trgt_norm.repeat_interleave(n_samples, dim = 0))**2)).mean()
			loss = (((1/trgt_norm_val)**2)*((pred_norm_val*pred - trgt_batch*trgt_norm_val)**2)).mean()


			losses_vald.append(loss.item())

			print('%d %0.8f (Vald)'%(i, loss.item()), flush = True)

		# with open(path_to_model + 'output_displacement_%d.txt'%n_ver_save, 'a') as text_file:
		#	text_file.write('%d loss %0.9f, %0.9f \n'%(i, losses[-1], losses_vald[-1]))

	if np.mod(i, n_save_steps) == 0:

		torch.save(m.state_dict(), path_to_model + 'trained_heterogenous_displacement_model_step_%d_ver_%d.h5'%(i, n_ver_save))
		torch.save(optimizer.state_dict(), path_to_model + 'trained_heterogenous_displacement_model_optimizer_step_%d_ver_%d.h5'%(i, n_ver_save))
		np.savez_compressed(path_to_model + 'trained_heterogenous_displacement_model_prediction_step_%d_ver_%d.npz'%(i, n_ver_save), pred = pred.cpu().detach().numpy(), trgt = trgt_batch.cpu().detach().numpy(), query = query_batch.cpu().detach().numpy(), losses = losses, losses_vald = losses_vald, itrain = itrain, ivald = ivald)
