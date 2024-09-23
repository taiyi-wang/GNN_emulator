import yaml
from utils import *
from module import *
from shape_functions import *
import pathlib

def load_model_mesh_enhanced(path_to_file):

	# path_to_file = str(pathlib.Path().absolute())
	seperator = '\\' if '\\' in path_to_file else '/'
	path_to_model = path_to_file + seperator + 'TrainedModels' + seperator

	## Load parameters
	#print(path_to_file)
	config = load_config(path_to_file + seperator + 'config.yaml')
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

	# n_ver_grid = 1

	## Choose application versions
	n_ver_load_displacement = config['n_ver_load_displacement']
	n_step_load_displacement = config['n_step_load_displacement']


	n_ver_load_lh_and_lv = config['n_ver_load_lh_and_lv']
	n_step_load_lh_and_lv = config['n_step_load_lh_and_lv']

	n_ver_load_norm = config['n_ver_load_norm']
	n_step_load_norm = config['n_step_load_norm']


	norm_vals = np.load(path_to_file + seperator + 'training_files_complex_max_values.npz')['norm_vals_max']


	ls, ms, tta, phi, PHI, TTA, lmax = load_spherical_harmonic_parameters()

	## Load spatial graphs
	pos_grid_l = np.load(path_to_file + seperator + 'Grids' + seperator + 'spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver_grid)['pos_grid_l']
	A_edges_l = [make_spatial_graph(torch.Tensor(pos_grid_l[i]).to(device), k_pos = k_spc_edges, device = device) for i in range(n_grids)]

	A_edges_c = torch.Tensor(np.load(path_to_file + seperator + 'Grids' + seperator + 'cayley_grid_heterogenous_ver_%d.npz'%n_ver_expander_grid)['A_edges_c']).to(device).long()
	A_edges_c_mesh = torch.Tensor(np.load(path_to_file + seperator + 'Grids' + seperator + 'cayley_grid_heterogenous_mesh_ver_%d.npz'%n_ver_expander_grid)['A_edges_c']).to(device).long()

	## Apply

	device = torch.device(device)

	n_vers_load = [n_ver_load_displacement, n_ver_load_lh_and_lv, n_ver_load_norm]
	n_steps_load = [n_step_load_displacement, n_step_load_lh_and_lv, n_step_load_norm]

	m = GNN_Merged_Mesh_Enhanced(path_to_file + seperator, n_vers_load, n_steps_load, pos_grid_l, A_edges_c, A_edges_c_mesh, norm_vals, device = device)

	return m

