import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from torch_scatter import scatter
import h5py
import glob
import networkx as nx

from torch import nn, optim
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.utils import to_networkx, to_undirected, from_networkx
from torch_geometric.transforms import FaceToEdge, GenerateMeshNormals
from joblib import Parallel, delayed
import multiprocessing
from scipy.io import loadmat
from scipy.special import sph_harm
from scipy.spatial import cKDTree
from torch.autograd import Variable
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from sklearn.metrics import pairwise_distances

from shape_functions import *
import yaml

def load_config(file_path: str) -> dict:
	"""Load configuration from a YAML file."""
	with open(file_path, 'r') as file:
		return yaml.safe_load(file)

def make_cayleigh_graph(n):

	generators = [np.array([1, 1, 0, 1]).reshape(2,2), np.array([1, 0, 1, 1]).reshape(2,2)]
	nodes = np.vstack([generators[0].reshape(1,-1), generators[1].reshape(1,-1)])
	edges = []

	new = np.inf
	cnt = 0

	while new > 0:

		print('iteration %d, num nodes %d'%(cnt, len(nodes)))

		tree = cKDTree(nodes)
		len_nodes = len(nodes)

		new_nodes_1 = []
		new_nodes_2 = []

		for i in range(len(nodes)):

			new_nodes_1.append(np.mod(nodes[i].reshape(2,2) @ generators[0], n).reshape(1,-1))
			new_nodes_2.append(np.mod(nodes[i].reshape(2,2) @ generators[1], n).reshape(1,-1))

		new_nodes_1 = np.vstack(new_nodes_1) # has size nodes
		new_nodes_2 = np.vstack(new_nodes_2)
		new_nodes = np.unique(np.concatenate((new_nodes_1, new_nodes_2), axis = 0), axis = 0)

		q = tree.query(new_nodes)[0]
		inew = np.where(q > 0)[0]
		new_nodes = new_nodes[inew]

		if len(inew) == 0:
			new = 0
			continue # Break loop

		## Now need to find which entries in new_nodes are linked for each input node
		tree_new = cKDTree(new_nodes)
		ip = tree_new.query(new_nodes_1)
		ip1 = np.where(ip[0] == 0)[0] ## Points to current absolute node indices that are linked to new node
		edges_new_1 = np.concatenate((ip1.reshape(-1,1), len_nodes + ip[1][ip1].reshape(-1,1)), axis = 1)

		ip = tree_new.query(new_nodes_2)
		ip1 = np.where(ip[0] == 0)[0] ## Points to current absolute node indices that are linked to new node
		edges_new_2 = np.concatenate((ip1.reshape(-1,1), len_nodes + ip[1][ip1].reshape(-1,1)), axis = 1)

		# edges.append(np.unique(np.concatenate((edges_new_1, edges_new_2), axis = 0), axis = 0))
		nodes = np.concatenate((nodes, new_nodes), axis = 0)
		
		cnt += 1

	## Find inverses to generators
	inv_indices_1 = []
	inv_indices_2 = []
	for i in range(len(nodes)):
		if np.abs(np.mod(generators[0] @ nodes[i].reshape(2,2), n) - np.eye(2)).max() == 0:
			inv_indices_1.append(i)
		if np.abs(np.mod(generators[1] @ nodes[i].reshape(2,2), n) - np.eye(2)).max() == 0:
			inv_indices_2.append(i)

	assert(len(inv_indices_1) == 1)
	assert(len(inv_indices_2) == 1)

	generators_inverses = [nodes[inv_indices_1[0]].reshape(2,2), nodes[inv_indices_2[0]].reshape(2,2)]

	## Now must add missing edges between all previously created nodes. (can do this outside of the loop)
	for i in range(len(nodes)):
		for j in range(len(nodes)):
			dist1 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators[0], n) - nodes[j].reshape(2,2)).max()
			dist2 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators[1], n) - nodes[j].reshape(2,2)).max()
			if ((dist1 == 0) + (dist2 == 0)) > 0:
				edges.append(np.array([i,j]).reshape(1,-1))

	for i in range(len(nodes)):
		for j in range(len(nodes)):
			dist1 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators_inverses[0], n) - nodes[j].reshape(2,2)).max()
			dist2 = np.abs(np.mod(nodes[i].reshape(2,2) @ generators_inverses[1], n) - nodes[j].reshape(2,2)).max()
			if ((dist1 == 0) + (dist2 == 0)) > 0:
				edges.append(np.array([i,j]).reshape(1,-1))

	edges = np.unique(np.vstack(edges), axis = 0)
	## Check for all edges, if each node is really linked to the declared nodes.
	for i in range(len(edges)):
		dist1 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators[0], n) - nodes[edges[i,1]].reshape(2,2)).max()
		dist2 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators[1], n) - nodes[edges[i,1]].reshape(2,2)).max()
		dist3 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators_inverses[0], n) - nodes[edges[i,1]].reshape(2,2)).max()
		dist4 = np.abs(np.mod(nodes[edges[i,0]].reshape(2,2) @ generators_inverses[1], n) - nodes[edges[i,1]].reshape(2,2)).max()
		assert(((dist1 == 0) + (dist2 == 0) + (dist3 == 0) + (dist4 == 0)) > 0)
		# print(i)

	return edges

## This global_mean_pool function is taken from 
## TORCH_GEOMETRIC.GRAPHGYM.MODELS.POOLING
## It relies on scatter.
def global_mean_pool(x, batch, size=None):
	"""
	Globally pool node embeddings into graph embeddings, via elementwise mean.
	Pooling function takes in node embedding [num_nodes x emb_dim] and
	batch (indices) and outputs graph embedding [num_graphs x emb_dim].

	Args:
		x (torch.tensor): Input node embeddings
		batch (torch.tensor): Batch tensor that indicates which node
		belongs to which graph
		size (optional): Total number of graphs. Can be auto-inferred.

	Returns: Pooled graph embeddings

	"""
	size = batch.max().item() + 1 if size is None else size
	return scatter(x, batch, dim=0, dim_size=size, reduce='mean')

def global_max_pool(x, batch, size=None):
	"""
	Globally pool node embeddings into graph embeddings, via elementwise mean.
	Pooling function takes in node embedding [num_nodes x emb_dim] and
	batch (indices) and outputs graph embedding [num_graphs x emb_dim].

	Args:
		x (torch.tensor): Input node embeddings
		batch (torch.tensor): Batch tensor that indicates which node
		belongs to which graph
		size (optional): Total number of graphs. Can be auto-inferred.

	Returns: Pooled graph embeddings

	"""
	size = batch.max().item() + 1 if size is None else size
	return scatter(x, batch, dim=0, dim_size=size, reduce='max')

def global_min_pool(x, batch, size=None):
	"""
	Globally pool node embeddings into graph embeddings, via elementwise mean.
	Pooling function takes in node embedding [num_nodes x emb_dim] and
	batch (indices) and outputs graph embedding [num_graphs x emb_dim].

	Args:
		x (torch.tensor): Input node embeddings
		batch (torch.tensor): Batch tensor that indicates which node
		belongs to which graph
		size (optional): Total number of graphs. Can be auto-inferred.

	Returns: Pooled graph embeddings

	"""
	size = batch.max().item() + 1 if size is None else size
	return scatter(x, batch, dim=0, dim_size=size, reduce='min')


def batch_inputs(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, node_ind_max):

	inpt_batch = torch.vstack(signal_slice) # .to(device)
	mask_batch = inpt_batch[:,3::] # Only select non-position points for mask
	pos_batch = inpt_batch[:,0:3]
	query_batch = torch.vstack(query_slice) # .to(device)
	edges_batch = torch.cat([edges_slice[j] + j*node_ind_max for j in range(len(edges_slice))], dim = 1) # .to(device)
	edges_batch_c = torch.cat([edges_c_slice[j] + j*node_ind_max for j in range(len(edges_c_slice))], dim = 1) # .to(device)

	if len(trgt_slice) > 0:
		trgt_batch = torch.vstack(trgt_slice) # .to(device)
	else:
		trgt_batch = []

	return inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch

def batch_inputs_mesh(signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, pos_slice, trgt_slice, node_ind_max, device = 'cpu'):

	## Let mask be the same as

	inpt_batch = torch.vstack(signal_slice) # .to(device)
	# mask_batch = inpt_batch[:,3::] # Only select non-position points for mask
	pos_batch = torch.vstack(pos_slice)
	query_batch = torch.vstack(query_slice) # .to(device)
	edges_feature_batch = torch.vstack(edges_feature_slice)
	edges_batch = torch.cat([edges_slice[j] + j*node_ind_max for j in range(len(edges_slice))], dim = 1) # .to(device)
	edges_batch_c = torch.cat([edges_c_slice[j] + j*node_ind_max for j in range(len(edges_c_slice))], dim = 1) # .to(device)

	if len(trgt_slice) > 0:
		trgt_batch = torch.vstack(trgt_slice) # .to(device)
	else:
		trgt_batch = []

	## Using inpt_batch as mask
	return inpt_batch, inpt_batch, pos_batch, query_batch, edges_batch, edges_feature_batch, edges_batch_c, trgt_batch

def kmeans_packing_logarithmic(scale_x, offset_x, ndim, n_clusters, n_batch = 3000, n_steps = 1000, n_sim = 1, lr = 0.01):

	V_results = []
	Losses = []

	center_x = ((offset_x[0,0:2] + scale_x[0,0:2]/2.0)).reshape(1,-1)

	a_param = 3.0
	num_thresh = 0.1 # 3.0
	n_up_sample = 10

	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = np.random.rand(n_clusters, ndim)*scale_x + offset_x

			tree = cKDTree(v)
			x = np.random.rand(n_batch, ndim)*scale_x + offset_x
			x_r = np.random.pareto(a_param, size = 100000)
			ifind = np.where(x_r < num_thresh)[0]
			ifind = np.random.choice(ifind, size = n_up_sample*n_batch)
			x_r = x_r[ifind].reshape(-1,1)/num_thresh
			x_theta = np.random.rand(n_up_sample*n_batch)*2.0*np.pi
			x_xy = x_r*np.concatenate((np.cos(x_theta[:,None]), np.sin(x_theta[:,None])), axis = 1)*(scale_x[:,0:2]/2.0)
			x_xy = np.concatenate((x_xy + center_x, np.random.rand(n_up_sample*n_batch,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
			x = np.concatenate((x, x_xy), axis = 0)
			x = x[np.random.choice(x.shape[0], size = n_batch, replace = False)]

			q, ip = tree.query(x)

			rs = []
			ipu = np.unique(ip)
			for j in range(len(ipu)):
				ipz = np.where(ip == ipu[j])[0]
				# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
				update = (x[ipz,:] - v[ipu[j],:]).mean(0)
				v[ipu[j],:] = v[ipu[j],:] + lr*update
				rs.append(np.linalg.norm(update)/np.sqrt(ndim))

			rz.append(np.mean(rs)) # record average update size.

			if np.mod(i, 10) == 0:
				print('%d %f'%(i, rz[-1]))

		# Evaluate loss (5 times batch size)
		x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		q, ip = tree.query(x)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return V_results[ibest], V_results, Losses, losses, rz

def kmeans_packing_focused(scale_x, offset_x, ndim, n_clusters, n_batch = 3000, n_steps = 1000, n_sim = 1, lr = 0.01):

	V_results = []
	Losses = []

	center_x = ((offset_x[0,0:2] + scale_x[0,0:2]/2.0)).reshape(1,-1)

	a_param = 3.0
	num_thresh = 0.1 # 3.0
	n_up_sample = 10

	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = np.random.rand(n_clusters, ndim)*scale_x + offset_x

			tree = cKDTree(v)
			x = np.random.rand(n_batch, ndim)*scale_x + offset_x

			x_factors = np.arange(0.7, 0.0, -0.1)


			x_sample = np.vstack([np.random.rand(inc*n_batch, ndim)*x_factors[j]*scale_x + offset_x + ((1.0 - x_factors[j])/2.0)*scale_x for inc, j in enumerate(range(len(x_factors)))])

			# x1 = np.random.rand(n_batch, ndim)*0.5*scale_x + offset_x + 0.25*scale_x
			# x2 = np.random.rand(n_batch, ndim)*0.3*scale_x + offset_x + 0.35*scale_x

			x = np.concatenate((x, x_sample), axis = 0)
			x = x[np.random.choice(x.shape[0], size = n_batch, replace = False)]

			q, ip = tree.query(x)

			rs = []
			ipu = np.unique(ip)
			for j in range(len(ipu)):
				ipz = np.where(ip == ipu[j])[0]
				# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
				update = (x[ipz,:] - v[ipu[j],:]).mean(0)
				v[ipu[j],:] = v[ipu[j],:] + lr*update
				rs.append(np.linalg.norm(update)/np.sqrt(ndim))

			rz.append(np.mean(rs)) # record average update size.

			if np.mod(i, 10) == 0:
				print('%d %f'%(i, rz[-1]))

		# Evaluate loss (5 times batch size)
		x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		q, ip = tree.query(x)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return V_results[ibest], V_results, Losses, losses, rz

def kmeans_packing_logarithmic_focused(scale_x, offset_x, ndim, n_clusters, n_batch = 3000, n_steps = 1000, n_sim = 1, lr = 0.01):

	V_results = []
	Losses = []

	center_x = ((offset_x[0,0:2] + scale_x[0,0:2]/2.0)).reshape(1,-1)

	a_param = 3.0
	num_thresh = 0.1 # 3.0
	n_up_sample = 10

	for n in range(n_sim):

		losses, rz = [], []
		for i in range(n_steps):
			if i == 0:
				v = np.random.rand(int(n_clusters/2), ndim)*scale_x + offset_x
				v1 = np.random.rand(int(n_clusters/2), ndim)*scale_x*0.3 + offset_x + 0.35*scale_x
				v = np.concatenate((v, v1), axis = 0)

			tree = cKDTree(v)
			x = np.random.rand(int(n_batch/2), ndim)*scale_x + offset_x
			x1 = np.random.rand(int(n_batch/2), ndim)*scale_x*0.3 + offset_x + 0.35*scale_x
			x = np.concatenate((x, x1), axis = 0)

			x_r = np.random.pareto(a_param, size = 100000)
			ifind = np.where(x_r < num_thresh)[0]
			ifind = np.random.choice(ifind, size = n_up_sample*n_batch)
			x_r = x_r[ifind].reshape(-1,1)/num_thresh
			x_theta = np.random.rand(n_up_sample*n_batch)*2.0*np.pi
			x_xy = x_r*np.concatenate((np.cos(x_theta[:,None]), np.sin(x_theta[:,None])), axis = 1)*(scale_x[:,0:2]/2.0)
			x_xy = np.concatenate((x_xy + center_x, np.random.rand(n_up_sample*n_batch,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
			x = np.concatenate((x, x_xy), axis = 0)
			x = x[np.random.choice(x.shape[0], size = n_batch, replace = False)]

			q, ip = tree.query(x)

			rs = []
			ipu = np.unique(ip)
			for j in range(len(ipu)):
				ipz = np.where(ip == ipu[j])[0]
				# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
				update = (x[ipz,:] - v[ipu[j],:]).mean(0)
				v[ipu[j],:] = v[ipu[j],:] + lr*update
				rs.append(np.linalg.norm(update)/np.sqrt(ndim))

			rz.append(np.mean(rs)) # record average update size.

			if np.mod(i, 10) == 0:
				print('%d %f'%(i, rz[-1]))

		# Evaluate loss (5 times batch size)
		x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
		q, ip = tree.query(x)
		Losses.append(q.mean())
		V_results.append(np.copy(v))

	Losses = np.array(Losses)
	ibest = np.argmin(Losses)

	return V_results[ibest], V_results, Losses, losses, rz

def kmeans_packing_logarithmic_parallel(num_cores, scale_x_list, offset_x_list, ndim, n_clusters, n_batch = 3000, n_steps = 1000, n_sim = 1, lr = 0.01):

	def step_test(args):

		scale_x, offset_x, ndim, n_clusters = args

		V_results = []
		Losses = []

		center_x = ((offset_x[0,0:2] + scale_x[0,0:2]/2.0)).reshape(1,-1)

		a_param = 3.0
		num_thresh = 0.1 # 3.0
		n_up_sample = 10

		for n in range(n_sim):

			losses, rz = [], []
			for i in range(n_steps):
				if i == 0:
					v = np.random.rand(n_clusters, ndim)*scale_x + offset_x

				tree = cKDTree(v)
				x = np.random.rand(n_batch, ndim)*scale_x + offset_x
				x_r = np.random.pareto(a_param, size = 100000)
				ifind = np.where(x_r < num_thresh)[0]
				ifind = np.random.choice(ifind, size = n_up_sample*n_batch)
				x_r = x_r[ifind].reshape(-1,1)/num_thresh
				x_theta = np.random.rand(n_up_sample*n_batch)*2.0*np.pi
				x_xy = x_r*np.concatenate((np.cos(x_theta[:,None]), np.sin(x_theta[:,None])), axis = 1)*(scale_x[:,0:2]/2.0)
				x_xy = np.concatenate((x_xy + center_x, np.random.rand(n_up_sample*n_batch,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
				x = np.concatenate((x, x_xy), axis = 0)
				x = x[np.random.choice(x.shape[0], size = n_batch, replace = False)]

				q, ip = tree.query(x)

				rs = []
				ipu = np.unique(ip)
				for j in range(len(ipu)):
					ipz = np.where(ip == ipu[j])[0]
					# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
					update = (x[ipz,:] - v[ipu[j],:]).mean(0)
					v[ipu[j],:] = v[ipu[j],:] + lr*update
					rs.append(np.linalg.norm(update)/np.sqrt(ndim))

				rz.append(np.mean(rs)) # record average update size.

				if np.mod(i, 10) == 0:
					print('%d %f'%(i, rz[-1]))

			# Evaluate loss (5 times batch size)
			x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
			q, ip = tree.query(x)
			Losses.append(q.mean())
			V_results.append(np.copy(v))

		Losses = np.array(Losses)
		ibest = np.argmin(Losses)

		return V_results[ibest] # , ind

	results = Parallel(n_jobs = num_cores)(delayed(step_test)( [ scale_x_list[i], offset_x_list[i], ndim, n_clusters] ) for i in range(num_cores))

	pos_grid_l = []
	for i in range(num_cores):
		pos_grid_l.append(results[i])

	return pos_grid_l

def kmeans_packing_logarithmic_focused_parallel(num_cores, scale_x_list, offset_x_list, ndim, n_clusters, n_batch = 3000, n_steps = 1000, n_sim = 1, lr = 0.01):

	def step_test(args):

		scale_x, offset_x, ndim, n_clusters = args

		V_results = []
		Losses = []

		center_x = ((offset_x[0,0:2] + scale_x[0,0:2]/2.0)).reshape(1,-1)

		a_param = 3.0
		num_thresh = 0.1 # 3.0
		n_up_sample = 10

		for n in range(n_sim):

			losses, rz = [], []
			for i in range(n_steps):
				if i == 0:
					v = np.random.rand(int(n_clusters/2), ndim)*scale_x + offset_x
					v1 = np.random.rand(int(n_clusters/2), ndim)*scale_x*0.3 + offset_x + 0.35*scale_x
					v = np.concatenate((v, v1), axis = 0)

				tree = cKDTree(v)
				x = np.random.rand(int(n_batch/2), ndim)*scale_x + offset_x
				x1 = np.random.rand(int(n_batch/2), ndim)*scale_x*0.3 + offset_x + 0.35*scale_x
				x = np.concatenate((x, x1), axis = 0)

				x_r = np.random.pareto(a_param, size = 100000)
				ifind = np.where(x_r < num_thresh)[0]
				ifind = np.random.choice(ifind, size = n_up_sample*n_batch)
				x_r = x_r[ifind].reshape(-1,1)/num_thresh
				x_theta = np.random.rand(n_up_sample*n_batch)*2.0*np.pi
				x_xy = x_r*np.concatenate((np.cos(x_theta[:,None]), np.sin(x_theta[:,None])), axis = 1)*(scale_x[:,0:2]/2.0)
				x_xy = np.concatenate((x_xy + center_x, np.random.rand(n_up_sample*n_batch,1)*scale_x[0,2] + offset_x[0,2]), axis = 1)
				x = np.concatenate((x, x_xy), axis = 0)
				x = x[np.random.choice(x.shape[0], size = n_batch, replace = False)]

				q, ip = tree.query(x)

				rs = []
				ipu = np.unique(ip)
				for j in range(len(ipu)):
					ipz = np.where(ip == ipu[j])[0]
					# update = x[ipz,:].mean(0) - v[ipu[j],:] # which update rule?
					update = (x[ipz,:] - v[ipu[j],:]).mean(0)
					v[ipu[j],:] = v[ipu[j],:] + lr*update
					rs.append(np.linalg.norm(update)/np.sqrt(ndim))

				rz.append(np.mean(rs)) # record average update size.

				if np.mod(i, 10) == 0:
					print('%d %f'%(i, rz[-1]))

			# Evaluate loss (5 times batch size)
			x = np.random.rand(n_batch*5, ndim)*scale_x + offset_x
			q, ip = tree.query(x)
			Losses.append(q.mean())
			V_results.append(np.copy(v))

		Losses = np.array(Losses)
		ibest = np.argmin(Losses)

		return V_results[ibest] # , ind

	results = Parallel(n_jobs = num_cores)(delayed(step_test)( [ scale_x_list[i], offset_x_list[i], ndim, n_clusters] ) for i in range(num_cores))

	pos_grid_l = []
	for i in range(num_cores):
		pos_grid_l.append(results[i])

	return pos_grid_l

def make_spatial_graph_cKDTree(pos, k_pos = 15, device = 'cuda'):

	## For every mesh node, link to k pos nodes
	## Note: we could attach all spatial nodes to
	## the nearest mesh nodes, though this seems
	## less natural (as it would introduce
	## very long range connections, linking to
	## a small part of the mesh grid. It could
	## potentially make learning the mapping
	## easier).

	## Can give absolute node locations as features

	n_pos = pos.shape[0]

	# A_edges_mesh = knn(mesh, mesh, k = k + 1).flip(0).contiguous()[0].to(device)

	tree = cKDTree(pos.cpu().detach().numpy())
	edges = tree.query(pos.cpu().detach().numpy(), k = k_pos + 1)[1]
	edges = torch.Tensor(np.hstack([np.concatenate((edges[i].reshape(1,-1), i*np.ones(k_pos + 1).reshape(1,-1)), axis = 0) for i in range(len(pos))])).long().to(device)

	## transfer
	A_edges = remove_self_loops(edges)[0] # .to(device)
	# edges_offset = pos[A_edges[1]] - pos[A_edges[0]]

	return A_edges # , edges_offset

def make_spatial_graph(pos, k_pos = 15, device = 'cuda'):

	## For every mesh node, link to k pos nodes
	## Note: we could attach all spatial nodes to
	## the nearest mesh nodes, though this seems
	## less natural (as it would introduce
	## very long range connections, linking to
	## a small part of the mesh grid. It could
	## potentially make learning the mapping
	## easier).

	## Can give absolute node locations as features

	n_pos = pos.shape[0]

	# A_edges_mesh = knn(mesh, mesh, k = k + 1).flip(0).contiguous()[0].to(device)

	## transfer
	A_edges = remove_self_loops(knn(pos, pos, k = k_pos + 1).flip(0).contiguous())[0] # .to(device)
	# edges_offset = pos[A_edges[1]] - pos[A_edges[0]]

	return A_edges # , edges_offset

def make_bipartite_spatial_graph_cKDTree(pos_recieve, pos_send, k_pos = 15, device = 'cuda'):

	## For every mesh node, link to k pos nodes
	## Note: we could attach all spatial nodes to
	## the nearest mesh nodes, though this seems
	## less natural (as it would introduce
	## very long range connections, linking to
	## a small part of the mesh grid. It could
	## potentially make learning the mapping
	## easier).

	## Can give absolute node locations as features

	# n_pos = pos.shape[0]

	# A_edges_mesh = knn(mesh, mesh, k = k + 1).flip(0).contiguous()[0].to(device)

	tree = cKDTree(pos_recieve.cpu().detach().numpy())
	edges = tree.query(pos_send.cpu().detach().numpy(), k = k_pos)[1]
	A_edges = torch.Tensor(np.hstack([np.concatenate((edges[i].reshape(1,-1), i*np.ones(k_pos).reshape(1,-1)), axis = 0) for i in range(len(pos_send))])).long().flip(0).contiguous().to(device)

	## transfer
	# A_edges = knn(pos_recieve, pos_send, k = k_pos) # .flip(0).contiguous()[0] # .to(device)
	# edges_offset = pos[A_edges[1]] - pos[A_edges[0]]

	return A_edges # , edges_offset

def make_bipartite_spatial_graph(pos_recieve, pos_send, k_pos = 15, device = 'cuda'):

	## For every mesh node, link to k pos nodes
	## Note: we could attach all spatial nodes to
	## the nearest mesh nodes, though this seems
	## less natural (as it would introduce
	## very long range connections, linking to
	## a small part of the mesh grid. It could
	## potentially make learning the mapping
	## easier).

	## Can give absolute node locations as features

	# n_pos = pos.shape[0]

	# A_edges_mesh = knn(mesh, mesh, k = k + 1).flip(0).contiguous()[0].to(device)

	## transfer ## Note: this outputs edges in the flipped "sorted" format
	A_edges = knn(pos_recieve, pos_send, k = k_pos) # .flip(0).contiguous()[0] # .to(device)
	# edges_offset = pos[A_edges[1]] - pos[A_edges[0]]

	return A_edges # , edges_offset

def load_logarithmic_grids(ext_type, n_ver):

	if ext_type == 'local':
		
		pos_grid_l = np.load('D:/Projects/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver)['pos_grid_l']
		pos_grid_l = [torch.Tensor(pos_grid_l[j]) for j in range(pos_grid_l.shape[0])]

	elif ext_type == 'remote':

		pos_grid_l = np.load('/work/wavefront/imcbrear/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver)['pos_grid_l']
		pos_grid_l = [torch.Tensor(pos_grid_l[j]) for j in range(pos_grid_l.shape[0])]

	elif ext_type == 'server':

		pos_grid_l = np.load('/oak/stanford/schools/ees/beroza/imcbrear/Laplace/ApplyHeterogenous/Grids/spatial_grid_logarithmic_heterogenous_ver_%d.npz'%n_ver)['pos_grid_l']
		pos_grid_l = [torch.Tensor(pos_grid_l[j]) for j in range(pos_grid_l.shape[0])]

	return pos_grid_l


def load_batch_data_norm_mesh_enhanced(st_files, shape_vals, params, use_shape_feature = True, use_extra_features = False):

	assert(use_shape_feature == True)

	scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device = params

	pos_slice = []
	signal_slice = []
	query_slice = []
	edges_slice = []
	edges_feature_slice = []
	edges_c_slice = []
	trgt_slice = []

	len_files = len(st_files)

	for i in range(len_files):

		z = h5py.File(st_files[i], 'r')
		if norm_version == 1:
			norm_val_slice = np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]]))
		elif norm_version == 2:
			norm_val_slice = np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max()
		assert((np.log10(norm_val_slice) > min_val)*(np.log10(norm_val_slice) < max_val))
		z.close()

		if use_shape_feature == True:
			
			n_factor_dist = 100.0
			n_factor_embed = 5.0
			x = apply_shape_function(st_files[i], shape_vals)/scale_val

			if use_extra_features == True:
				dz = x.mean(0)[2]
				RMax = pairwise_distances(x).max()

			x = torch.Tensor(x).to(device)

		trgt = np.array([np.log10(norm_val_slice)]).reshape(1,-1)

		if use_extra_features == True:
			signal_slice.append(torch.cat(( x, (torch.ones(x.shape[0],2)*torch.Tensor([dz, RMax]).reshape(1,-1)).to(device) ), dim = 1))
		else:
			signal_slice.append(x) # torch.cat(( pos, (torch.Tensor(inpt)*torch.ones(n_nodes_grid, n_features)).to(device) ), dim = 1).to(device))

		pos_slice.append(x)
		edges_slice.append(make_spatial_graph(x, k_pos = k_spc_edges, device = device))
		# edges_c_slice.append(A_edges_c)
		trgt_slice.append(torch.Tensor(trgt).to(device))

	return pos_slice, signal_slice, edges_slice, trgt_slice


def load_batch_data_displacement_mesh_enhanced(st_files, grid_ind, Lh_and_Lv_vals, pos_grid_l, A_edges_c, A_edges_c_mesh, shape_vals, params, use_shape_feature = True):

	## Must be true for this function, so that the mesh is concatenated into the spatial graph
	assert(use_shape_feature == True)

	if len(grid_ind) == 1:
		grid_ind = grid_ind*np.ones(len(st_files)).astype('int')

	scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device = params

	pos_slice = []
	signal_slice = []
	query_slice = []
	edges_slice = []
	edges_feature_slice = []
	edges_c_slice = []
	trgt_slice = []

	len_files = len(st_files)

	for i in range(len_files):

		z = h5py.File(st_files[i], 'r')

		# Trgts
		Lh = z['input/Lh'][:][0][0]/scale_val
		Lv = z['input/Lv'][:][0][0]/scale_val

		# Inpts
		dz = z['input/dz'][:][0][0]/scale_val # depth? (scalar)
		f1 = z['input/fs'][:]
		if f1.dtype.kind == 'f':
			Fs = np.transpose(np.concatenate((z['input/fs'][:].reshape(1,-1), np.zeros((1, z['input/fs'][:].shape[1]))), axis = 1))
		else:
			Fs = np.transpose(np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1))
			
		# Fs = np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1)
		NormF = z['input/normF'][:].reshape(-1)
		RMax = z['input/rmax'][:].reshape(-1)

		if norm_version == 1:
			norm_val_slice = np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]]))
		elif norm_version == 2:
			norm_val_slice = np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max()

		assert((np.log10(norm_val_slice) > min_val)*(np.log10(norm_val_slice) < max_val))

		Ux = z['output/Ux'][0,0:n_samples]/norm_val_slice # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
		Uy = z['output/Uy'][0,0:n_samples]/norm_val_slice # [1 x 1440]
		Uz = z['output/Uz'][0,0:n_samples]/norm_val_slice # [1 x 1440]
		X = z['output/X'][0,0:n_samples]/scale_val # [1 x 1440]
		Y = z['output/Y'][0,0:n_samples]/scale_val # [1 x 1440]
		Z = z['output/Z'][0,0:n_samples]/scale_val # [1 x 1440]
		z.close()

		trgt = np.concatenate((Ux[:,None], Uy[:,None], Uz[:,None]), axis = 1)
		x_query = np.concatenate((X[:,None], Y[:,None], Z[:,None]), axis = 1)
		inpt = np.concatenate([np.array([dz]), Fs.reshape(-1), NormF, RMax], axis = 0).reshape(1,-1)/norm_vals


		## Scale spatial graphs (but for Lh and Lv model it is constant) 
		pos = np.array([Lh_and_Lv_vals[i][0]/2.0, Lh_and_Lv_vals[i][0]/2.0, Lh_and_Lv_vals[i][1]]).reshape(1,-1) # .to(device) # *pos_grid
		pos = torch.Tensor(pos*pos_grid_l[grid_ind[i]]).to(device)

		if use_shape_feature == True:
			n_factor_dist = 100.0
			n_factor_embed = 5.0
			x = apply_shape_function(st_files[i], shape_vals)/scale_val
			tree = cKDTree(x)
			dist = tree.query(pos.cpu().detach().numpy())[0]/(RMax/scale_val)/n_factor_dist # /np.max(Lh_and_Lv_vals[i])
			dist_mesh = tree.query(x)[0]/(RMax/scale_val)/n_factor_dist # /np.max(Lh_and_Lv_vals[i])

			inpt_extend = torch.Tensor(np.concatenate((dist.reshape(-1,1), np.exp(-0.5*(dist.reshape(-1,1)**2)/((n_factor_embed*RMax/scale_val)**2))), axis = 1)).to(device)
			inpt_extend_mesh = torch.Tensor(np.concatenate((dist_mesh.reshape(-1,1), np.exp(-0.5*(dist_mesh.reshape(-1,1)**2)/((n_factor_embed*RMax/scale_val)**2))), axis = 1)).to(device)

			x = torch.Tensor(x).to(device)

		## Make mesh indicator signal
		signal_feature = torch.cat((torch.zeros(pos.shape[0]), torch.ones(x.shape[0])), dim = 0).reshape(-1,1).to(device)

		## Append concatenation of spatial graph and mesh
		pos_slice.append(torch.cat((pos, x), dim = 0))

		## Note: removing absolute position from input feature

		## Maybe should remove absolute position from input feature
		signal_slice.append(torch.cat(( torch.cat((pos, x), dim = 0), torch.cat((inpt_extend, inpt_extend_mesh), dim = 0), signal_feature ), dim = 1).to(device))
		# signal_slice.append(torch.cat(( torch.cat((inpt_extend, inpt_extend_mesh), dim = 0), signal_feature ), dim = 1).to(device))

		query_slice.append(torch.Tensor(x_query).to(device))
		# edges_slice = [A_edges_l[j] for j in igrids]

		edges_spatial_graph = make_spatial_graph(pos, k_pos = k_spc_edges, device = device)
		edges_mesh_graph = make_spatial_graph(x, k_pos = k_spc_edges, device = device)
		edges_mesh_to_spatial_graph = make_bipartite_spatial_graph(pos, x, k_pos = k_spc_edges, device = device)

		## Increment edges by the correct offset
		edges_mesh_graph = edges_mesh_graph + pos.shape[0]
		edges_mesh_to_spatial_graph[0] = edges_mesh_to_spatial_graph[0] + pos.shape[0]
		edges_merged = torch.cat((edges_spatial_graph, edges_mesh_graph, edges_mesh_to_spatial_graph), dim = 1)
		edges_feature = torch.cat((torch.zeros(edges_spatial_graph.shape[1]), 2.0*torch.ones(edges_mesh_graph.shape[1]), torch.ones(edges_mesh_to_spatial_graph.shape[1])), dim = 0).reshape(-1,1).to(device)

		edges_slice.append(edges_merged)
		edges_feature_slice.append(edges_feature)

		# self_loops_c = torch.arange(x.shape[0]).reshape(1,-1).repeat(2,1).to(device) + pos.shape[0]
		edges_c_slice.append(torch.cat((A_edges_c, A_edges_c_mesh + pos.shape[0]), dim = 1))
		trgt_slice.append(torch.Tensor(trgt).to(device))

	return pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, trgt_slice


def load_batch_data_displacement_mesh_enhanced_both_edges(st_files, grid_ind, Lh_and_Lv_vals, pos_grid_l, A_edges_c, A_edges_c_mesh, shape_vals, params, use_shape_feature = True):

	## Must be true for this function, so that the mesh is concatenated into the spatial graph
	assert(use_shape_feature == True)

	if len(grid_ind) == 1:
		grid_ind = grid_ind*np.ones(len(st_files)).astype('int')

	scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device = params

	pos_slice = []
	signal_slice = []
	query_slice = []
	edges_slice = []
	edges_feature_slice = []
	edges_c_slice = []
	trgt_slice = []

	len_files = len(st_files)

	for i in range(len_files):

		z = h5py.File(st_files[i], 'r')

		# Trgts
		Lh = z['input/Lh'][:][0][0]/scale_val
		Lv = z['input/Lv'][:][0][0]/scale_val

		# Inpts
		dz = z['input/dz'][:][0][0]/scale_val # depth? (scalar)
		f1 = z['input/fs'][:]
		if f1.dtype.kind == 'f':
			Fs = np.transpose(np.concatenate((z['input/fs'][:].reshape(1,-1), np.zeros((1, z['input/fs'][:].shape[1]))), axis = 1))
		else:
			Fs = np.transpose(np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1))
			
		# Fs = np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1)
		NormF = z['input/normF'][:].reshape(-1)
		RMax = z['input/rmax'][:].reshape(-1)

		if norm_version == 1:
			norm_val_slice = np.linalg.norm(np.array([z['output/Ux'][0,0], z['output/Uy'][0,0], z['output/Uz'][0,0]]))
		elif norm_version == 2:
			norm_val_slice = np.linalg.norm(np.concatenate((z['output/Ux'][0,0:n_samples][:,None], z['output/Uy'][0,0:n_samples][:,None], z['output/Uz'][0,0:n_samples][:,None]), axis = 1), axis = 1).max()

		assert((np.log10(norm_val_slice) > min_val)*(np.log10(norm_val_slice) < max_val))

		Ux = z['output/Ux'][0,0:n_samples]/norm_val_slice # [1 x 1440] ## X, Y, Z, and Ux, Uy, Uz, are displacement field coordinates and vectors
		Uy = z['output/Uy'][0,0:n_samples]/norm_val_slice # [1 x 1440]
		Uz = z['output/Uz'][0,0:n_samples]/norm_val_slice # [1 x 1440]
		X = z['output/X'][0,0:n_samples]/scale_val # [1 x 1440]
		Y = z['output/Y'][0,0:n_samples]/scale_val # [1 x 1440]
		Z = z['output/Z'][0,0:n_samples]/scale_val # [1 x 1440]
		z.close()

		trgt = np.concatenate((Ux[:,None], Uy[:,None], Uz[:,None]), axis = 1)
		x_query = np.concatenate((X[:,None], Y[:,None], Z[:,None]), axis = 1)
		inpt = np.concatenate([np.array([dz]), Fs.reshape(-1), NormF, RMax], axis = 0).reshape(1,-1)/norm_vals


		## Scale spatial graphs (but for Lh and Lv model it is constant) 
		pos = np.array([Lh_and_Lv_vals[i][0]/2.0, Lh_and_Lv_vals[i][0]/2.0, Lh_and_Lv_vals[i][1]]).reshape(1,-1) # .to(device) # *pos_grid
		pos = torch.Tensor(pos*pos_grid_l[grid_ind[i]]).to(device)

		if use_shape_feature == True:
			n_factor_dist = 100.0
			n_factor_embed = 5.0
			x = apply_shape_function(st_files[i], shape_vals)/scale_val
			tree = cKDTree(x)
			dist = tree.query(pos.cpu().detach().numpy())[0]/(RMax/scale_val)/n_factor_dist # /np.max(Lh_and_Lv_vals[i])
			dist_mesh = tree.query(x)[0]/(RMax/scale_val)/n_factor_dist # /np.max(Lh_and_Lv_vals[i])

			inpt_extend = torch.Tensor(np.concatenate((dist.reshape(-1,1), np.exp(-0.5*(dist.reshape(-1,1)**2)/((n_factor_embed*RMax/scale_val)**2))), axis = 1)).to(device)
			inpt_extend_mesh = torch.Tensor(np.concatenate((dist_mesh.reshape(-1,1), np.exp(-0.5*(dist_mesh.reshape(-1,1)**2)/((n_factor_embed*RMax/scale_val)**2))), axis = 1)).to(device)

			x = torch.Tensor(x).to(device)

		## Make mesh indicator signal
		signal_feature = torch.cat((torch.zeros(pos.shape[0]), torch.ones(x.shape[0])), dim = 0).reshape(-1,1).to(device)

		## Append concatenation of spatial graph and mesh
		pos_slice.append(torch.cat((pos, x), dim = 0))

		## Note: removing absolute position from input feature

		## Maybe should remove absolute position from input feature
		signal_slice.append(torch.cat(( torch.cat((pos, x), dim = 0), torch.cat((inpt_extend, inpt_extend_mesh), dim = 0), signal_feature ), dim = 1).to(device))

		query_slice.append(torch.Tensor(x_query).to(device))
		# edges_slice = [A_edges_l[j] for j in igrids]

		edges_spatial_graph = make_spatial_graph(pos, k_pos = k_spc_edges, device = device)
		edges_mesh_graph = make_spatial_graph(x, k_pos = k_spc_edges, device = device)
		edges_mesh_to_spatial_graph = make_bipartite_spatial_graph(pos, x, k_pos = k_spc_edges, device = device)

		## Increment edges by the correct offset
		edges_mesh_graph = edges_mesh_graph + pos.shape[0]
		edges_mesh_to_spatial_graph[0] = edges_mesh_to_spatial_graph[0] + pos.shape[0]

		# Flipping edges
		edges_mesh_to_spatial_graph_flipped = make_bipartite_spatial_graph(x, pos, k_pos = k_spc_edges, device = device)
		edges_mesh_to_spatial_graph_flipped[1] = edges_mesh_to_spatial_graph_flipped[1] + pos.shape[0]
		edges_mesh_to_spatial_graph_flipped = edges_mesh_to_spatial_graph_flipped.flip(0).contiguous()
		edges_merged = torch.cat((edges_spatial_graph, edges_mesh_graph, edges_mesh_to_spatial_graph, edges_mesh_to_spatial_graph_flipped), dim = 1)


		edges_feature = torch.cat((torch.zeros(edges_spatial_graph.shape[1]), 2.0*torch.ones(edges_mesh_graph.shape[1]), 0.5*torch.ones(edges_mesh_to_spatial_graph.shape[1]), torch.ones(edges_mesh_to_spatial_graph_flipped.shape[1])), dim = 0).reshape(-1,1).to(device)


		edges_slice.append(edges_merged)
		edges_feature_slice.append(edges_feature)

		# self_loops_c = torch.arange(x.shape[0]).reshape(1,-1).repeat(2,1).to(device) + pos.shape[0]
		edges_c_slice.append(torch.cat((A_edges_c, A_edges_c_mesh + pos.shape[0]), dim = 1))
		trgt_slice.append(torch.Tensor(trgt).to(device))

	return pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, trgt_slice


def load_batch_data_Lh_and_Lv_mesh_enhanced(st_files, shape_vals, params, use_shape_feature = True, use_extra_features = False):

	assert(use_shape_feature == True)

	scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device = params

	pos_slice = []
	signal_slice = []
	query_slice = []
	edges_slice = []
	edges_feature_slice = []
	edges_c_slice = []
	trgt_slice = []

	len_files = len(st_files)

	for i in range(len_files):

		z = h5py.File(st_files[i], 'r')
		# Trgts
		Lh = z['input/Lh'][:][0][0]/scale_val
		Lv = z['input/Lv'][:][0][0]/scale_val
		z.close()

		if use_shape_feature == True:
			# n_factor_dist = 100.0
			# n_factor_embed = 5.0
			x = apply_shape_function(st_files[i], shape_vals)/scale_val

			if use_extra_features == True:
				dz = x.mean(0)[2]
				RMax = pairwise_distances(x).max()

			x = torch.Tensor(x).to(device)

		trgt = np.array([Lh, Lv]).reshape(1,-1)

		if use_extra_features == True:
			signal_slice.append(torch.cat(( x, (torch.ones(x.shape[0],2)*torch.Tensor([dz, RMax]).reshape(1,-1)).to(device) ), dim = 1))
		else:
			signal_slice.append(x) # torch.cat(( pos, (torch.Tensor(inpt)*torch.ones(n_nodes_grid, n_features)).to(device) ), dim = 1).to(device))

		pos_slice.append(x)
		edges_slice.append(make_spatial_graph(x, k_pos = k_spc_edges, device = device))
		# edges_c_slice.append(A_edges_c)
		trgt_slice.append(torch.Tensor(trgt).to(device))

	return pos_slice, signal_slice, edges_slice, trgt_slice

def assemble_batch_data_norm_mesh_enhanced(dz_list, Fs_list, NormF_list, RMax_list, shape_vals, params, use_extra_features = False):

	scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device = params

	pos_slice = []
	signal_slice = []
	edges_slice = []

	len_files = len(dz_list)

	for i in range(len_files):

		n_factor_dist = 100.0
		n_factor_embed = 5.0
		x = apply_shape_function_direct(dz_list[i], Fs_list[i], NormF_list[i], RMax_list[i], shape_vals)/scale_val

		x = torch.Tensor(x).to(device)

		if use_extra_features == True:
			signal_slice.append(torch.cat(( x, (torch.ones(x.shape[0],2)*torch.Tensor([dz, RMax]).reshape(1,-1)).to(device) ), dim = 1))
		else:
			signal_slice.append(x) # torch.cat(( pos, (torch.Tensor(inpt)*torch.ones(n_nodes_grid, n_features)).to(device) ), dim = 1).to(device))

		pos_slice.append(x)

		edges_slice.append(make_spatial_graph(x, k_pos = k_spc_edges, device = device))

	return pos_slice, signal_slice, edges_slice # , edges_c_slice, trgt_slice

def assemble_batch_data_displacement_mesh_enhanced(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, Lh_and_Lv_vals, pos_grid_l, A_edges_c, A_edges_c_mesh, shape_vals, params, use_shape_feature = True):

	if isinstance(grid_ind, int):
		grid_ind = (grid_ind*np.ones(len(dz_list))).astype('int')
	elif len(grid_ind) == 1:
		grid_ind = (grid_ind*np.ones(len(dz_list))).astype('int')
		
	scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device = params

	pos_slice = []
	signal_slice = []
	query_slice = []
	edges_slice = []
	edges_c_slice = []
	edges_feature_slice = []
	trgt_slice = []

	len_files = len(dz_list)

	for i in range(len_files):

		# trgt = np.concatenate((Ux[:,None], Uy[:,None], Uz[:,None]), axis = 1)
		x_query = X_query_list[i]/scale_val # np.concatenate((X[:,None], Y[:,None], Z[:,None]), axis = 1)
		# inpt = np.concatenate((np.array([dz_list[i]/scale_val]), Fs_list[i].reshape(-1), np.array([NormF_list[i]]), np.array([RMax_list[i]])), axis = 0).reshape(1,-1)/norm_vals


		## Scale spatial graphs (but for Lh and Lv model it is constant) 
		pos = np.array([Lh_and_Lv_vals[i][0]/2.0, Lh_and_Lv_vals[i][0]/2.0, Lh_and_Lv_vals[i][1]]).reshape(1,-1) # .to(device) # *pos_grid
		pos = torch.Tensor(pos*pos_grid_l[grid_ind[i]]).to(device)

		if use_shape_feature == True:

			n_factor_dist = 100.0
			n_factor_embed = 5.0
			x = apply_shape_function_direct(dz_list[i], Fs_list[i], NormF_list[i], RMax_list[i], shape_vals)/scale_val
			tree = cKDTree(x)
			dist = tree.query(pos.cpu().detach().numpy())[0]/(RMax_list[i]/scale_val)/n_factor_dist # /np.max(Lh_and_Lv_vals[i])
			dist_mesh = tree.query(x)[0]/(RMax_list[i]/scale_val)/n_factor_dist # /np.max(Lh_and_Lv_vals[i])

			inpt_extend = torch.Tensor(np.concatenate((dist.reshape(-1,1), np.exp(-0.5*(dist.reshape(-1,1)**2)/((n_factor_embed*RMax_list[i]/scale_val)**2))), axis = 1)).to(device)
			inpt_extend_mesh = torch.Tensor(np.concatenate((dist_mesh.reshape(-1,1), np.exp(-0.5*(dist_mesh.reshape(-1,1)**2)/((n_factor_embed*RMax_list[i]/scale_val)**2))), axis = 1)).to(device)

			x = torch.Tensor(x).to(device)

		## Make mesh indicator signal
		signal_feature = torch.cat((torch.zeros(pos.shape[0]), torch.ones(x.shape[0])), dim = 0).reshape(-1,1).to(device)

		## Append concatenation of spatial graph and mesh
		pos_slice.append(torch.cat((pos, x), dim = 0))

		## Maybe should remove absolute position from input feature
		signal_slice.append(torch.cat(( torch.cat((pos, x), dim = 0), torch.cat((inpt_extend, inpt_extend_mesh), dim = 0), signal_feature ), dim = 1).to(device))
		# signal_slice.append(torch.cat(( torch.cat((inpt_extend, inpt_extend_mesh), dim = 0), signal_feature ), dim = 1).to(device))

		query_slice.append(torch.Tensor(x_query).to(device))
		# edges_slice = [A_edges_l[j] for j in igrids]

		edges_spatial_graph = make_spatial_graph(pos, k_pos = k_spc_edges, device = device)
		edges_mesh_graph = make_spatial_graph(x, k_pos = k_spc_edges, device = device)
		edges_mesh_to_spatial_graph = make_bipartite_spatial_graph(pos, x, k_pos = k_spc_edges, device = device)
		## recieve: pos (spatial graph), send: x (mesh)

		## Increment edges by the correct offset
		edges_mesh_graph = edges_mesh_graph + pos.shape[0]
		edges_mesh_to_spatial_graph[0] = edges_mesh_to_spatial_graph[0] + pos.shape[0]
		edges_merged = torch.cat((edges_spatial_graph, edges_mesh_graph, edges_mesh_to_spatial_graph), dim = 1)
		edges_feature = torch.cat((torch.zeros(edges_spatial_graph.shape[1]), 2.0*torch.ones(edges_mesh_graph.shape[1]), torch.ones(edges_mesh_to_spatial_graph.shape[1])), dim = 0).reshape(-1,1).to(device)

		edges_slice.append(edges_merged)
		edges_feature_slice.append(edges_feature)

		# self_loops_c = torch.arange(x.shape[0]).reshape(1,-1).repeat(2,1).to(device) + pos.shape[0]
		edges_c_slice.append(torch.cat((A_edges_c, A_edges_c_mesh + pos.shape[0]), dim = 1))
		# trgt_slice.append(torch.Tensor(trgt).to(device))

	return pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice # , trgt_slice

def assemble_batch_data_displacement_mesh_enhanced_both_edges(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, Lh_and_Lv_vals, pos_grid_l, A_edges_c, A_edges_c_mesh, shape_vals, params, use_shape_feature = True):

	if isinstance(grid_ind, int):
		grid_ind = (grid_ind*np.ones(len(dz_list))).astype('int')
	elif len(grid_ind) == 1:
		grid_ind = (grid_ind*np.ones(len(dz_list))).astype('int')
		
	scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device = params

	pos_slice = []
	signal_slice = []
	query_slice = []
	edges_slice = []
	edges_c_slice = []
	edges_feature_slice = []
	trgt_slice = []

	len_files = len(dz_list)

	for i in range(len_files):

		# trgt = np.concatenate((Ux[:,None], Uy[:,None], Uz[:,None]), axis = 1)
		x_query = X_query_list[i]/scale_val # np.concatenate((X[:,None], Y[:,None], Z[:,None]), axis = 1)
		# inpt = np.concatenate((np.array([dz_list[i]/scale_val]), Fs_list[i].reshape(-1), np.array([NormF_list[i]]), np.array([RMax_list[i]])), axis = 0).reshape(1,-1)/norm_vals


		## Scale spatial graphs (but for Lh and Lv model it is constant) 
		pos = np.array([Lh_and_Lv_vals[i][0]/2.0, Lh_and_Lv_vals[i][0]/2.0, Lh_and_Lv_vals[i][1]]).reshape(1,-1) # .to(device) # *pos_grid
		pos = torch.Tensor(pos*pos_grid_l[grid_ind[i]]).to(device)

		if use_shape_feature == True:

			n_factor_dist = 100.0
			n_factor_embed = 5.0
			x = apply_shape_function_direct(dz_list[i], Fs_list[i], NormF_list[i], RMax_list[i], shape_vals)/scale_val
			tree = cKDTree(x)
			dist = tree.query(pos.cpu().detach().numpy())[0]/(RMax_list[i]/scale_val)/n_factor_dist # /np.max(Lh_and_Lv_vals[i])
			dist_mesh = tree.query(x)[0]/(RMax_list[i]/scale_val)/n_factor_dist # /np.max(Lh_and_Lv_vals[i])

			inpt_extend = torch.Tensor(np.concatenate((dist.reshape(-1,1), np.exp(-0.5*(dist.reshape(-1,1)**2)/((n_factor_embed*RMax_list[i]/scale_val)**2))), axis = 1)).to(device)
			inpt_extend_mesh = torch.Tensor(np.concatenate((dist_mesh.reshape(-1,1), np.exp(-0.5*(dist_mesh.reshape(-1,1)**2)/((n_factor_embed*RMax_list[i]/scale_val)**2))), axis = 1)).to(device)

			x = torch.Tensor(x).to(device)

		## Make mesh indicator signal
		signal_feature = torch.cat((torch.zeros(pos.shape[0]), torch.ones(x.shape[0])), dim = 0).reshape(-1,1).to(device)

		## Append concatenation of spatial graph and mesh
		pos_slice.append(torch.cat((pos, x), dim = 0))

		## Maybe should remove absolute position from input feature
		signal_slice.append(torch.cat(( torch.cat((pos, x), dim = 0), torch.cat((inpt_extend, inpt_extend_mesh), dim = 0), signal_feature ), dim = 1).to(device))
		# signal_slice.append(torch.cat(( torch.cat((inpt_extend, inpt_extend_mesh), dim = 0), signal_feature ), dim = 1).to(device))

		query_slice.append(torch.Tensor(x_query).to(device))
		# edges_slice = [A_edges_l[j] for j in igrids]

		edges_spatial_graph = make_spatial_graph(pos, k_pos = k_spc_edges, device = device)
		edges_mesh_graph = make_spatial_graph(x, k_pos = k_spc_edges, device = device)


		edges_mesh_to_spatial_graph = make_bipartite_spatial_graph(pos, x, k_pos = k_spc_edges, device = device)

		## Increment edges by the correct offset
		edges_mesh_graph = edges_mesh_graph + pos.shape[0]
		edges_mesh_to_spatial_graph[0] = edges_mesh_to_spatial_graph[0] + pos.shape[0]

		# Flipping edges
		edges_mesh_to_spatial_graph_flipped = make_bipartite_spatial_graph(x, pos, k_pos = k_spc_edges, device = device)
		edges_mesh_to_spatial_graph_flipped[1] = edges_mesh_to_spatial_graph_flipped[1] + pos.shape[0]
		edges_mesh_to_spatial_graph_flipped = edges_mesh_to_spatial_graph_flipped.flip(0).contiguous()
		edges_merged = torch.cat((edges_spatial_graph, edges_mesh_graph, edges_mesh_to_spatial_graph, edges_mesh_to_spatial_graph_flipped), dim = 1)

		edges_feature = torch.cat((torch.zeros(edges_spatial_graph.shape[1]), 2.0*torch.ones(edges_mesh_graph.shape[1]), 0.5*torch.ones(edges_mesh_to_spatial_graph.shape[1]), torch.ones(edges_mesh_to_spatial_graph_flipped.shape[1])), dim = 0).reshape(-1,1).to(device)

		edges_slice.append(edges_merged)
		edges_feature_slice.append(edges_feature)

		# self_loops_c = torch.arange(x.shape[0]).reshape(1,-1).repeat(2,1).to(device) + pos.shape[0]
		edges_c_slice.append(torch.cat((A_edges_c, A_edges_c_mesh + pos.shape[0]), dim = 1))
		# trgt_slice.append(torch.Tensor(trgt).to(device))

	return pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice # , trgt_slice

def assemble_batch_data_Lh_and_Lv_mesh_enhanced(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, params):

	assert(use_shape_feature == True)

	scale_val, n_nodes_grid, n_features, n_samples, min_val, max_val, k_spc_edges, norm_version, norm_vals, device = params

	pos_slice = []
	signal_slice = []
	query_slice = []
	edges_slice = []
	edges_feature_slice = []
	edges_c_slice = []
	trgt_slice = []

	len_files = len(st_files)

	for i in range(len_files):

		# z.close()

		# trgt = np.array([Lh, Lv]).reshape(1,-1)
		x_query = X_query_list[i]/scale_val # np.concatenate((X[:,None], Y[:,None], Z[:,None]), axis = 1)
		inpt = np.concatenate((np.array([dz_list[i]/scale_val]), Fs_list[i].reshape(-1), np.array([NormF_list[i]]), np.array([RMax_list[i]])), axis = 0).reshape(1,-1)/norm_vals

		## Scale spatial graphs (but for Lh and Lv model it is constant) 
		pos = np.array([1.0, 1.0, 1.0]).reshape(1,-1) # .to(device) # *pos_grid
		pos = torch.Tensor(pos*pos_grid_l[grid_ind[i]]).to(device)

		pos_slice.append(pos)
		signal_slice.append(torch.cat(( pos, (torch.Tensor(inpt)*torch.ones(n_nodes_grid, n_features)).to(device) ), dim = 1).to(device))
		query_slice.append(torch.Tensor(x_query).to(device))
		# edges_slice = [A_edges_l[j] for j in igrids]
		edges_slice.append(make_spatial_graph(pos, k_pos = k_spc_edges, device = device))
		edges_c_slice.append(A_edges_c)

	return pos_slice, signal_slice, query_slice, edges_slice, edges_c_slice, trgt_slice


def apply_shape_function(s, shape_vals):

	ls, ms, TTA, PHI = shape_vals

	# For heterogeneous spherical harmonic shapes, model parameters are:
	z = h5py.File(s, 'r')
	dp2mu = z['input/dp2mu'][:][0][0]
	dz = z['input/dz'][:][0][0]
	# fs = np.transpose(np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1))
	# fs = np.transpose(np.concatenate((np.real(z['input/fs']).reshape(1,-1), np.imag(z['input/fs']).reshape(1,-1)), axis = 1)).astype('complex128')


	f1 = z['input/fs'][:]
	if f1.dtype.kind == 'f':
		fs = np.transpose(np.concatenate((z['input/fs'][:].reshape(1,-1), np.zeros((1, z['input/fs'][:].shape[1]))), axis = 1))
	else:
		fs = np.transpose(np.concatenate((z['input/fs']['real'].reshape(1,-1), z['input/fs']['imag'].reshape(1,-1)), axis = 1))


	normF = z['input/normF'][:].reshape(-1)[0]
	RMax = z['input/rmax'][:].reshape(-1)[0]

	fs_plt = fs[0:np.size(ls)] + 1j*fs[np.size(ls):]# re-create complex fs vector for plotting

	# t0 = Ti.time()
	X_s, Y_s, Z_s = cmp_SH_shape(ls, ms, fs_plt, RMax, dz, TTA, PHI)
	z.close()

	return np.concatenate((X_s.reshape(-1,1), Y_s.reshape(-1,1), Z_s.reshape(-1,1)), axis = 1) # /scale_val

def apply_shape_function_direct(dz, fs, normF, RMax, shape_vals):

	ls, ms, TTA, PHI = shape_vals

	fs_plt = fs[0:np.size(ls)] + 1j*fs[np.size(ls):]# re-create complex fs vector for plotting

	# t0 = Ti.time()
	X_s, Y_s, Z_s = cmp_SH_shape(ls, ms, fs_plt, RMax, dz, TTA, PHI)
	# z.close()

	return np.concatenate((X_s.reshape(-1,1), Y_s.reshape(-1,1), Z_s.reshape(-1,1)), axis = 1) # /scale_val

def compute_shape_distance(pos, x, sig = 1.0):

	tree = cKDTree(x)
	dist = tree.query(pos)[0]

	return np.exp(-0.5*(dist**2)/(sig**2)), dist

def load_spherical_harmonic_parameters():

	lmax = 5 # maximum degree used to train emulator
	ls, ms = generateDegreeOrder(lmax)
	ls = ls[np.where(ms >= 0)]
	ms = ms[np.where(ms >= 0)]

	tta = np.linspace(0, np.pi, 50)
	phi = np.linspace(0, 2*np.pi, 50)
	PHI, TTA = np.meshgrid(phi, tta)

	return ls, ms, tta, phi, PHI, TTA, lmax

