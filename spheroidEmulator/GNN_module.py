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
import time

from torch import nn, optim
from torch_cluster import knn
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.utils import to_networkx, to_undirected, from_networkx
from torch_geometric.transforms import FaceToEdge, GenerateMeshNormals
from scipy.io import loadmat
from scipy.special import sph_harm
from scipy.spatial import cKDTree
from torch.autograd import Variable
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

import GNN_functions as GNN_f

n_ver_load = 1 # 45
n_ver_load_norm = 1 # 44
n_ver_load_grid = 1
n_ver_load_hv = 1

device = torch.device('cpu')

curr_dir = os.getcwd()

class BipartiteGraphOperator(MessagePassing):
	def __init__(self, ndim_in, ndim_out, n_hidden = 15, ndim_edges = 3):
		super(BipartiteGraphOperator, self).__init__('mean') # Use mean aggregation, not sum.
		# include a single projection map
		self.fc1 = nn.Linear(ndim_in + ndim_edges, n_hidden)
		self.fc2 = nn.Linear(n_hidden, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.

	def forward(self, inpt, edges_grid_in_prod, edges_grid_in_prod_offsets, n_grid, n_mesh):

		N = n_grid*n_mesh
		M = n_grid

		return self.activate2(self.fc2(self.propagate(edges_grid_in_prod, size = (N, M), x = inpt, edge_offsets = edges_grid_in_prod_offsets)))

	def message(self, x_j, edge_offsets):

		return self.activate1(self.fc1(torch.cat((x_j, edge_offsets), dim = 1)))

class BipartiteGraphOperatorDirect(MessagePassing):
	def __init__(self, ndim_in, ndim_out, n_hidden = 30, ndim_edges = 3):
		super(BipartiteGraphOperatorDirect, self).__init__('mean') # Use mean aggregation, not sum.
		# include a single projection map
		self.fc1 = nn.Linear(ndim_in + ndim_edges, n_hidden)
		self.fc2 = nn.Linear(n_hidden, ndim_out) # added additional layer

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.

	def forward(self, inpt, edges_grid_in_prod, edges_grid_in_prod_offsets, n_grid, n_mesh):

		N = n_mesh
		M = n_grid

		return self.activate2(self.fc2(self.propagate(edges_grid_in_prod, size = (N, M), x = inpt, edge_offsets = edges_grid_in_prod_offsets)))

	def message(self, x_j, edge_offsets):

		return self.activate1(self.fc1(torch.cat((x_j, edge_offsets), dim = 1)))

class SpatialAggregation(MessagePassing): # make equivelent version with sum operations. (no need for "complex" concatenation version). Assuming concat version is worse/better?
	def __init__(self, in_channels, out_channels, scale_rel = 1.0, n_dim = 3, n_global = 3, n_hidden = 15, n_mask = 6):
		super(SpatialAggregation, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?
		self.fedges = nn.Linear(n_dim, n_hidden)
		self.fc1 = nn.Linear(in_channels + n_hidden + n_global + n_mask, n_hidden)
		self.fc2 = nn.Linear(n_hidden + in_channels + n_global + n_mask, out_channels) ## NOTE: correcting out_channels, here.
		self.fglobal = nn.Linear(in_channels, n_global)
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.activate4 = nn.PReLU()
		self.scale_rel = scale_rel

	def forward(self, x, mask, A_edges, pos, batch, n_nodes):

		# Because inputs are batched, the "global pool" needs to be applied per disjoint graph
		# Each nodes "global graph" index is assigned in the variable batch
		global_pool = GNN_f.global_mean_pool(self.activate3(self.fglobal(x)), batch) # Could do max-pool
		global_pool_repeat = global_pool.repeat_interleave(n_nodes, dim = 0)

		return self.activate2(self.fc2(torch.cat((x, mask, self.propagate(A_edges, x = torch.cat((x, mask, global_pool_repeat), dim = 1), pos = pos/self.scale_rel), global_pool_repeat), dim = -1)))

	def message(self, x_j, pos_i, pos_j):

		return self.activate1(self.fc1(torch.cat((x_j, self.activate4(self.fedges(pos_i - pos_j))), dim = -1))) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.


## Note, adding one additional fcn, to the read-out layer
class SpatialAttention(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim = 3, n_latent = 15, scale_rel = 1.0, n_hidden = 15, n_heads = 5, device = 'cpu'):
		super(SpatialAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
		self.fedges = nn.Linear(n_dim, n_hidden)
		self.f_context = nn.Linear(inpt_dim + n_hidden, n_heads*n_latent) # add second layer transformation.
		self.f_values = nn.Linear(inpt_dim + n_hidden, n_heads*n_latent) # add second layer transformation.
		self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.proj1 = nn.Linear(n_latent, n_latent) # can remove this layer possibly.
		self.proj2 = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.device = device

	def forward(self, inpt, x_query, x_context, batch, batch_query, n_nodes, k = 10): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

		## Note, make sure this batched version of knn is working
		## for the disjoint, input graphs.

		edge_index = knn(x_context.cpu(), x_query.cpu(), k = k, batch_x = batch.cpu(), batch_y = batch_query.cpu()).flip(0).to(self.device) # Can add batch
		edge_attr = self.activate3(self.fedges((x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel)) # /scale_x

		return self.proj2(self.activate2(self.proj1(self.propagate(edge_index, x = inpt, edge_attr = edge_attr, size = (x_context.shape[0], x_query.shape[0])).mean(1)))) # mean over different heads

	def message(self, x_j, index, edge_attr):

		context_embed = self.f_context(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		value_embed = self.f_values(torch.cat((x_j, edge_attr), dim = -1)).view(-1, self.n_heads, self.n_latent)
		alpha = self.activate1((self.param_vector*context_embed).sum(-1)/self.scale)

		alpha = softmax(alpha, index)

		return alpha.unsqueeze(-1)*value_embed


class BipartiteReadOut(MessagePassing):
	def __init__(self, ndim_in, ndim_out, n_hidden = 15, ndim_edges = 3, scale_rel = 1.0):
		super(BipartiteReadOut, self).__init__('mean') # Use mean aggregation, not sum.
		# include a single projection map
		self.fc1 = nn.Linear(ndim_in + ndim_edges, n_hidden)
		self.fc2 = nn.Linear(n_hidden, n_hidden) # added additional layer
		self.fc3 = nn.Linear(n_hidden, ndim_out)

		self.activate1 = nn.PReLU() # added activation.
		self.activate2 = nn.PReLU() # added activation.
		self.scale_rel = torch.Tensor([scale_rel]).to(device)

		# self.fd1 = nn.Linear(3,30)
		# self.fd2 = nn.Linear(30,15)

	def forward(self, inpt, x_query, x_context, k = 15):

		n_grid = x_context.shape[0]
		n_query = x_query.shape[0]

		# inpt = torch.relu(self.fd2(torch.relu(self.fd1(inpt))))

		edge_index = knn(x_context, x_query, k = k).flip(0)
		edge_attr = (x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel # /scale_x

		N = n_grid
		M = n_query

		return self.fc3(self.activate2(self.fc2(self.propagate(edge_index, size = (N, M), x = inpt, edge_offsets = edge_attr))))

	def message(self, x_j, edge_offsets):

		return self.activate1(self.fc1(torch.cat((x_j, edge_offsets), dim = 1)))

class GNN_Network_spheroid(nn.Module):

	def __init__(self, n_hidden = 20, device = 'cpu'):
		super(GNN_Network_spheroid, self).__init__()

		self.embed = nn.Sequential(nn.Linear(9, 30), nn.PReLU(), nn.Linear(30, 30), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(6, 30), nn.PReLU(), nn.Linear(30, 30), nn.PReLU())

		self.SpatialAggregation1 = SpatialAggregation(30, n_hidden, n_mask = 30)
		self.SpatialAggregation2 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation3 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation4 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation5 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation6 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation7 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation8 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation9 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation10 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.BipartiteReadOut = SpatialAttention(n_hidden, 3, device = device)
		# self.PredictScale = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 3))
		# self.ten = torch.Tensor([10.0]).to(device)

		# self.fc1 = nn.Linear(15, 1)
    
	def forward(self, x, mask, x_query, A_edges, A_edges_c_1, merged_nodes, batch, batch_query, n_nodes, permute = False):

		if permute == True:

			rand_perm = torch.hstack([torch.Tensor(np.random.permutation(n_nodes)).long().to(device) + j*n_nodes for j in range(batch.max() + 1)])
			A_edges_c = rand_perm[A_edges_c_1].contiguous()

		else:

			A_edges_c = A_edges_c_1

		x = self.embed(x)
		mask = self.embed_mask(mask)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, merged_nodes, batch, n_nodes)

		pred = self.BipartiteReadOut(out, x_query, merged_nodes, batch, batch_query, n_nodes) # linear output layer.

		return pred

class GNN_Network_heterogeneous(nn.Module):

	def __init__(self, n_hidden = 20, device = 'cpu'):
		super(GNN_Network_heterogeneous, self).__init__()

		self.embed = nn.Sequential(nn.Linear(78, 50), nn.PReLU(), nn.Linear(50, 50), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(75, 50), nn.PReLU(), nn.Linear(50, 30), nn.PReLU())

		self.SpatialAggregation1 = SpatialAggregation(50, n_hidden, n_mask = 30)
		self.SpatialAggregation2 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation3 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation4 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation5 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation6 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation7 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation8 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation9 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation10 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.BipartiteReadOut = SpatialAttention(n_hidden, 3)

		# self.fc1 = nn.Linear(15, 1)
	
	def forward(self, x, mask, x_query, A_edges, A_edges_c_1, merged_nodes, batch, batch_query, n_nodes, permute = False):

		if permute == True:

			rand_perm = torch.hstack([torch.Tensor(np.random.permutation(n_nodes)).long().cuda() + j*n_nodes for j in range(batch.max() + 1)])
			A_edges_c = rand_perm[A_edges_c_1].contiguous()

		else:

			A_edges_c = A_edges_c_1

		x = self.embed(x)
		mask = self.embed_mask(mask)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, merged_nodes, batch, n_nodes)
		pred = self.BipartiteReadOut(out, x_query, merged_nodes, batch, batch_query, n_nodes) # linear output layer.

		return pred

	
class GNN_Network_Norm_spheroid(nn.Module):

	def __init__(self, n_hidden = 20, device = 'cpu'):
		super(GNN_Network_Norm_spheroid, self).__init__()

		self.embed = nn.Sequential(nn.Linear(9, 30), nn.PReLU(), nn.Linear(30, 30), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(6, 30), nn.PReLU(), nn.Linear(30, 30), nn.PReLU())

		self.SpatialAggregation1 = SpatialAggregation(30, n_hidden, n_mask = 30)
		self.SpatialAggregation2 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation3 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation4 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation5 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation6 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation7 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation8 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation9 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation10 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		# self.BipartiteReadOut = SpatialAttention(n_hidden, 3)
		self.PredictScale = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 1))
		# self.ten = torch.Tensor([10.0]).to(device)

		# self.fc1 = nn.Linear(15, 1)
	
	def forward(self, x, mask, x_query, A_edges, A_edges_c_1, merged_nodes, batch, batch_query, n_nodes, permute = False):

		if permute == True:

			rand_perm = torch.hstack([torch.Tensor(np.random.permutation(n_nodes)).long().to(device) + j*n_nodes for j in range(batch.max() + 1)])
			A_edges_c = rand_perm[A_edges_c_1].contiguous()

		else:

			A_edges_c = A_edges_c_1

		x = self.embed(x)
		mask = self.embed_mask(mask)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, merged_nodes, batch, n_nodes)

		global_pool = GNN_f.global_mean_pool(out, batch) # Could do max-pool		
		pred = self.PredictScale(global_pool)
		# norm = torch.pow(self.ten, norm_log10)

		# pred = self.BipartiteReadOut(out, x_query, merged_nodes, batch, batch_query, n_nodes) # linear output layer.

		return pred

class GNN_Network_Norm_heterogeneous(nn.Module):

	def __init__(self, n_hidden = 20, device = 'cpu'):
		super(GNN_Network_Norm_heterogeneous, self).__init__()

		self.embed = nn.Sequential(nn.Linear(78, 50), nn.PReLU(), nn.Linear(50, 50), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(75, 50), nn.PReLU(), nn.Linear(50, 30), nn.PReLU())

		self.SpatialAggregation1 = SpatialAggregation(50, n_hidden, n_mask = 30)
		self.SpatialAggregation2 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation3 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation4 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation5 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation6 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation7 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation8 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation9 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation10 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.BipartiteReadOut = SpatialAttention(n_hidden, 3)
		self.PredictNorm = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 1))
		self.ten = torch.Tensor([10.0]).to(device)

		# self.fc1 = nn.Linear(15, 1)

	def forward(self, x, mask, x_query, A_edges, A_edges_c_1, merged_nodes, batch, batch_query, n_nodes, permute = False):

		if permute == True:

			rand_perm = torch.hstack([torch.Tensor(np.random.permutation(n_nodes)).long().cuda() + j*n_nodes for j in range(batch.max() + 1)])
			A_edges_c = rand_perm[A_edges_c_1].contiguous()

		else:

			A_edges_c = A_edges_c_1

		x = self.embed(x)
		mask = self.embed_mask(mask)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, merged_nodes, batch, n_nodes)

		global_pool = GNN_f.global_mean_pool(out, batch) # Could do max-pool		
		pred = self.PredictNorm(global_pool)
		# norm = torch.pow(self.ten, norm_log10)

		# pred = self.BipartiteReadOut(out, x_query, merged_nodes, batch, batch_query, n_nodes) # linear output layer.

		return pred

	

class GNN_Network_Lh_and_Lv_spheroid(nn.Module):

	def __init__(self, n_hidden = 20, device = 'cpu'):
		super(GNN_Network_Lh_and_Lv_spheroid, self).__init__()

		self.embed = nn.Sequential(nn.Linear(9, 30), nn.PReLU(), nn.Linear(30, 30), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(6, 30), nn.PReLU(), nn.Linear(30, 30), nn.PReLU())

		self.SpatialAggregation1 = SpatialAggregation(30, n_hidden, n_mask = 30)
		self.SpatialAggregation2 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation3 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation4 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation5 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation6 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation7 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation8 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation9 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation10 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.BipartiteReadOut = SpatialAttention(n_hidden, 3)
		self.PredictScale = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 2))
		# self.ten = torch.Tensor([10.0]).to(device)

		# self.fc1 = nn.Linear(15, 1)

	def forward(self, x, mask, x_query, A_edges, A_edges_c_1, merged_nodes, batch, batch_query, n_nodes, permute = False):

		if permute == True:

			rand_perm = torch.hstack([torch.Tensor(np.random.permutation(n_nodes)).long().to(device) + j*n_nodes for j in range(batch.max() + 1)])
			A_edges_c = rand_perm[A_edges_c_1].contiguous()

		else:

			A_edges_c = A_edges_c_1

		x = self.embed(x)
		mask = self.embed_mask(mask)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, merged_nodes, batch, n_nodes)

		global_pool = GNN_f.global_mean_pool(out, batch) # Could do max-pool		
		pred = self.PredictScale(global_pool)
		# norm = torch.pow(self.ten, norm_log10)

		# pred = self.BipartiteReadOut(out, x_query, merged_nodes, batch, batch_query, n_nodes) # linear output layer.

		return pred

class GNN_Network_Lh_and_Lv_heterogeneous(nn.Module):

	def __init__(self, n_hidden = 20, device = 'cpu'):
		super(GNN_Network_Lh_and_Lv_heterogeneous, self).__init__()

		self.embed = nn.Sequential(nn.Linear(78, 50), nn.PReLU(), nn.Linear(50, 50), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(75, 50), nn.PReLU(), nn.Linear(50, 30), nn.PReLU())

		self.SpatialAggregation1 = SpatialAggregation(50, n_hidden, n_mask = 30)
		self.SpatialAggregation2 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation3 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation4 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation5 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation6 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation7 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation8 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation9 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.SpatialAggregation10 = SpatialAggregation(n_hidden, n_hidden, n_mask = 30)
		self.BipartiteReadOut = SpatialAttention(n_hidden, 3)
		self.PredictNorm = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 2))
		self.ten = torch.Tensor([10.0]).to(device)

		# self.fc1 = nn.Linear(15, 1)

	def forward(self, x, mask, x_query, A_edges, A_edges_c_1, merged_nodes, batch, batch_query, n_nodes, permute = False):

		if permute == True:

			rand_perm = torch.hstack([torch.Tensor(np.random.permutation(n_nodes)).long().cuda() + j*n_nodes for j in range(batch.max() + 1)])
			A_edges_c = rand_perm[A_edges_c_1].contiguous()

		else:

			A_edges_c = A_edges_c_1

		x = self.embed(x)
		mask = self.embed_mask(mask)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges_c, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, merged_nodes, batch, n_nodes)

		global_pool = GNN_f.global_mean_pool(out, batch) # Could do max-pool		
		pred = self.PredictNorm(global_pool)
		# norm = torch.pow(self.ten, norm_log10)

		# pred = self.BipartiteReadOut(out, x_query, merged_nodes, batch, batch_query, n_nodes) # linear output layer.

		return pred

