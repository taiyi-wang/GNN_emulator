
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
from torch_geometric.nn.conv import PointTransformerConv
from torch.optim.lr_scheduler import StepLR
from scipy.io import loadmat
from scipy.special import sph_harm
from scipy.spatial import cKDTree
from torch.autograd import Variable
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

from utils import *

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
		global_pool = global_mean_pool(self.activate3(self.fglobal(x)), batch) # Could do max-pool
		global_pool_repeat = global_pool.repeat_interleave(n_nodes, dim = 0)

		return self.activate2(self.fc2(torch.cat((x, mask, self.propagate(A_edges, x = torch.cat((x, mask, global_pool_repeat), dim = 1), pos = pos/self.scale_rel), global_pool_repeat), dim = -1)))

	def message(self, x_j, pos_i, pos_j):

		return self.activate1(self.fc1(torch.cat((x_j, self.activate4(self.fedges(pos_i - pos_j))), dim = -1))) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.

class SpatialAggregationMesh(MessagePassing): # make equivelent version with sum operations. (no need for "complex" concatenation version). Assuming concat version is worse/better?
	def __init__(self, in_channels, out_channels, scale_rel = 1.0, n_dim = 3, n_dim_edges = 10, n_global = 3, n_hidden = 15, n_mask = 6):
		super(SpatialAggregationMesh, self).__init__('mean') # node dim
		## Use two layers of SageConv. Explictly or implicitly?
		self.fedges = nn.Linear(n_dim + 1, n_hidden)
		self.fc1 = nn.Linear(in_channels + n_hidden + n_global + n_mask + n_dim_edges, n_hidden)
		self.fc2 = nn.Linear(n_hidden + in_channels + n_global + n_mask, out_channels) ## NOTE: correcting out_channels, here.
		self.fglobal = nn.Linear(in_channels, n_global)
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()
		self.activate4 = nn.PReLU()
		self.scale_rel = scale_rel

	def forward(self, x, mask, A_edges, edge_feature, pos, batch, n_nodes):

		# Because inputs are batched, the "global pool" needs to be applied per disjoint graph
		# Each nodes "global graph" index is assigned in the variable batch
		global_pool = global_mean_pool(self.activate3(self.fglobal(x)), batch) # Could do max-pool
		global_pool_repeat = global_pool.repeat_interleave(n_nodes, dim = 0)

		return self.activate2(self.fc2(torch.cat((x, mask, self.propagate(A_edges, x = torch.cat((x, mask, global_pool_repeat), dim = 1), pos = pos/self.scale_rel, edge_attr = edge_feature), global_pool_repeat), dim = -1)))

	def message(self, x_j, pos_i, pos_j, edge_attr):

		dist = torch.norm(pos_i - pos_j, dim = 1, keepdim = True)**2

		return self.activate1(self.fc1(torch.cat((x_j, self.activate4(self.fedges(torch.cat((pos_i - pos_j, dist), dim = 1))), edge_attr), dim = -1))) # instead of one global signal, map to several, based on a corsened neighborhood. This allows easier time to predict multiple sources simultaneously.


## Note, adding one additional fcn, to the read-out layer
class SpatialAttention(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim = 3, n_latent = 20, scale_rel = 1.0, n_hidden = 30, n_heads = 5):
		super(SpatialAttention, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
		self.fedges = nn.Sequential(nn.Linear(n_dim + 1, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
		self.f_context = nn.Sequential(nn.Linear(inpt_dim + n_hidden + 1, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent)) # add second layer transformation.
		self.f_values = nn.Sequential(nn.Linear(inpt_dim + n_hidden + 1, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent)) # add second layer transformation.
		# self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.proj1 = nn.Linear(n_latent, n_latent) # can remove this layer possibly.
		self.proj2 = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()

	def forward(self, inpt, x_query, x_context, batch, batch_query, n_nodes, k = 15): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

		## Note, make sure this batched version of knn is working
		## for the disjoint, input graphs.

		edge_index = knn(x_context, x_query, k = k, batch_x = batch, batch_y = batch_query).flip(0).contiguous() # Can add batch
		dist = torch.norm((x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel, dim = 1, keepdim = True)**2
		edge_attr = self.activate3(self.fedges(torch.cat(((x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel, dist), dim = 1))) # /scale_x

		return self.proj2(self.activate2(self.proj1(self.propagate(edge_index, x = inpt, edge_attr = edge_attr, dist = dist, size = (x_context.shape[0], x_query.shape[0])).mean(1)))) # mean over different heads

	def message(self, x_j, index, edge_attr, dist):

		context_embed = self.f_context(torch.cat((x_j, edge_attr, dist), dim = -1)).view(-1, self.n_heads, self.n_latent)
		value_embed = self.f_values(torch.cat((x_j, edge_attr, dist), dim = -1)).view(-1, self.n_heads, self.n_latent)
		alpha = self.activate1((self.param_vector*context_embed).sum(-1)/self.scale)

		alpha = softmax(alpha, index)

		return alpha.unsqueeze(-1)*value_embed

## Note, adding one additional fcn, to the read-out layer
class SpatialAttentionFine(MessagePassing):
	def __init__(self, inpt_dim, out_channels, n_dim = 3, n_latent = 20, scale_rel = 1.0, n_hidden = 30, n_heads = 5):
		super(SpatialAttentionFine, self).__init__(node_dim = 0, aggr = 'add') #  "Max" aggregation.
		# notice node_dim = 0.
		self.param_vector = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, n_heads, n_latent)))
		self.fedges = nn.Sequential(nn.Linear(n_dim + 1, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
		self.f_context = nn.Sequential(nn.Linear(inpt_dim + n_hidden + 1, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent)) # add second layer transformation.
		self.f_values = nn.Sequential(nn.Linear(inpt_dim + n_hidden + 1, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_heads*n_latent)) # add second layer transformation.
		# self.f_direct = nn.Linear(inpt_dim, out_channels) # direct read-out for context coordinates.
		self.proj1 = nn.Linear(n_latent, n_latent) # can remove this layer possibly.
		self.proj2 = nn.Linear(n_latent, out_channels) # can remove this layer possibly.
		self.scale = np.sqrt(n_latent)
		self.n_heads = n_heads
		self.n_latent = n_latent
		self.scale_rel = scale_rel
		self.activate1 = nn.PReLU()
		self.activate2 = nn.PReLU()
		self.activate3 = nn.PReLU()

	def forward(self, inpt, x_query, x_context, batch, batch_query, n_nodes, k = 10): # Note: spatial attention k is a SMALLER fraction than bandwidth on spatial graph. (10 vs. 15).

		## Note, make sure this batched version of knn is working
		## for the disjoint, input graphs.

		edge_index = knn(x_context, x_query, k = k, batch_x = batch, batch_y = batch_query).flip(0).contiguous() # Can add batch
		dist = torch.norm((x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel, dim = 1, keepdim = True)**2
		edge_attr = self.activate3(self.fedges(torch.cat(((x_query[edge_index[1]] - x_context[edge_index[0]])/self.scale_rel, dist), dim = 1))) # /scale_x

		return self.proj2(self.activate2(self.proj1(self.propagate(edge_index, x = inpt, edge_attr = edge_attr, dist = dist, size = (x_context.shape[0], x_query.shape[0])).mean(1)))) # mean over different heads

	def message(self, x_j, index, edge_attr, dist):

		context_embed = self.f_context(torch.cat((x_j, edge_attr, dist), dim = -1)).view(-1, self.n_heads, self.n_latent)
		value_embed = self.f_values(torch.cat((x_j, edge_attr, dist), dim = -1)).view(-1, self.n_heads, self.n_latent)
		alpha = self.activate1((self.param_vector*context_embed).sum(-1)/self.scale)

		alpha = softmax(alpha, index)

		return alpha.unsqueeze(-1)*value_embed

class GNN_Network_Mesh_Enhanced(nn.Module):

	def __init__(self, n_inpt = 6, n_inpt_edges = 1, n_hidden = 20, n_embed = 10, device = 'cpu'):
		super(GNN_Network_Mesh_Enhanced, self).__init__()

		n_hidden_embed = 30
		n_dim_embed = 10
		n_dim_embed_mask = 10
		n_dim_embed_edges = 10

		## Add norm value to input
		self.embed = nn.Sequential(nn.Linear(n_inpt + 1, n_hidden_embed), nn.PReLU(), nn.Linear(n_hidden_embed, n_dim_embed), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(n_inpt + 1, n_hidden_embed), nn.PReLU(), nn.Linear(n_hidden_embed, n_dim_embed_mask), nn.PReLU())
		self.embed_edges = nn.Sequential(nn.Linear(n_inpt_edges, n_hidden_embed), nn.PReLU(), nn.Linear(n_hidden_embed, n_dim_embed_edges), nn.PReLU())

		## Inputs: two distance metrics, and 1 or 0 for node type
		## Can also give absolute positions as an additional input feature (but limited from mask)

		self.SpatialAggregation1 = SpatialAggregationMesh(n_dim_embed, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation2 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation3 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation4 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation5 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation6 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation7 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation8 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation9 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.SpatialAggregation10 = SpatialAggregationMesh(n_hidden, n_hidden, n_mask = n_dim_embed_mask, n_dim_edges = n_dim_embed_edges)
		self.BipartiteReadOut1 = SpatialAttention(n_hidden, n_embed)
		self.BipartiteReadOut2 = SpatialAttentionFine(n_hidden, n_embed)
		self.Pred = nn.Sequential(nn.Linear(2*n_embed, 2*n_embed), nn.PReLU(), nn.Linear(2*n_embed, 2*n_embed), nn.PReLU(), nn.Linear(2*n_embed, 3))

		## Extra parameters
		self.device = device
		self.n_hidden_embed = n_hidden_embed
		self.n_dim_embed = n_dim_embed
		self.n_dim_embed_mask = n_dim_embed_mask
		self.n_dim_embed_edges = n_dim_embed_edges

		# self.permute = permute

		# self.fc1 = nn.Linear(15, 1)

	def forward(self, x, mask, norm_val, x_query, A_edges, A_feature_edges, A_edges_c_1, merged_nodes, batch, batch_query, subset_indices, n_nodes, permute = False):

		if permute == True:

			rand_perm = torch.hstack([torch.Tensor(np.random.permutation(n_nodes)).long().cuda() + j*n_nodes for j in range(batch.max() + 1)])
			A_edges_c = rand_perm[A_edges_c_1].contiguous()

		else:

			A_edges_c = A_edges_c_1

		x = self.embed(torch.cat((x, norm_val), dim = 1))
		mask = self.embed_mask(torch.cat((mask, norm_val), dim = 1))
		edge_feature = self.embed_edges(A_feature_edges)
		edge_feature_null = torch.zeros(A_edges_c.shape[1], self.n_dim_embed_edges).to(self.device)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, edge_feature, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, edge_feature, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges_c, edge_feature_null, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, edge_feature, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, edge_feature, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges_c, edge_feature_null, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, edge_feature, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, edge_feature, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges_c, edge_feature_null, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, edge_feature, merged_nodes, batch, n_nodes)
		pred1 = self.BipartiteReadOut1(out[subset_indices], x_query, merged_nodes[subset_indices], batch[subset_indices], batch_query, n_nodes) # linear output layer.
		pred2 = self.BipartiteReadOut2(out[subset_indices], x_query, merged_nodes[subset_indices], batch[subset_indices], batch_query, n_nodes) # linear output layer.
		pred = self.Pred(torch.cat((pred1, pred2), dim = 1))

		return pred

class GNN_Network_Norm_Mesh_Enhanced(nn.Module):

	def __init__(self, n_inpt = 3, n_hidden = 20, device = 'cuda'):
		super(GNN_Network_Norm_Mesh_Enhanced, self).__init__()

		n_hidden_embed = 30
		n_dim_embed = 10
		n_dim_embed_mask = 10
		n_dim_embed_edges = 10

		self.embed = nn.Sequential(nn.Linear(n_inpt, n_hidden_embed), nn.PReLU(), nn.Linear(n_hidden_embed, n_dim_embed), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(n_inpt, n_hidden_embed), nn.PReLU(), nn.Linear(n_hidden_embed, n_dim_embed_mask), nn.PReLU())

		self.SpatialAggregation1 = SpatialAggregation(n_dim_embed, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation2 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation3 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation4 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation5 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation6 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation7 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation8 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation9 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation10 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.PredictNorm = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 1))
		self.five = torch.Tensor([5.0]).to(device)

		## Extra parameters
		self.device = device
		self.n_hidden_embed = n_hidden_embed
		self.n_dim_embed = n_dim_embed
		self.n_dim_embed_mask = n_dim_embed_mask
		self.n_dim_embed_edges = n_dim_embed_edges

	def forward(self, x, mask, A_edges, merged_nodes, batch, n_nodes):

		x = self.embed(x)
		mask = self.embed_mask(mask)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, merged_nodes, batch, n_nodes)

		global_pool = global_mean_pool(out, batch) # Could do max-pool		
		pred = self.five*self.PredictNorm(global_pool)

		return pred


class GNN_Network_Lh_and_Lv_Mesh_Enhanced(nn.Module):

	def __init__(self, n_inpt = 3, n_hidden = 20, device = 'cuda'):
		super(GNN_Network_Lh_and_Lv_Mesh_Enhanced, self).__init__()

		n_hidden_embed = 30
		n_dim_embed = 10
		n_dim_embed_mask = 10
		n_dim_embed_edges = 10

		self.embed = nn.Sequential(nn.Linear(n_inpt, n_hidden_embed), nn.PReLU(), nn.Linear(n_hidden_embed, n_dim_embed), nn.PReLU())
		self.embed_mask = nn.Sequential(nn.Linear(n_inpt, n_hidden_embed), nn.PReLU(), nn.Linear(n_hidden_embed, n_dim_embed_mask), nn.PReLU())

		self.SpatialAggregation1 = SpatialAggregation(n_dim_embed, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation2 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation3 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation4 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation5 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation6 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation7 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation8 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation9 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.SpatialAggregation10 = SpatialAggregation(n_hidden, n_hidden, n_mask = n_dim_embed_mask)
		self.PredictNorm = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 2))
		self.ten = torch.Tensor([10.0]).to(device)

		# self.fc1 = nn.Linear(15, 1)

	def forward(self, x, mask, A_edges, merged_nodes, batch, n_nodes):

		x = self.embed(x)
		mask = self.embed_mask(mask)

		out_1 = self.SpatialAggregation1(x, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation2(out_1, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation3(out, mask, A_edges, merged_nodes, batch, n_nodes) + out_1
		out_2 = self.SpatialAggregation4(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation5(out_2, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation6(out, mask, A_edges, merged_nodes, batch, n_nodes) + out_2
		out_3 = self.SpatialAggregation7(out, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation8(out_3, mask, A_edges, merged_nodes, batch, n_nodes)
		out = self.SpatialAggregation9(out, mask, A_edges, merged_nodes, batch, n_nodes) + out_3
		out = self.SpatialAggregation10(out, mask, A_edges, merged_nodes, batch, n_nodes)

		global_pool = global_mean_pool(out, batch) # Could do max-pool		
		pred = self.ten*self.PredictNorm(global_pool)

		return pred

class GNN_Merged_Mesh_Enhanced(nn.Module):

	def __init__(self, path_to_model, n_vers_load, n_steps_load, pos_grid_l, A_edges_c, A_edges_c_mesh, norm_vals, use_interquery = False, use_interquery_expanded = False, device = 'cpu'):
		super(GNN_Merged_Mesh_Enhanced, self).__init__()

		assert((use_interquery + use_interquery_expanded) <= 1)

		n_ver_load_displacement, n_ver_load_lh_and_lv, n_ver_load_norm = n_vers_load
		n_step_load_displacement, n_step_load_lh_and_lv, n_step_load_norm = n_steps_load	

		## Choose if using regular GNN displacement network or with inter-query interpolation
		if (use_interquery == False)*(use_interquery_expanded == False):
			m_displacement = GNN_Network_Mesh_Enhanced(device = device).to(device)
			m_displacement.load_state_dict(torch.load(path_to_model + 'TrainedModels/trained_heterogenous_displacement_model_step_%d_ver_%d.h5'%(n_step_load_displacement, n_ver_load_displacement), map_location = device))

		m_displacement.eval()
		self.m_displacement = m_displacement

		## Load pre-trained Lh and Lv prediction model
		m_lh_and_lv = GNN_Network_Lh_and_Lv_Mesh_Enhanced(device = device).to(device)
		m_lh_and_lv.load_state_dict(torch.load(path_to_model + 'TrainedModels/trained_heterogenous_Lh_and_Lv_model_step_%d_ver_%d.h5'%(n_step_load_lh_and_lv, n_ver_load_lh_and_lv), map_location = device))
		m_lh_and_lv.eval()
		self.m_lh_and_lv = m_lh_and_lv

		## Load pre-trained norm prediction model
		m_norm = GNN_Network_Norm_Mesh_Enhanced(device = device).to(device)
		m_norm.load_state_dict(torch.load(path_to_model + 'TrainedModels/trained_heterogenous_norm_model_step_%d_ver_%d.h5'%(n_step_load_norm, n_ver_load_norm), map_location = device))
		m_norm.eval()
		self.m_norm = m_norm

		config = load_config(path_to_model +'config.yaml')
		self.scale_val = config['scale_val']
		self.n_nodes_grid = config['number_of_spatial_nodes']
		self.n_features = config['n_features']
		self.min_val = config['min_val']
		self.max_val = config['max_val']
		self.n_samples = config['n_samples']
		self.k_spc_edges = config['k_spc_edges']
		self.norm_version = config['norm_version']
		self.norm_vals = norm_vals
		self.device = device
		self.params = [self.scale_val, self.n_nodes_grid, self.n_features, self.n_samples, self.min_val, self.max_val, self.k_spc_edges, self.norm_version, self.norm_vals, self.device]

		## Load spherical harmonic parameters
		ls, ms, tta, phi, PHI, TTA, lmax = load_spherical_harmonic_parameters()
		self.ls = ls
		self.ms = ms
		self.tta = tta
		self.phi = phi
		self.PHI = PHI
		self.TTA = TTA
		self.lmax = lmax
		self.shape_vals = [self.ls, self.ms, self.TTA, self.PHI]
		self.n_nodes_mesh = PHI.shape[0]*PHI.shape[1]
		self.n_nodes_pos = [len(pos_grid_l[j]) for j in range(len(pos_grid_l))]

		## Set spatial graphs
		self.pos_grid_l = pos_grid_l
		self.A_edges_c = A_edges_c
		self.A_edges_c_mesh = A_edges_c_mesh

	def prediction(self, inpt, X_query, grid_ind, flipped_edges = 'both'): # True ## Files and grid indices as input

		# st_files = [st[isample[j]] for j in range(n_batch)]

		# batch_index = torch.hstack([torch.ones(self.n_nodes_grid)*j for j in range(n_batch)]).long().to(self.device)
		# batch_index_query = torch.hstack([torch.ones(self.n_samples)*j for j in range(n_batch)]).long().to(self.device)

		grid_ind = int(grid_ind)
		batch_index = torch.zeros(len(self.pos_grid_l[grid_ind])).to(self.device).long()
		batch_index_grid_and_mesh = torch.zeros(len(self.pos_grid_l[grid_ind]) + self.n_nodes_mesh).to(self.device).long()
		batch_index_mesh = torch.zeros(self.n_nodes_mesh).to(self.device).long()
		batch_index_query = torch.zeros(len(X_query)).to(self.device).long()
		X_query_list = [X_query]

		subset_indices = torch.arange(self.n_nodes_grid).to(self.device).long()
		# subset_indices = torch.hstack([torch.arange(n_nodes_grid) + (n_nodes_grid + n_nodes_mesh)*j for j in range(n_batch)]).long().to(device)

		dz_list = [inpt[0]]
		Fs_list = [inpt[1:-2]]
		NormF_list = [inpt[-2]]
		RMax_list = [inpt[-1]]
		
		## First apply Lh and Lv prediction model to get Lh and Lv (old version)
		# pos_slice, signal_slice, query_slice, edges_slice, edges_c_slice, trgt_slice = assemble_batch_data_Lh_and_Lv(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, self.pos_grid_l, self.A_edges_c, self.params)
		# inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_lh_and_lv_batch = batch_inputs(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, self.n_nodes_grid)
		# pred_lh_and_lv = self.m_lh_and_lv(inpt_batch.contiguous(), mask_batch.contiguous(), query_batch, edges_batch, edges_batch_c, pos_batch, batch_index, batch_index_query, self.n_nodes_grid)	
		# pred_lh_and_lv = pred_lh_and_lv.detach().cpu().numpy()

		## First apply norm prediction model (note, both enhanced mesh Lh and Lv, and enhanced mesh norm model require same inputs)
		pos_slice, signal_slice, edges_slice = assemble_batch_data_norm_mesh_enhanced(dz_list, Fs_list, NormF_list, RMax_list, self.shape_vals, self.params)
		inpt_batch = torch.vstack(signal_slice) # .to(device)
		mask_batch = torch.vstack(signal_slice) # Only select non-position points for mask
		pos_batch = torch.vstack(pos_slice)
		edges_batch = torch.cat([edges_slice[j] + j*self.n_nodes_mesh for j in range(len(edges_slice))], dim = 1) # .to(device)

		pred_norm = self.m_norm(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index_mesh, self.n_nodes_mesh)
		pred_lh_and_lv = self.m_lh_and_lv(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index_mesh, self.n_nodes_mesh).cpu().detach().numpy()

		## Now make displacement prediction (with predicted Lh and Lv spatial graph size scaling values)

		if flipped_edges == False:
			pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice = assemble_batch_data_displacement_mesh_enhanced(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, pred_lh_and_lv, self.pos_grid_l, self.A_edges_c, self.A_edges_c_mesh, self.shape_vals, self.params)
		elif flipped_edges == True:
			pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice = assemble_batch_data_displacement_mesh_enhanced_flipped_edges(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, pred_lh_and_lv, self.pos_grid_l, self.A_edges_c, self.A_edges_c_mesh, self.shape_vals, self.params)			
		elif flipped_edges == 'both':
			pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice = assemble_batch_data_displacement_mesh_enhanced_both_edges(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, pred_lh_and_lv, self.pos_grid_l, self.A_edges_c, self.A_edges_c_mesh, self.shape_vals, self.params)						
		else:
			error('No flag for flipped edges')

		## Pass in norm as input
		pred_norm_repeat = pred_norm.repeat_interleave(self.n_nodes_pos[grid_ind] + self.n_nodes_mesh, dim = 0)

		trgt_slice = []
		inpt_batch, mask_batch, pos_batch, query_batch_x, edges_batch, edges_feature_batch, edges_batch_c, trgt_batch = batch_inputs_mesh(signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, pos_slice, trgt_slice, self.n_nodes_grid + self.n_nodes_mesh)
		pred = self.m_displacement(inpt_batch.contiguous(), mask_batch.contiguous(), pred_norm_repeat, query_batch_x, edges_batch, edges_feature_batch, edges_batch_c, pos_batch, batch_index_grid_and_mesh, batch_index_query, subset_indices, self.n_nodes_grid + self.n_nodes_mesh)

		return (pred.cpu().detach()*torch.pow(torch.Tensor([10.0]), pred_norm.reshape(-1,1).cpu().detach()).repeat_interleave(len(X_query), dim = 0)).numpy().reshape(-1, len(X_query), 3)
	

	def prediction_average_grids(self, inpt, X_query, flipped_edges = 'both'): # True ## Files and grid indices as input

		n_grids = len(self.pos_grid_l)
		batch_index = torch.hstack([i*torch.ones(len(self.pos_grid_l[i])) for i in range(n_grids)]).to(self.device).long()
		batch_index_grid_and_mesh = torch.hstack([i*torch.ones(len(self.pos_grid_l[i]) + self.n_nodes_mesh) for i in range(n_grids)]).to(self.device).long()
		batch_index_mesh = torch.hstack([i*torch.ones(self.n_nodes_mesh) for i in range(n_grids)]).to(self.device).long()
		batch_index_query = torch.hstack([i*torch.ones(len(X_query)) for i in range(n_grids)]).to(self.device).long()
		X_query_list = [X_query for i in range(n_grids)]
		grid_ind = np.arange(n_grids)

		dz_list = [inpt[0] for i in range(n_grids)]
		Fs_list = [inpt[1:-2] for i in range(n_grids)]
		NormF_list = [inpt[-2] for i in range(n_grids)]
		RMax_list = [inpt[-1] for i in range(n_grids)]
		
		subset_indices = torch.hstack([torch.arange(self.n_nodes_grid) + (self.n_nodes_grid + self.n_nodes_mesh)*j for j in range(n_grids)]).long().to(self.device)

		## First apply norm prediction model (note, both enhanced mesh Lh and Lv, and enhanced mesh norm model require same inputs)
		pos_slice, signal_slice, edges_slice = assemble_batch_data_norm_mesh_enhanced(dz_list, Fs_list, NormF_list, RMax_list, self.shape_vals, self.params)
		inpt_batch = torch.vstack(signal_slice) # .to(device)
		mask_batch = torch.vstack(signal_slice) # Only select non-position points for mask
		pos_batch = torch.vstack(pos_slice)
		edges_batch = torch.cat([edges_slice[j] + j*self.n_nodes_mesh for j in range(len(edges_slice))], dim = 1) # .to(device)

		pred_norm = self.m_norm(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index_mesh, self.n_nodes_mesh)
		pred_lh_and_lv = self.m_lh_and_lv(inpt_batch.contiguous(), mask_batch.contiguous(), edges_batch, pos_batch, batch_index_mesh, self.n_nodes_mesh).cpu().detach().numpy()

		## Now make displacement prediction (with predicted Lh and Lv spatial graph size scaling values)
		
		trgt_slice = []
		if flipped_edges == False:
			pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice = assemble_batch_data_displacement_mesh_enhanced(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, pred_lh_and_lv, self.pos_grid_l, self.A_edges_c, self.A_edges_c_mesh, self.shape_vals, self.params)
		elif flipped_edges == True:
			pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice = assemble_batch_data_displacement_mesh_enhanced_flipped_edges(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, pred_lh_and_lv, self.pos_grid_l, self.A_edges_c, self.A_edges_c_mesh, self.shape_vals, self.params)
		elif flipped_edges == 'both':
			pos_slice, signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice = assemble_batch_data_displacement_mesh_enhanced_both_edges(dz_list, Fs_list, NormF_list, RMax_list, X_query_list, grid_ind, pred_lh_and_lv, self.pos_grid_l, self.A_edges_c, self.A_edges_c_mesh, self.shape_vals, self.params)						
		else:
			error('No flag for flipped edges')

		## Pass in norm as input
		pred_norm_repeat = pred_norm.repeat_interleave(self.n_nodes_pos[0] + self.n_nodes_mesh, dim = 0)


		inpt_batch, mask_batch, pos_batch, query_batch_x, edges_batch, edges_feature_batch, edges_batch_c, trgt_batch = batch_inputs_mesh(signal_slice, query_slice, edges_slice, edges_feature_slice, edges_c_slice, pos_slice, trgt_slice, self.n_nodes_grid + self.n_nodes_mesh)
		pred = self.m_displacement(inpt_batch.contiguous(), mask_batch.contiguous(), pred_norm_repeat, query_batch_x, edges_batch, edges_feature_batch, edges_batch_c, pos_batch, batch_index_grid_and_mesh, batch_index_query, subset_indices, self.n_nodes_grid + self.n_nodes_mesh)

		return (pred.cpu().detach()*torch.pow(torch.Tensor([10.0]), pred_norm.reshape(-1,1).cpu().detach()).repeat_interleave(len(X_query), dim = 0)).numpy().reshape(-1, len(X_query), 3)


	def predict(self, fs, dx, dy, dz, Rmax, dp2mu, X, Y, Z, avgFlag):
		# prediction function for user interface

		# Compute normF-----------------------------------------------------------
		tta = np.linspace(0, np.pi, 50)
		phi = np.linspace(0, 2*np.pi, 50)
		PHI, TTA = np.meshgrid(phi, tta)

		lmax = 5 # maximum degree used to train emulator
		ls, ms = generateDegreeOrder(lmax)

    	# only use non-negative orders for constructing real shapes
		ls = ls[np.where(ms >= 0)]
		ms = ms[np.where(ms >= 0)]

		fs_0 = fs[0:np.size(ls)] + 1j*fs[np.size(ls):]# re-create complex fs vector for plotting
		_, _, _, normF = cmp_SH_shape_with_norm(ls, ms, fs_0, Rmax, dz, TTA, PHI)

		inpt = np.concatenate((np.array([dz]), fs, np.array([normF]), np.array([Rmax])), axis = 0)

		# Correct for horizontal location of chamber------------------------------------------
		X = X - dx; Y = Y - dy; 
		queries = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)), axis = 1) # receiver locations

		# Compute displacement---------------------------------------------------
		if avgFlag == 0:
			grid_ind = 0
			pred = self.prediction(inpt, queries, grid_ind)*dp2mu/1e-3
		else:
			pred = self.prediction_average_grids(inpt, queries).mean(0)*dp2mu/1e-3

		return pred
