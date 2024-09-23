#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
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
import copy


from torch_scatter import scatter


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

def batch_inputs(signal_slice, query_slice, edges_slice, edges_c_slice, pos_slice, trgt_slice, node_ind_max, device):

	inpt_batch = torch.vstack(signal_slice).to(device)
	mask_batch = inpt_batch[:,3::] # Only select non-position points for mask
	pos_batch = inpt_batch[:,0:3]
	query_batch = torch.Tensor(np.vstack(query_slice)).to(device)
	edges_batch = torch.cat([edges_slice[j] + j*node_ind_max for j in range(len(edges_slice))], dim = 1).to(device)
	edges_batch_c = torch.cat([edges_c_slice[j] + j*node_ind_max for j in range(len(edges_c_slice))], dim = 1).to(device)
	trgt_batch = torch.Tensor(np.vstack(trgt_slice)).to(device)

	return inpt_batch, mask_batch, pos_batch, query_batch, edges_batch, edges_batch_c, trgt_batch

def make_spatial_graph(pos, k_pos = 15, device = 'cpu'):

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
	A_edges = remove_self_loops(knn(pos.cpu(), pos.cpu(), k = k_pos + 1).flip(0).contiguous())[0].to(device)
	# edges_offset = pos[A_edges[1]] - pos[A_edges[0]]

	return A_edges # , edges_offset

def make_graph_from_mesh(mesh):
	## Input is T

	## Check if these edges are correctly defined. E.g.,
	## are incoming and outgoing edges correctly defined?

	data = Data()
	data.face = mesh
	trns = FaceToEdge()
	edges = trns(data).edge_index

	return edges

def apply_models_make_displacement_prediction_spheroid(inpt, queries, m, m1, m2, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu'):

	## Based on inpt, create the (scaled) input and mask vectors, batch and batch query vectors,
	## Then apply predictions.

	## Lh, Lv, dz_val, queries all given in absolute unit scale, and use scale_val to normalize

	n_nodes_grid = pos_grid.shape[0]
	batch_index = torch.zeros(n_nodes_grid).long().to(device) # *j for j in range(n_batch)]).long().to(device)
	batch_index_query = torch.zeros(queries.shape[0]).long().to(device) # *j for j in range(n_batch)]).long().to(device)

	Ra_val, Rb_val, ra2d_val, dx_val, dy_val, dz_val, thetax_val, thetaz_val, dp2mu = inpt

	# Lh_val, Lv_val = Lhv_scale

	# shift query points so that dx and dy are accounted for
	queries_shift = copy.deepcopy(queries)
	queries_shift[:, 0] = queries_shift[:, 0] - dx_val
	queries_shift[:, 1] = queries_shift[:, 1] - dy_val
 
	signal_val = torch.Tensor(np.array([Ra_val, Rb_val, ra2d_val, dz_val/scale_val, thetax_val, thetaz_val]).reshape(1,-1)/norm_vals).to(device)

	queries_inpt = torch.Tensor(queries_shift/scale_val).to(device)

	signal_inpt_unscaled = torch.cat((pos_grid, signal_val*torch.ones(n_nodes_grid,6).to(device)), dim = 1)

	mask_inpt = signal_inpt_unscaled[:,3::]

	# Predict Lh and Lv
	pred_lhv = m2(signal_inpt_unscaled, mask_inpt, queries_inpt, A_edges, A_edges_c, pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	pos = torch.Tensor(np.array([pred_lhv[0,0]/2.0, pred_lhv[0,0]/2.0, pred_lhv[0,1]]).reshape(1,-1)).to(device) # *pos_grid

	# pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), query_batch, edges_batch, edges_c_batch, pos_batch, batch_index, batch_index_query, n_nodes_grid)

	signal_inpt = torch.cat((pos*pos_grid, signal_val*torch.ones(n_nodes_grid,6).to(device)), dim = 1)

	A_edges_scaled = make_spatial_graph(pos*pos_grid)

	# Predict displacement
	pred = m(signal_inpt, mask_inpt, queries_inpt, A_edges_scaled, A_edges_c, pos*pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	# Predict norm
	pred_norm = m1(signal_inpt, mask_inpt, queries_inpt, A_edges_scaled, A_edges_c, pos*pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	return pred*np.power(10.0, pred_norm)*dp2mu/1e-6 # .cpu().detach().numpy()

def apply_models_make_displacement_prediction_heterogeneous(inpt, normF, queries, m, m1, m2, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu'):

	## Based on inpt, create the (scaled) input and mask vectors, batch and batch query vectors,
	## Then apply predictions.

	## Lh, Lv, dz_val, queries all given in absolute unit scale, and use scale_val to normalize

	n_nodes_grid = pos_grid.shape[0]
	batch_index = torch.zeros(n_nodes_grid).long().to(device) # *j for j in range(n_batch)]).long().to(device)
	batch_index_query = torch.zeros(queries.shape[0]).long().to(device) # *j for j in range(n_batch)]).long().to(device)

	# Lh_val, Lv_val = Lhv_scale
 
	one_vec = np.ones(75).reshape(1,-1)
	one_vec[0,0] = scale_val ## Have to divide the first entry, dz, by scale_val

	# signal_val = torch.Tensor(np.array([Ra_val, Rb_val, ra2d_val, dz_val/scale_val, thetax_val, thetaz_val]).reshape(1,-1)/norm_vals).to(device)

	queries_inpt = torch.Tensor(queries/scale_val).to(device)

	dp2mu = inpt[-2]
	inpt0 = np.concatenate(([inpt[1], ], inpt[2:74], [[normF], ], [inpt[0],]), axis = 0)

	signal_inpt_unscaled = torch.cat((pos_grid, torch.Tensor((inpt0.reshape(1,-1)/one_vec)/norm_vals).to(device)*torch.ones(n_nodes_grid,75).to(device)), dim = 1)

	mask_inpt = signal_inpt_unscaled[:,3::]

	# Predict Lh and Lv
	pred_lhv = m2(signal_inpt_unscaled, mask_inpt, queries_inpt, A_edges, A_edges_c, pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	pos = torch.Tensor(np.array([pred_lhv[0,0]/2.0, pred_lhv[0,0]/2.0, pred_lhv[0,1]]).reshape(1,-1)).to(device) # *pos_grid

	# pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), query_batch, edges_batch, edges_c_batch, pos_batch, batch_index, batch_index_query, n_nodes_grid)

	# signal_inpt = torch.cat((pos*pos_grid, signal_val*torch.ones(n_nodes_grid,6).to(device)), dim = 1)

	signal_inpt = torch.cat((pos*pos_grid, torch.Tensor((inpt0.reshape(1,-1)/one_vec)/norm_vals).to(device)*torch.ones(n_nodes_grid,75).to(device)), dim = 1)

	A_edges_scaled = make_spatial_graph(pos*pos_grid)

	# Predict displacement
	pred = m(signal_inpt, mask_inpt, queries_inpt, A_edges_scaled, A_edges_c, pos*pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	# Predict norm
	pred_norm = m1(signal_inpt, mask_inpt, queries_inpt, A_edges_scaled, A_edges_c, pos*pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	return pred*np.power(10.0, pred_norm)*dp2mu/1e-3 # .cpu().detach().numpy()

## Need to implement
def apply_models_make_displacement_batched_prediction(inpt, queries, m, m1, m2, pos_grid, A_edges, A_edges_c, norm_vals, scale_val, device = 'cpu'):

    ## Based on inpt, create the (scaled) input and mask vectors, batch and batch query vectors,
	## Then apply predictions.

	## Lh, Lv, dz_val, queries all given in absolute unit scale, and use scale_val to normalize

	n_nodes_grid = pos_grid.shape[0]
	batch_index = torch.zeros(n_nodes_grid).long().to(device) # *j for j in range(n_batch)]).long().to(device)
	batch_index_query = torch.zeros(queries.shape[0]).long().to(device) # *j for j in range(n_batch)]).long().to(device)

	Ra_val, Rb_val, ra2d_val, dz_val, thetax_val, thetaz_val = inpt

	# Lh_val, Lv_val = Lhv_scale
 
	signal_val = torch.Tensor(np.array([Ra_val, Rb_val, ra2d_val, dz_val/scale_val, thetax_val, thetaz_val]).reshape(1,-1)/norm_vals).to(device)

	queries_inpt = torch.Tensor(queries/scale_val).to(device)

	signal_inpt_unscaled = torch.cat((pos_grid, signal_val*torch.ones(n_nodes_grid,6).to(device)), dim = 1)

	mask_inpt = signal_inpt_unscaled[:,3::]

	# Predict Lh and Lv
	pred_lhv = m2(signal_inpt_unscaled, mask_inpt, queries_inpt, A_edges, A_edges_c, pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	pos = torch.Tensor(np.array([pred_lhv[0,0]/2.0, pred_lhv[0,0]/2.0, pred_lhv[0,1]]).reshape(1,-1)).to(device) # *pos_grid

	# pred = m(inpt_batch.contiguous(), mask_batch.contiguous(), query_batch, edges_batch, edges_c_batch, pos_batch, batch_index, batch_index_query, n_nodes_grid)

	signal_inpt = torch.cat((pos*pos_grid, signal_val*torch.ones(n_nodes_grid,6).to(device)), dim = 1)

	A_edges_scaled = make_spatial_graph(pos*pos_grid)

	# Predict displacement
	pred = m(signal_inpt, mask_inpt, queries_inpt, A_edges_scaled, A_edges_c, pos*pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	# Predict norm
	pred_norm = m1(signal_inpt, mask_inpt, queries_inpt, A_edges_scaled, A_edges_c, pos*pos_grid, batch_index, batch_index_query, n_nodes_grid).cpu().detach().numpy()

	return pred*np.power(10.0, pred_norm) # .cpu().detach().numpy()
