import torch
# import torch.onnx
# import hiddenlayer as hl
# from torchviz import make_dot, make_dot_from_trace
from models import SpKBGATModified, SpKBGATConvOnly, SpKBGCN
from layers import ConvKB, ConvE
# from baselines import TransE, LOSS_TRANSE
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle

# %%
# %%from torchviz import make_dot, make_dot_from_trace


def parse_args():
	args = argparse.ArgumentParser()
	# network arguments
	args.add_argument("-data", "--data",
					  default="./data/WN18RR/", help="data directory")
	args.add_argument("-e", "--epochs", type=int,
					  default=3000, help="Number of epochs")
	args.add_argument("-w", "--weight_decay", type=float,
					  default=5e-6, help="L2 reglarization")
	args.add_argument("-iters", "--iterations", type=int,
					  default=1000, help="Number of iterations in training")
	args.add_argument("-val_iters", "--validation_iters",
					  type=int, default=20, help="validation time iterations")
	args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
					  default=True, help="Use pretrained embeddings")
	args.add_argument("-modified", "--modified", type=bool,
					  default=True, help="modified version")
	args.add_argument("-emb_size", "--embedding_size", type=int,
					  default=50, help="Size of embeddings (if pretrained not used)")
	args.add_argument("-adj_type", "--weighted_adj", type=str, default="weighted", help="To use relation type in adjacency matrix \
						or use unweighted adjacency matrix")
	args.add_argument("-b", "--batch_size", type=int,
					  default=86835, help="Batch size")
	args.add_argument("-neg_s", "--valid_invalid_ratio", type=int, default=2,
					  help="Ratio of valid to invalid triples")
	args.add_argument("-l", "--lr", type=float, default=1e-3)
	args.add_argument("-top_k", "--top_k", type=int, default=1500,
					  help="top_k tail entities per validation triple")

	# arguments for GAT
	args.add_argument("-gat_loss_factor", "--gat_loss_factor",
					  type=float, default=1e-3, help="Loss factor for GAT norm loss")
	args.add_argument("-drop_GAT", "--drop_GAT", type=float,
					  default=0.3, help="Dropout probability for SpGAT layer")
	args.add_argument("-alpha", "--alpha", type=float,
					  default=0.2, help="LeakyRelu alphs for SpGAT layer")
	args.add_argument("-alpha_conv", "--alpha_conv", type=float,
					  default=0.2, help="leaky alpha for convolution layer")
	args.add_argument("-clip", "--gradient_clip_norm", type=float,
					  default=0.25, help="maximum norm value for clipping")
	args.add_argument("-out_dim", "--entity_out_dim", type=float,
					  default=[100, 200], help="Entity output embedding dimensions")
	args.add_argument("-h_gat", "--nheads_GAT", type=int,
					  default=[2, 2], help="Multihead attention SpGAT")
	args.add_argument("-dp", "--doping_factor", type=float,
					  default=1, help="Doping factor for adding new embeddings")

	# arguments for convolutions network
	args.add_argument("-o", "--out_channels", type=int, default=50,
					  help="Number of output channels in conv layer")
	args.add_argument("-drop_conv", "--drop_conv", type=float,
					  default=0.3, help="Dropout probability for convolution layer")

	args.add_argument("-type", "--type", default="without_relations",
					  help="Number of output channels in conv layer")
	args.add_argument("-ep_load_num", "--epoch_num", default=2999,
					  help="Number of output channels in conv layer")

	args = args.parse_args()
	return args


args = parse_args()
# %%


def load_data(args):
	train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
		args.data, is_unweigted=False, directed=True)

	if args.pretrained_emb:
		entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
																 os.path.join(args.data, 'relation2vec.txt'))
		print("Initialised relations and entities from TransE")

	else:
		# keeping embedding size same here, preferably 200 for relation embeddings
		entity_embeddings = np.random.randn(
			len(entity2id), args.embedding_size)
		relation_embeddings = np.random.randn(
			len(relation2id), args.embedding_size)
		print("Initialised relations and entities randomly")

	corpus = Corpus(train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, args.batch_size,
					args.valid_invalid_ratio, unique_entities_train)
	return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


Corpus_, entity_embeddings, relation_embeddings = load_data(args)


# with open('2hop_wn.pickle', 'wb') as handle:
#     pickle.dump(Corpus_.node_neighbors_2hop, handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)

# with open('2hop_nell.pickle', 'wb') as handle:
#     pickle.dump(Corpus_.node_neighbors_2hop, handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)   

# with open('3hop_fb.pickle', 'wb') as handle:
#     pickle.dump(Corpus_.node_neighbors_3hop, handle,
# protocol = pickle.HIGHEST_PROTOCOL)


# print("Opening node_neighbors pickle object")
# with open('2hop_wn.pickle', 'rb') as handle:
#    node_neighbors_2hop = pickle.load(handle)

# with open('2hop_nell.pickle', 'rb') as handle:
# 	node_neighbors_2hop = pickle.load(handle)


# with open('2hop_fb.pickle', 'rb') as handle:
#     node_neighbors_2hop = pickle.load(handle)
# print("Loaded node_neighbors pickle object")
#
# neighbors_2hop=neighbors_3hop=0

# for key in node_neighbors_2hop.keys():
#     neighbors_2hop += len(node_neighbors_2hop[key][2])
# for key in node_neighbors_3hop.keys():
#     neighbors_3hop += len(node_neighbors_3hop[key][3])
#
# print("2-hop paths-> ", neighbors_2hop)
# print("3-hop paths-> ", neighbors_3hop)
# sys.exit()

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
	entity_embeddings.size(), relation_embeddings.size()))
# %%

CUDA = torch.cuda.is_available()


def get_validation_score(model, unique_entities):
	model.eval()
	with torch.no_grad():
		# Corpus_.get_validation_pred(model, unique_entities)
		# Corpus_.get_validation_pred_relation(model, unique_entities)
		Corpus_.gat_eval_GAT(model, unique_entities)


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed, is_nhop=False):
	len_pos_triples = int(train_indices.shape[0]/(int(args.valid_invalid_ratio) + 1))

	pos_triples = train_indices[:len_pos_triples]
	neg_triples = train_indices[len_pos_triples:]

	pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio), 1)

	if not is_nhop:
		source_embeds = entity_embed[pos_triples[:, 0]]
		relation_embeds = relation_embed[pos_triples[:, 1]]
		tail_embeds = entity_embed[pos_triples[:, 2]]
	else:
		source_embeds = entity_embed[pos_triples[:, 0]]
		relation_embeds = relation_embed[pos_triples[:, 1]
										 ] + relation_embed[pos_triples[:, 2]]
		tail_embeds = entity_embed[pos_triples[:, 3]]

	x = source_embeds + relation_embeds - tail_embeds
	pos_norm = torch.norm(x, p=1, dim=1)

	if not is_nhop:
		source_embeds = entity_embed[neg_triples[:, 0]]
		relation_embeds = relation_embed[neg_triples[:, 1]]
		tail_embeds = entity_embed[neg_triples[:, 2]]
	else:
		source_embeds = entity_embed[neg_triples[:, 0]]
		relation_embeds = relation_embed[neg_triples[:, 1]
										 ] + relation_embed[neg_triples[:, 2]]
		tail_embeds = entity_embed[neg_triples[:, 3]]

	x = source_embeds + relation_embeds - tail_embeds
	neg_norm = torch.norm(x, p=1, dim=1)

	y = torch.ones(int(args.valid_invalid_ratio) * len_pos_triples).cuda()

	loss = gat_loss_func(pos_norm, neg_norm, y)
	return loss

def train(args):

	# Creating the end-to-end model here.
	####################################

	print("Defining model")
	if not args.modified:
		print("Model type -> is_modified {}".format(args.modified))
		model = SpKBGAT(entity_embeddings, relation_embeddings, args.entity_out_dim,
						args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
						args.nheads_GAT, args.out_channels)

	else:
		print(
			"Model type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
		model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
									args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
									args.nheads_GAT, args.out_channels, args.doping_factor)
		print("Only Conv model trained")
		model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
		                             args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
		                             args.nheads_GAT, args.out_channels, args.doping_factor)

		# model_conv2 = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
		#                              args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
		#                              args.nheads_GAT, args.out_channels, args.doping_factor)
		# print("GCN model trained")
		# model_gcn = SpKBGCN(entity_embeddings, relation_embeddings, args.entity_out_dim,
		#                     args.drop_GAT, args.alpha)
		# print("GCN_BCE model trained")
		# model_gcnBCE = SpKBGCN_withBCE(entity_embeddings, relation_embeddings, args.entity_out_dim,
		#                                args.drop_GAT, args.alpha)

	# model = ConvE(len(Corpus_.entity2id), len(Corpus_.relation2id), 100, 3, 1, 32, 0.2, 0.2)

	if CUDA:
		model_conv.cuda()
		# model_conv2.cuda()
		model_gat.cuda()
		# model_gcn.cuda()
		# model_gcnBCE.cuda()

	# print("Loading model only from epoch 245")
	# model_conv.load_state_dict(torch.load(
	#      './checkpoints/nell/simple_gat2head2hop/conv/trained_199.pth'))
	# model_conv2.load_state_dict(torch.load(
	#      './checkpoints/wn/reproducing_results/trained_135.pth'))


	# model_gat.load_state_dict(torch.load(
	#   './checkpoints/kinship/simple_gat2head2hop/trained_2999.pth'))
	# model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
	# model_conv.final_relation_embeddings = model_gat.final_relation_embeddings
	# get_validation_score(model_conv, Corpus_.unique_entities_train)
	# print(model_gat.final_entity_embeddings[:10] -
	#       model_conv.final_entity_embeddings[:10])
	# return

	# Optimizer -> Adam, Can also use sparse version of Adam, but have to add L2 norm loss separately
	
	# optimizer = torch.optim.Adam(
	# 	model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	optimizer = torch.optim.Adam(
	    model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	scheduler = torch.optim.lr_scheduler.StepLR(
	    optimizer, step_size=500, gamma=0.5, last_epoch=-1)

	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
	#                                                        patience=2, verbose=True, threshold=0.001, threshold_mode='rel',
	#                                                        cooldown=0, min_lr=1e-6, eps=1e-08)

	# conv_loss_func = torch.nn.BCELoss()
	conv_loss_func = torch.nn.BCEWithLogitsLoss()
	margin_loss = torch.nn.SoftMarginLoss()
	gat_loss_func = nn.MarginRankingLoss(margin=5)

	# epoch_losses = []   # losses of all epochs
	# print("Number of epochs {}".format(args.epochs))

	current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(
	   Corpus_.unique_entities_train, node_neighbors_2hop)
	print("length of current_batch_indices is ", len(current_batch_2hop_indices))

	if CUDA:
	   current_batch_2hop_indices = Variable(
	       torch.LongTensor(current_batch_2hop_indices)).cuda()
	else:
	   current_batch_2hop_indices = Variable(
	       torch.LongTensor(current_batch_2hop_indices))

	epoch_losses = []   # losses of all epochs
	print("Number of epochs {}".format(args.epochs))

	for epoch in range(args.epochs):
		print("\nepoch-> ", epoch)
		random.shuffle(Corpus_.train_triples)
		Corpus_.train_indices = np.array(
			list(Corpus_.train_triples)).astype(np.int32)
		print("Training set shuffled, length is ", Corpus_.train_indices.shape)
		model_gat.train()  # getting in training mode
		start_time = time.time()
		epoch_loss = []
		
		if len(Corpus_.train_indices) % args.batch_size == 0:
			num_iters_per_epoch = len(Corpus_.train_indices) // args.batch_size
		else:
			num_iters_per_epoch = (
				len(Corpus_.train_indices) // args.batch_size) + 1

		for iters in range(num_iters_per_epoch):
			start_time_iter = time.time()
			train_indices, train_values = Corpus_.get_iteration_batch(iters)

			if CUDA:
				train_indices = Variable(
					torch.LongTensor(train_indices)).cuda()
				train_values = Variable(torch.FloatTensor(train_values)).cuda()

			else:
				train_indices = Variable(torch.LongTensor(train_indices))
				train_values = Variable(torch.FloatTensor(train_values))
			# print("len batch ", train_indices.size())
			# output = model_transE(train_indices)

			entity_embed, relation_embed = model_gat(
			   Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

			# preds = model_conv(
			# 	Corpus_, Corpus_.train_adj_matrix, train_indices)
			
			optimizer.zero_grad()
			
			# train_values = torch.cat(
			#     (train_values.unsqueeze(-1), train_values_nhop.unsqueeze(-1)), dim=0)
			# print(type(entity_embed), relation_embed)
			# return

			loss = batch_gat_loss(
			   gat_loss_func, train_indices, entity_embed, relation_embed)

			# loss_nhop = batch_gat_loss(
			#     gat_loss_func, train_indices_nhop, entity_embed, relation_embed, True)
			# loss = loss_nhop + loss_1
			# print(loss_1, loss_nhop)
			# print(preds[:10], preds[-10:])
			# return

			# loss = margin_loss(preds.view(-1), train_values.view(-1))
			
			loss.backward()
			optimizer.step()

			epoch_loss.append(loss.data.item())

			end_time_iter = time.time()
			
			print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
				iters, end_time_iter - start_time_iter, loss.data.item()))

		scheduler.step()
		print("Epoch {} , average loss {} , epoch_time {}".format(
			epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
		epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

		save_model(model_gat, args.data, epoch,
				   "without_relations")

# %%
# train(args)


print("Model type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
				args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
				args.nheads_GAT, args.out_channels, args.doping_factor)

model_gat.load_state_dict(torch.load(
         './checkpoints/wn/{0}/trained_{1}.pth'.format(args.type, args.epoch_num)))

model_gat.cuda()
get_validation_score(model_gat, Corpus_.unique_entities_train)

# model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
#                                      args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
#                                      args.nheads_GAT, args.out_channels, args.doping_factor)

# model_conv.load_state_dict(torch.load(
#          './checkpoints/kinship/simple_gat2head2hop/conv/trained_399.pth'))

# model_conv.cuda()
# get_validation_score(model_conv, Corpus_.unique_entities_train)
