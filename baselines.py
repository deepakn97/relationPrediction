import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import RelationGraphAttentionLayer, SpGraphAttentionLayer, ConvKB, ConvE

CUDA = torch.cuda.is_available()  # checking cuda availability


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


class TransE(nn.Module):
  	def __init__(self, dimensions, num_entites, num_relations):
    		super().__init__()
    		self.entity_embeddings = nn.Embedding(num_entites, dimensions)
    		self.relation_embeddings = nn.Embedding(num_relations, dimensions)

    		nn.init.xavier_uniform_(self.entity_embeddings.weight, gain=1.414)
    		nn.init.xavier_uniform_(self.relation_embeddings.weight, gain=1.414)

  	def normalize(self):
    		norms = torch.norm(self.entity_embeddings.weight, p=1, dim=1).data
    		self.entity_embeddings.weight.data = self.entity_embeddings.weight.div(norms.unsqueeze(-1).expand_as(self.entity_embeddings.weight))

    		norms = torch.norm(self.relation_embeddings.weight, p=1, dim=1).data
    		self.relation_embeddings.weight.data = self.relation_embeddings.weight.div(norms.unsqueeze(-1).expand_as(self.relation_embeddings.weight))


  	def forward(self, batch_inputs):
    		self.normalize()
    		# print(torch.norm(self.entity_embeddings.weight, p=1, dim=1))
    		e1 = batch_inputs[:, 0]
    		rel = batch_inputs[:, 1]
    		e2 = batch_inputs[:, 2]

    		e1_emebd = self.entity_embeddings(e1)
    		rel_emebd = self.relation_embeddings(rel)
    		e2_emebd = self.entity_embeddings(e2)

    		outputs = e1_emebd + rel_emebd - e2_emebd

    		outputs = torch.norm(outputs, p=1, dim=1)

    		return outputs

loss_func = nn.MarginRankingLoss(margin=5)

def LOSS_TRANSE(outputs):
  	# print(outputs.size())
  	len_pos_triples = int(outputs.shape[0]/2)
  	
  	pos_norm = outputs[:len_pos_triples]
  	neg_norm = outputs[len_pos_triples:]

  	y = torch.ones(len_pos_triples).cuda()
  	loss = loss_func(pos_norm, neg_norm, y)
  	return loss



class TuckER(torch.nn.Module):
    def __init__(self, dimensions, num_entites, num_relations, **kwargs):
        super(TuckER, self).__init__()

        # self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        # self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(num_entites, dimensions))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(num_relations, dimensions))

        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dimensions, dimensions, dimensions)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(dimensions)
        self.bn1 = torch.nn.BatchNorm1d(dimensions)
        

    # def init(self):
    #     xavier_normal_(self.E.weight.data)
    #     xavier_normal_(self.R.weight.data)

    def forward(self, batch_inputs):

        e1_idx = batch_inputs[:, 0]
        r_idx = batch_inputs[:, 1]

        e1 = self.final_entity_embeddings[e1_idx]
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.final_relation_embeddings[r_idx]
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = F.sigmoid(x)
        return pred