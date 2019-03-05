import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import RelationGraphAttentionLayer, SpGraphAttentionLayer, ConvKB, SpGraphAttentionLayerNoRelation

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

        # self.out_att = SpGraphAttentionLayerNoRelation(num_nodes, nhid * nheads,
        #                                      nheads * nhid, nheads * nhid,
        #                                      dropout=dropout,
        #                                      alpha=alpha,
        #                                      concat=False
        #                                      )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed, 
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        edge_embed_nhop = relation_embed[
            edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]

        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        edge_embed_nhop = out_relation_1[
            edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        x = F.elu(self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop))
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):

        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)
        

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]
        # print(train_indices_nhop.shape)
        edge_list_nhop = torch.cat(
            (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        self.relation_embeddings.data = F.normalize(
            self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings, 
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        # print(entities_upgraded.size())
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)
        # out_relation_1 = F.normalize(out_relation_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

        # self.entity_embeddings = nn.Embedding(self.num_nodes, 100)
        # self.relation_embed = nn.Embedding(238, 100)

        # self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
        #                           self.drop_GAT, self.alpha, self.nheads_GAT_1, self.doping_factor)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv





'''class RelationGAT(nn.Module):
    def __init__(self, nfeat, nhid, entity_dim, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(RelationGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [RelationGraphAttentionLayer(
            nfeat, nhid, entity_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.W = nn.Parameter(torch.zeros(size=(entity_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = RelationGraphAttentionLayer(
            nhid * nheads, nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, entity_embeds, adj):
        x = self.dropout_layer(x)
        x = torch.cat([att(x, entity_embeds, adj)
                       for att in self.attentions], dim=1)

        entity_embeds = entity_embeds.view(adj.shape[0]**2, -1)
        final_entity_embed = entity_embeds.mm(self.W)
        entity_embeds = final_entity_embed.view(
            adj.shape[0], adj.shape[0], -1)

        x = self.dropout_layer(x)
        x = F.elu(self.out_att(x, entity_embeds, adj))
        return x'''


class SpKBGCN(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, drop_GCN, alpha):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.num_relation = initial_relation_emb.shape[0]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.entity_out_dim_2 = entity_out_dim[1]

        self.drop_GCN = drop_GCN
        self.alpha = alpha      # For leaky relu
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_2))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_2))

        self.embeddings = nn.Embedding.from_pretrained(
            torch.cat((initial_entity_emb, initial_relation_emb), dim=0), freeze=False)

        self.sparse_gcn_1 = SpGCNlayer(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1,
                                       self.drop_GCN, self.alpha)
        self.sparse_gcn_2 = SpGCNlayer(self.num_nodes, self.entity_out_dim_1, self.entity_out_dim_2,
                                       self.drop_GCN, self.alpha)

    def forward(self, adj):
        # getting edge list
        out_1 = self.sparse_gcn_1(self.embeddings.weight, adj)
        out_2 = self.sparse_gcn_2(out_1, adj)
        out_2 = F.normalize(out_2, p=2, dim=1)

        self.final_entity_embeddings.data = out_2[:self.num_nodes]
        self.final_relation_embeddings.data = out_2[self.num_nodes:]

        return out_2[:self.num_nodes], out_2[self.num_nodes:]
