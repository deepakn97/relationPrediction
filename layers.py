import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()

# Same Model as in ConvKB paper


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        # self.non_linearity = nn.LeakyReLU(alpha_leaky)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)
        # self.fc_layer_2 = nn.Linear(1000, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        # nn.init.xavier_uniform_(self.fc_layer_2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        # print( "fc inputs are ", input_fc[input_fc != 0.0], input_fc.size())
        # print(torch.sum(input_fc != 0.0), torch.sum(input_fc == 0))
        # print(input_fc.size())
        output = self.fc_layer(input_fc)
        # print("fc weights are -> ", torch.sum(self.fc_layer.weight > 1e-5))
        # print("fc outputs are -> ", output, output.size())
        # output = self.dropout(self.non_linearity(output))
        # output = self.fc_layer_2(output)
        # output = F.sigmoid(output)
        return output


class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()
        # self.emb_e = torch.nn.Embedding(num_entities, 200, padding_idx=0)
        # self.emb_rel = torch.nn.Embedding(num_relations, 200, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.relu = nn.ReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))

        fc_dim = 32 * (10 - 2) * (20 - 2)
        self.fc = torch.nn.Linear(10368, 100)
        # self.fc2 = torch.nn.Linear(2 * input_dim, 1)

        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1.414)

    def forward(self, conv_input, entity_embeddings):
        batch_size = conv_input.size()[0]
        e1_embedded = conv_input[:, 0, :].view(-1, 1, 5, 20)
        rel_embedded = conv_input[:, 1, :].view(-1, 1, 5, 20)
        e2_embedded = conv_input[:, 2, :]

        # e1 = conv_input[:, 0]
        # rel = conv_input[:, 1]

        # e1_embedded = self.emb_e(e1).view(-1, 1, 10, 20)
        # rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)

        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        # stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = torch.mm(x, entity_embeddings.transpose(1, 0))
        x += self.b.expand_as(x)
        # output = torch.sum(x * e2_embedded, dim=1)
        # output = F.sigmoid(x)
        return x


class RelationGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, entity_dim, dropout, alpha, concat=True):
        super(RelationGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.entity_dim = entity_dim
        self.alpha = alpha
        self.concat = concat

        self.dropout_layer = nn.Dropout(self.dropout)
        self.softmax_layer = nn.Softmax(dim=1)

        # attention mechanism
        self.a = nn.Parameter(torch.zeros(
            size=(self.out_features, 2 * self.in_features + self.entity_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # calculating attention
        self.a_2 = nn.Parameter(torch.zeros(size=(1, self.out_features)))
        nn.init.xavier_uniform_(self.a_2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_relations, input_entity_embeds, relation_adj):
        # h = torch.mm(input, self.W)

        # N is number of Relations
        N = input_relations.shape[0]
        # concatenation of source, relation, target

        a_input = torch.cat([input_relations.repeat(1, N).view(N * N, -1), input_relations.repeat(N, 1),
                             input_entity_embeds.view(N * N, -1), ], dim=1).t()

        # hidden_layer of MLP to learn embeddings
        e = self.a.mm(a_input)

        # learning alpha
        e_2 = self.leakyrelu(self.a_2.mm(e)).view(N, N)

        zero_vec = -9e15 * torch.ones_like(e_2)
        attention = torch.where(relation_adj > 0, e_2, zero_vec)
        attention = self.softmax_layer(attention)

        # applying attention to hidden layer embeddings
        attention = attention.unsqueeze(-1)
        h = a_input.t().view(N, N, -1)
        attention_ = attention.expand(
            attention.shape[0], attention.shape[1], self.out_features)
        e = e.t().view(N, N, -1)

        h_prime = attention_ * e
        h_prime = h_prime.sum(dim=1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # print((grad_output > 0).sum())
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
            # print((grad_values > 0).sum())
            # print("e_rowsum gradients -> ", grad_values)
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        # self.W = nn.Parameter(torch.zeros(
        #     size=(in_features, out_features)))
        # nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        # print("Input shape-> ", input.shape)
        # print("Edge embed-> ", edge_embed.shape)
        # print("Edge shape-> ", edge.shape)
        # print(edge_list_nhop.shape)
        N = input.size()[0]

        # new_embeds = input.mm(self.W)

        # Self-attention on the nodes - Shared attention mechanism
        # edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        # edge_embed = torch.cat(
        #     (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        # print("edge_h-> ", edge_h)
        edge_m = self.a.mm(edge_h)
        # print("edge_m-> ", edge_m)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        # print("powers-> ", powers)
        edge_e = torch.exp(powers).unsqueeze(1)
        # print("a-> ", self.a)
        # print("a_2-> ", self.a_2)
        # print("edge_e-> ", edge_e)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        # e_rowsum = self.special_spmm(edge, edge_e, torch.Size(
        #     [N, N]), torch.ones(size=(N, 1)).cuda())
        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        # + 1e-6*Variable(torch.ones(e_rowsum.size())).cuda()
        e_rowsum = e_rowsum
        # print("e_rowsum-> ", e_rowsum)
        # e_rowsum = torch.ones(self.num_nodes, 1)
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        # print("h_prime before division by e_rowsum -> ", h_prime)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        # print("e_rowsum vals are -> ", e_rowsum)
        h_prime = h_prime.div(e_rowsum)
        # print("h_prime before division by e_rowsum -> ", h_prime)
        # h_prime: N x out

        # h_prime = h_prime + new_embeds

        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'






class SpGraphAttentionLayerNoRelation(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, doping_factor, dropout, alpha, concat=True):
        super(SpGraphAttentionLayerNoRelation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim
        self.doping_factor = doping_factor

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        # self.W = nn.Parameter(torch.zeros(
        #     size=(in_features, out_features)))
        # nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        # print("Input shape-> ", input.shape)
        # print("Edge embed-> ", edge_embed.shape)
        # print("Edge shape-> ", edge.shape)
        # print(edge_list_nhop.shape)
        N = input.size()[0]

        # new_embeds = input.mm(self.W)

        # Self-attention on the nodes - Shared attention mechanism
        # edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        # edge_embed = torch.cat(
        #     (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :]), dim=1).t()
        # edge_h: (2*in_dim) x E

        # print("edge_h-> ", edge_h.size())
        edge_m = self.a.mm(edge_h)
        # print("edge_m-> ", edge_m)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        # print("powers-> ", powers)
        edge_e = torch.exp(powers).unsqueeze(1)
        # print("a-> ", self.a)
        # print("a_2-> ", self.a_2)
        # print("edge_e-> ", edge_e)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        # e_rowsum = self.special_spmm(edge, edge_e, torch.Size(
        #     [N, N]), torch.ones(size=(N, 1)).cuda())
        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        # + 1e-6*Variable(torch.ones(e_rowsum.size())).cuda()
        e_rowsum = e_rowsum
        # print("e_rowsum-> ", e_rowsum)
        # e_rowsum = torch.ones(self.num_nodes, 1)
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        # print("h_prime before division by e_rowsum -> ", h_prime)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        # print("e_rowsum vals are -> ", e_rowsum)
        h_prime = h_prime.div(e_rowsum)
        # print("h_prime before division by e_rowsum -> ", h_prime)
        # h_prime: N x out

        # h_prime = h_prime + new_embeds

        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'







class SpGCNlayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, dropout, alpha):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha

        self.a = nn.Parameter(torch.zeros(
            size=(in_features, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        N = input.size()[0]
        # input: N x D

        entity_m = input.mm(self.a)
        # entity_m: N x F

        h_prime = torch.matmul(adj, entity_m)
        h_prime = self.leakyrelu(h_prime)
        h_prime = self.dropout(h_prime)

        assert not torch.isnan(h_prime).any()
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# class SpGraphAttentionLayerOriginal(nn.Module):
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """

#     def __init__(self, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
#         super(SpGraphAttentionLayerOriginal, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.nrela_dim = nrela_dim
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_normal_(self.W.data, gain=1.414)

#         self.a = nn.Parameter(torch.zeros(
#             size=(1, 2 * out_features + nrela_dim)))
#         nn.init.xavier_normal_(self.a.data, gain=1.414)

#         self.dropout = nn.Dropout(dropout)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.special_spmm = SpecialSpmm()

#     def forward(self, input, edge, edge_embed):
#         N = input.size()[0]

#         h = torch.mm(input, self.W)
#         # h: N x out
#         assert not torch.isnan(h).any()

#         # Self-attention on the nodes - Shared attention mechanism
#         # print(h.shape)
#         # print(edge_embed.shape)
#         # print(edge[0, :].shape)
#         edge_h = torch.cat(
#             (h[edge[0, :], :], edge_embed[:, :], h[edge[1, :], :]), dim=1).t()
#         # edge: 2*D x E

#         edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))
#         assert not torch.isnan(edge_e).any()
#         # edge_e: E

#         e_rowsum = self.special_spmm(edge, edge_e, torch.Size(
#             [N, N]), torch.ones(size=(N, 1)).cuda())
#         # e_rowsum: N x 1

#         edge_e = self.dropout(edge_e)
#         # edge_e: E

#         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
#         assert not torch.isnan(h_prime).any()
#         # h_prime: N x out

#         h_prime = h_prime.div(e_rowsum)
#         # h_prime: N x out
#         assert not torch.isnan(h_prime).any()

#         if self.concat:
#             # if this layer is not last layer,
#             return F.elu(h_prime)
#         else:
#             # if this layer is last layer,
#             return h_prime
