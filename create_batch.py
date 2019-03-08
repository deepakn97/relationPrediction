import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random
# from sklearn.preprocessing import LabelBinarizer


class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, entity2id,
                 relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train, get_2hop=False):
        self.train_triples = train_data[0]
        # self.train_adj = train_data[1]    # tuple of (rows, cols, data) format of sparse tensor
        # Converting to sparse tensor
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)
        # self.train_adj_matrix = torch.sparse.LongTensor(adj_indices, adj_values,
        #                                                 torch.Size([len(entity2id), len(entity2id)]))

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        # self.lb = LabelBinarizer()
        # self.lb.fit(range(len(self.entity2id)))

        if(get_2hop):
            self.graph = self.get_graph()
            self.node_neighbors_2hop = self.get_further_neighbors()
        # self.node_neighbors_3hop = self.get_further_neighbors(3)
        # self.get_prob()

        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(len(self.valid_triples_dict), len(self.train_indices),
                                                                                                           len(self.validation_indices), len(self.test_indices)))

        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

    '''def get_relation_adjacency(self, batch_inputs, entity_embeddings):
        num_relation = len(self.relation2id)
        relation_dim = entity_embeddings.shape[1]

        entity_relation_connections = {}
        averaged_relation_vectors = torch.zeros(
            [num_relation, num_relation, relation_dim], dtype=torch.float32)
        count = torch.zeros([num_relation, num_relation])

        counts = 0
        batch_inputs_positive = batch_inputs[:self.batch_size, :]
        for i, t in enumerate(batch_inputs_positive):
            source = batch_inputs[i, 0]
            target = batch_inputs[i, 2]
            relation = batch_inputs[i, 1]

            if(target in entity_relation_connections.keys()):
                if(relation in entity_relation_connections[target].keys()):
                    entity_relation_connections[target][relation] += 1
                else:
                    entity_relation_connections[target][relation] = 1
                counts += 1

            else:
                entity_relation_connections[target] = {}
                entity_relation_connections[target][relation] = 1
                counts += 1

        for entity in entity_relation_connections.keys():
            for r1 in entity_relation_connections[entity].keys():
                for r2 in entity_relation_connections[entity].keys():
                    if(r1 == r2):
                        continue

                    # averaged_relation_vectors[r1][r2] += entity_relation_connections[entity][r1] * \
                    #     entity_relation_connections[entity][r2] * \
                    #     entity_embeddings[entity]
                    # count[r1][r2] += entity_relation_connections[entity][r1] * \
                    #     entity_relation_connections[entity][r2]

                    averaged_relation_vectors[r1][r2] += entity_embeddings[entity]
                    count[r1][r2] += 1

        count_div = torch.where(
            count > 0, count, 1e-6 * torch.ones([num_relation, num_relation]))

        averaged_relation_vectors = torch.FloatTensor(
            averaged_relation_vectors / count_div.unsqueeze(-1))
        return averaged_relation_vectors.cuda(), (count > 0).type(torch.LongTensor).cuda()'''

    # get batch corresponding to the iteration

    '''def get_gcn_adj(self):
        self.newids = {}
        for key in self.entity2id.keys():
            self.newids[key] = self.entity2id[key]

        degree_node = defaultdict(int)
        # print(self.newids)
        new_edge_list = []
        for ent in self.newids.values():
            new_edge_list.append((ent, ent))
            degree_node[ent] += 2

        for rel in self.relation2id.keys():
            self.newids[rel] = len(self.newids)

        for edge in self.train_indices:
            # print(edge)
            new_edge_list.append(
                (edge[0], self.newids[self.id2relation[edge[1]]]))
            # new_edge_list.append(
            #     (self.newids[self.id2relation[edge[1]]], edge[0]))
            new_edge_list.append(
                (self.newids[self.id2relation[edge[1]]], edge[2]))
            # new_edge_list.append(
            #     (edge[2], self.newids[self.id2relation[edge[1]]]))

            degree_node[edge[0]] += 1
            degree_node[edge[2]] += 1
            degree_node[self.newids[self.id2relation[edge[1]]]] += 1

        new_edge_list = np.array(new_edge_list).astype(np.int32)
        adj_values = torch.ones((len(new_edge_list)))
        for i in range(len(new_edge_list)):
            adj_values[i] /= degree_node[new_edge_list[i][0]]

        new_edge_list = torch.LongTensor(new_edge_list).transpose(1, 0)

        print(new_edge_list.size(), adj_values.size())

        adj_sparse = torch.sparse.LongTensor(new_edge_list, adj_values,
                                             torch.Size([len(self.newids), len(self.newids)]))
        # degree_node = torch.sparse.sum(adj_sparse, dim=1)

        return adj_sparse.cuda()'''

    def get_iteration_batch(self, iter_num):
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,self.batch_size * (iter_num + 1))

            self.batch_indices[:self.batch_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:self.batch_size,
                              :] = self.train_values[indices, :]

            # self.batch_values = self.lb.transform(self.batch_indices[:, 2])

            last_index = self.batch_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)
                # print("Sampling negative triples ")
                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

        else:
            last_iter_size = len(self.train_indices) - \
                self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))
            self.batch_indices[:last_iter_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:last_iter_size,
                              :] = self.train_values[indices, :]
            # self.batch_values = self.lb.transform(self.batch_indices[:, 2])

            last_index = last_iter_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

    def get_iteration_batch_nhop(self, current_batch_indices, node_neighbors, batch_size):

        self.batch_indices = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 4)).astype(np.int32)
        self.batch_values = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        # print(len(current_batch_indices), current_batch_indices.shape)
        # print("length of current_batch_indices is ", len(current_batch_indices))
        indices = random.sample(range(len(current_batch_indices)), batch_size)

        self.batch_indices[:batch_size,
                           :] = current_batch_indices[indices, :]
        self.batch_values[:batch_size,
                          :] = np.ones((batch_size, 1))

        last_index = batch_size

        if self.invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    # while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                    #        self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                    #     random_entities[current_index] = np.random.randint(
                    #         0, len(self.entity2id))
                    self.batch_indices[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (self.invalid_valid_ratio // 2) + \
                        (i * (self.invalid_valid_ratio // 2) + j)

                    # while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                    #        random_entities[current_index]) in self.valid_triples_dict.keys():
                    #     random_entities[current_index] = np.random.randint(
                    #         0, len(self.entity2id))
                    self.batch_indices[last_index + current_index,
                                       3] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

            return self.batch_indices, self.batch_values

        return self.batch_indices, self.batch_values

    def get_graph(self):
        graph = {}
        # print(self.train_adj_matrix[0].size())
        all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
            0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)
        # print(all_tiples.size())
        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value
        print("Graph created")
        return graph


    def bfs(self, graph, source, nbd_size=2):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))
        # print("Currently on source ", source)
        while(not q.empty()):
            top = q.get()
            # print("Current source entity is ", top[0])
            # print("Current source target entities are ", graph[top[0]].keys())
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1
                        
                        visit[target] = 1
                        if distance[target] > 2:
                            continue
                        parent[target] = (top[0], graph[top[0]][target])

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors

    def get_prob(self):
        self.relation_prob = np.zeros(len(self.relation2id))
        self.entity_prob = np.zeros(len(self.entity2id))

        for trip in self.train_triples:
            self.entity_prob[trip[2]] += 1.0
            self.relation_prob[trip[1]] += 1.0

        self.relation_prob /= len(self.train_triples)
        self.entity_prob /= len(self.train_triples)

    def get_further_neighbors(self, nbd_size=2):
        neighbors = {}
        start_time = time.time()
        print("length of graph keys is ", len(self.graph.keys()))
        for source in self.graph.keys():
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph, source, nbd_size)
            for distance in temp_neighbors.keys():
                if(source in neighbors.keys()):
                    if(distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]
            # print("done the current source, time taken is ", time.time()-st_time)

        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        return neighbors

    def get_batch_nhop_neighbors_ps(self, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        for source in batch_sources:
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():
                # num_neighbors=len(node_neighbors[source][nbd_size])
                # random_neighbor=random.randint(0, num_neighbors - 1)
                # random_neighbor_tuple=node_neighbors[source][nbd_size][random_neighbor]
                # batch_source_triples.append([source, random_neighbor_tuple[0][0], random_neighbor_tuple[0][1],
                #                              random_neighbor_tuple[1]])
                nhop_list = node_neighbors[source][nbd_size]
                prob_list = np.zeros(len(nhop_list))

                for i, tup in enumerate(nhop_list):
                    P = 1.0
                    for rel in tup[0]:
                        P *= (1 - self.relation_prob[rel])
                    for ent in tup[1]:
                        P *= (1 - self.entity_prob[ent])
                    prob_list[i] = P

                prob_list /= sum(prob_list)

                if(len(prob_list) > 1):
                    random_neighbor_index = np.random.choice(range(len(prob_list)),
                                                             1, p=list(prob_list))
                else:
                    random_neighbor_index = np.random.choice(range(len(prob_list)),
                                                             1, p=list(prob_list))

                for ind in random_neighbor_index:
                    batch_source_triples.append([source, nhop_list[ind][0][-1], nhop_list[ind][0][0],
                                                 nhop_list[ind][1][0]])

        return np.array(batch_source_triples).astype(np.int32)


    def get_batch_nhop_neighbors_all(self, args, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        print("length of unique_entities ", len(batch_sources))
        count = 0
        for source in batch_sources:
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():
                nhop_list = node_neighbors[source][nbd_size]

                for i, tup in enumerate(nhop_list):
                    if(args.partial_2hop and i >= 1):
                        break

                    count += 1

                    # if 'FB' in args.data:
                    # # for freebase obly
                    #     batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                    #                              nhop_list[i][1]])
                    # else:
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                             nhop_list[i][1][0]])
                    # print(batch_source_triples)
        print("count ", count)
        # print("len of batch_source_triples ", len(batch_source_triples))
        return np.array(batch_source_triples).astype(np.int32)



    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        # x = source_embeds + relation_embeds - tail_embeds
        x = source_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x

    def gat_eval_GAT(self, model, unique_entities):
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        start_time = time.time()
        # indices = random.sample(
        #     range(len(self.test_values)), len(self.test_values))
        indices = [i for i in range(len(self.test_indices))]
        batch_indices = self.test_indices[indices, :]
        # print("Sampled indices")

        entity_list = [j for i, j in self.entity2id.items()]

        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_ten_head, hits_at_ten_tail = 0, 0
        hits_at_three_head, hits_at_three_tail = 0, 0
        hits_at_one_head, hits_at_one_tail = 0, 0

        for i in range(batch_indices.shape[0]):
            # print(len(ranks_head))
            new_x_batch_head = np.tile(
                batch_indices[i, :], (len(self.entity2id), 1))
            new_x_batch_tail = np.tile(
                batch_indices[i, :], (len(self.entity2id), 1))

            if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                continue

            new_x_batch_head[:, 0] = entity_list
            new_x_batch_tail[:, 2] = entity_list

            last_index_head = []  # array of already existing triples
            last_index_tail = []
            for tmp_index in range(len(new_x_batch_head)):
                temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                    new_x_batch_head[tmp_index][2])
                if temp_triple_head in self.valid_triples_dict.keys():
                    last_index_head.append(tmp_index)

                temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                    new_x_batch_tail[tmp_index][2])
                if temp_triple_tail in self.valid_triples_dict.keys():
                    last_index_tail.append(tmp_index)

            # Deleting already existing triples, leftover triples are invalid, according
            # to train, validation and test data
            # Note, all of them maynot be actually invalid
            new_x_batch_head = np.delete(
                new_x_batch_head, last_index_head, axis=0)
            new_x_batch_tail = np.delete(
                new_x_batch_tail, last_index_tail, axis=0)

            # adding the current valid triples to the top, i.e, index 0
            new_x_batch_head = np.insert(
                new_x_batch_head, 0, batch_indices[i], axis=0)
            new_x_batch_tail = np.insert(
                new_x_batch_tail, 0, batch_indices[i], axis=0)

            # print(new_x_batch_head.shape)
            import math
            # Have to do this, because it doesn't fit in memory

            scores_head = self.transe_scoring(
                new_x_batch_head, model.final_entity_embeddings, model.final_relation_embeddings)
            # scores_head = model.batch_test(new_x_batch_head)
            sorted_scores_head, sorted_indices_head = torch.sort(
                scores_head.view(-1), dim=-1, descending=False)
            # Just search for zeroth index in the sorted scores, we appended valid triple at top
            ranks_head.append(
                np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
            reciprocal_ranks_head.append(1.0 / ranks_head[-1])

            scores_tail = self.transe_scoring(
                new_x_batch_tail, model.final_entity_embeddings, model.final_relation_embeddings)
            # scores_tail = model.batch_test(new_x_batch_tail)
            sorted_scores_tail, sorted_indices_tail = torch.sort(
                scores_tail.view(-1), dim=-1, descending=False)
            # Just search for zeroth index in the sorted scores, we appended valid triple at top
            ranks_tail.append(
                np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
            reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
            print("sample - ", ranks_head[-1], ranks_tail[-1])

        # print("Current iteration Ranks are {}".format(ranks))
        for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail = hits_at_100_tail + 1
            if ranks_tail[i] <= 10:
                hits_at_ten_tail = hits_at_ten_tail + 1
            if ranks_tail[i] <= 3:
                hits_at_three_tail = hits_at_three_tail + 1
            if ranks_tail[i] == 1:
                hits_at_one_tail = hits_at_one_tail + 1

        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)
        # print("here {}".format(len(ranks_head)))
        # print("\nCurrent iteration time {}".format(time.time() - start_time))
        # print("Stats for replacing head are -> ")
        # print("Current iteration Hits@100 are {}".format(
        #     hits_at_100_head / float(len(ranks_head))))
        # print("Current iteration Hits@10 are {}".format(
        #     hits_at_ten_head / len(ranks_head)))
        # print("Current iteration Hits@3 are {}".format(
        #     hits_at_three_head / len(ranks_head)))
        # print("Current iteration Hits@1 are {}".format(
        #     hits_at_one_head / len(ranks_head)))
        # print("Current iteration Mean rank {}".format(
        #     sum(ranks_head) / len(ranks_head)))
        # print("Current iteration Mean Reciprocal Rank {}".format(
        #     sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

        # print("\nStats for replacing tail are -> ")
        # print("Current iteration Hits@100 are {}".format(
        #     hits_at_100_tail / len(ranks_head)))
        # print("Current iteration Hits@10 are {}".format(
        #     hits_at_ten_tail / len(ranks_head)))
        # print("Current iteration Hits@3 are {}".format(
        #     hits_at_three_tail / len(ranks_head)))
        # print("Current iteration Hits@1 are {}".format(
        #     hits_at_one_tail / len(ranks_head)))
        # print("Current iteration Mean rank {}".format(
        #     sum(ranks_tail) / len(ranks_tail)))
        # print("Current iteration Mean Reciprocal Rank {}".format(
        #     sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

        average_hits_at_100_head.append(
            hits_at_100_head / len(ranks_head))
        average_hits_at_ten_head.append(
            hits_at_ten_head / len(ranks_head))
        average_hits_at_three_head.append(
            hits_at_three_head / len(ranks_head))
        average_hits_at_one_head.append(
            hits_at_one_head / len(ranks_head))
        average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
        average_mean_recip_rank_head.append(
            sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

        average_hits_at_100_tail.append(
            hits_at_100_tail / len(ranks_head))
        average_hits_at_ten_tail.append(
            hits_at_ten_tail / len(ranks_head))
        average_hits_at_three_tail.append(
            hits_at_three_tail / len(ranks_head))
        average_hits_at_one_tail.append(
            hits_at_one_tail / len(ranks_head))
        average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
        average_mean_recip_rank_tail.append(
            sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

        print("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                               + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))




    def get_validation_pred(self, args, model, unique_entities):
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()
            # indices = random.sample(
            #     0, len(self.test_values), len(self.test_values))
            # indices = random.sample(
            #     range(len(self.test_values)), len(self.test_values))
            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            entity_list = [j for i, j in self.entity2id.items()]

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_indices.shape[0]):
                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))
                new_x_batch_tail = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))

                if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys():
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_dict.keys():
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)

                # print(new_x_batch_head.shape)
                import math
                # Have to do this, because it doesn't fit in memory

                if 'WN' in args.data:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_head.shape[0] / 4))

                    scores1_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[:num_triples_each_shot, :]).cuda())
                    scores2_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    # scores5_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[9 * num_triples_each_shot:, :]).cuda())

                    scores_head = torch.cat(
                        [scores1_head, scores2_head, scores3_head, scores4_head], dim = 0)
                        #scores5_head, scores6_head, scores7_head, scores8_head, 
                        #cores9_head, scores10_head], dim=0)
                else:
                    scores_head = model.batch_test(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here

                if 'WN' in args.data:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_tail.shape[0] / 4))

                    scores1_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                    scores2_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    # scores5_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[9 * num_triples_each_shot:, :]).cuda())

                    scores_tail = torch.cat(
                        [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim = 0)
                    #     scores5_tail, scores6_tail, scores7_tail, scores8_tail, 
                    #     scores9_tail, scores10_tail], dim=0)

                else:
                    scores_tail = model.batch_test(new_x_batch_tail)
                    
                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)
                # print("current tail scores -> ",sorted_scores_tail)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                print("sample - ", ranks_head[-1], ranks_tail[-1])
                # print("time taken ", time.time()-start_time_it)

            # print("Current iteration Ranks are {}".format(ranks))
            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

            print("\nStats for replacing tail are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_tail / len(ranks_head)))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_tail / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_tail / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_tail / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_tail) / len(ranks_tail)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

        print("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                               + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))


    '''
	def get_validation_pred_relation(self, model, unique_entities):
        average_hits_at_100_head  = []
        average_hits_at_ten_head  = []
        average_hits_at_three_head  = []
        average_hits_at_one_head  = []
        average_mean_rank_head  = []
        average_mean_recip_rank_head = []

        for iters in range(1):
            start_time = time.time()
            # indices = random.sample(
            #     0, len(self.test_values), len(self.test_values))
            # indices = random.sample(
            #     range(len(self.test_values)), len(self.test_values))
            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            relation_list = [j for i, j in self.relation2id.items()]

            ranks_head = []
            reciprocal_ranks_head = []
            hits_at_100_head = 0
            hits_at_ten_head = 0
            hits_at_three_head = 0
            hits_at_one_head = 0

            for i in range(batch_indices.shape[0]):
                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.relation2id), 1))
                # new_x_batch_tail = np.tile(
                #     batch_indices[i, :], (len(self.entity2id), 1))

                
                new_x_batch_head[:, 1] = relation_list
                # new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys():
                        last_index_head.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                
                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)

                scores_head = model.batch_test(new_x_batch_head)
                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                print("sample - ", ranks_head[-1])
                print("time taken ", time.time()-start_time_it)

            # print("Current iteration Ranks are {}".format(ranks))
            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)

            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))'''



    def evaluate_convE(self, model):
        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_ten_head, hits_at_ten_tail = 0, 0
        hits_at_one_head, hits_at_one_tail = 0, 0

        batch_test = torch.LongTensor(self.test_indices).cuda()
        # if(CUDA):
        # batch_test = batch_test.cuda()

        scores_tail = model(batch_test)
        sorted_scores_tail, sorted_indices_tail = torch.sort(
            scores_tail, dim=-1, descending=True)
        # print(sorted_indices_tail.shape)

        # print(sorted_indices_tail.shape)

        for i in range(len(self.test_indices)):
            ranks_tail.append(
                np.where(sorted_indices_tail[i].cpu().numpy() == self.test_indices[i, 2])[0][0] + 1)
            reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
        print(self.test_indices[:10, 2])
        print(ranks_tail[:10])

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail = hits_at_100_tail + 1
            if ranks_tail[i] <= 10:
                hits_at_ten_tail = hits_at_ten_tail + 1
            if ranks_tail[i] == 1:
                hits_at_one_tail = hits_at_one_tail + 1

        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        print("\nStats for replacing tail are -> ")
        print("Current iteration Hits@100 are {}".format(
            hits_at_100_tail / len(self.test_values)))
        print("Current iteration Hits@10 are {}".format(
            hits_at_ten_tail / len(self.test_values)))
        print("Current iteration Hits@1 are {}".format(
            hits_at_one_tail / len(self.test_values)))
        print("Current iteration Mean rank {}".format(
            sum(ranks_tail) / len(ranks_tail)))
        print("Current iteration Mean Reciprocal Rank {}".format(
            sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

    def get_batch(self, iter_num):
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):
            # indices = np.random.randint(0, len(self.train_values), self.batch_size)

            last_iter_size = self.batch_size
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,self.batch_size * (iter_num + 1))

            self.batch_indices[:self.batch_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:self.batch_size, :] = self.train_values[indices, :]
        
            last_index = self.batch_size
        
            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)
        
                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))
        
                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio):
                        start_time = time.time()
                        count = 0
                        current_index = i * self.invalid_valid_ratio + j
                        temp_relation_index = self.batch_indices[last_index
                                                                 + current_index, 1]
                        prob = self.headTailSelector[temp_relation_index]
        
                        # Random sapling of negative entities as to be improved as shown in KBGAN paper
                        # Sample a random head entity
                        if (np.random.randint(np.iinfo(np.int32).max) % 1000) > prob:
                            while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                                   self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                                random_entities[current_index] = np.random.randint(
                                    0, len(self.entity2id))
        
                            self.batch_indices[last_index + current_index,
                                               0] = random_entities[current_index]
        
                        # Sample random tail entity
                        else:
                            while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                                   random_entities[current_index]) in self.valid_triples_dict.keys():
                                random_entities[current_index] = np.random.randint(
                                    0, len(self.entity2id))
        
                            self.batch_indices[last_index + current_index,
                                               2] = random_entities[current_index]
        
                        self.batch_values[last_index + current_index, :] = [0]
        
                return self.batch_indices, self.batch_values
        
            return self.batch_indices, self.batch_values

        else:
            last_iter_size = len(self.train_indices) - \
                self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))
            self.batch_indices[:last_iter_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:last_iter_size,
                              :] = self.train_values[indices, :]
            # self.batch_values = self.lb.transform(self.batch_indices[:, 2])

            last_index = last_iter_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)
        
                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))
        
                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio):
                        start_time = time.time()
                        count = 0
                        current_index = i * self.invalid_valid_ratio + j
                        temp_relation_index = self.batch_indices[last_index
                                                                 + current_index, 1]
                        prob = self.headTailSelector[temp_relation_index]
        
                        # Random sapling of negative entities as to be improved as shown in KBGAN paper
                        # Sample a random head entity
                        if (np.random.randint(np.iinfo(np.int32).max) % 1000) > prob:
                            while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                                   self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                                random_entities[current_index] = np.random.randint(
                                    0, len(self.entity2id))
        
                            self.batch_indices[last_index + current_index,
                                               0] = random_entities[current_index]
        
                        # Sample random tail entity
                        else:
                            while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                                   random_entities[current_index]) in self.valid_triples_dict.keys():
                                random_entities[current_index] = np.random.randint(
                                    0, len(self.entity2id))
        
                            self.batch_indices[last_index + current_index,
                                               2] = random_entities[current_index]
        
                        self.batch_values[last_index + current_index, :] = [0]
        
                return self.batch_indices, self.batch_values

    def get_validation_pred_relation(self, model, unique_entities):
        average_hits_at_100_head  = []
        average_hits_at_ten_head  = []
        average_hits_at_three_head  = []
        average_hits_at_one_head  = []
        average_mean_rank_head  = []
        average_mean_recip_rank_head = []

        for iters in range(1):
            start_time = time.time()
            # indices = random.sample(
            #     0, len(self.test_values), len(self.test_values))
            # indices = random.sample(
            #     range(len(self.test_values)), len(self.test_values))
            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            relation_list = [j for i, j in self.relation2id.items()]

            ranks_head = []
            reciprocal_ranks_head = []
            hits_at_100_head = 0
            hits_at_ten_head = 0
            hits_at_three_head = 0
            hits_at_one_head = 0

            for i in range(batch_indices.shape[0]):
                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.relation2id), 1))
                # new_x_batch_tail = np.tile(
                #     batch_indices[i, :], (len(self.entity2id), 1))

                if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                
                new_x_batch_head[:, 1] = relation_list
                # new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys():
                        last_index_head.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                
                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)

                scores_head = model.batch_test(new_x_batch_head)
                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # print("sample - ", ranks_head[-1])
                # print("time taken ", time.time()-start_time_it)

            # print("Current iteration Ranks are {}".format(ranks))
            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)

            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))
