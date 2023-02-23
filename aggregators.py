import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


class Aggregator(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, act, self_included):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included

    def forward(self, relation_label, self_vectors, neighbor_vectors, masks):
        # self_vectors: [batch_size, -1, input_dim]
        # neighbor_vectors: [batch_size, -1, 2, n_neighbor, input_dim]
        # masks: [batch_size, -1, 2, n_neighbor, 1]
        '''
        ggregated_neighbors = self.aggregated_neighbors.unsqueeze(1)  # [batch_size, 1, n_relations]  在第1个维度添加一个维度
        attention_weights = torch.sum(aggregated_neighbors * inputs, dim=-1)  # [batch_size, path_samples]
        attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, path_samples]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, path_samples, 1]
        output = torch.sum(attention_weights * inputs, dim=1)  # [batch_size, n_relations]
        '''
        relation_label = torch.reshape(relation_label, (self.batch_size, -1, 1, 1, self.input_dim))  # [batch_size, -1, 1, 1, input_dim]
        attention_w  = torch.sum(neighbor_vectors * relation_label, dim=-1)  # [batch_size, -1, 2 ,n_neighbor, input_dim]
        attention_w  = F.softmax(attention_w, dim=-1)                      # [batch_size, -1, 2 ,n_neighbor]
        attention_w  = attention_w.unsqueeze(-1)                           # [batch_size, -1, 2 ,n_neighbor, 1]
        entity_vectors = torch.mean(attention_w * neighbor_vectors * masks, dim=-2)  # [batch_size, -1, 2, input_dim]
        outputs = self._call(self_vectors, entity_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        pass


class MeanAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(MeanAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)

        self.layer = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = torch.mean(entity_vectors, dim=-2)  # [batch_size, -1, input_dim]
        if self.self_included:
            output += self_vectors
        output = output.view([-1, self.input_dim])  # [-1, input_dim]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(ConcatAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)

        multiplier = 3 if self_included else 2

        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = entity_vectors.view([-1, self.input_dim * 2])  # [-1, input_dim * 2]
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim=-1)  # [-1, input_dim * 3]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)


class CrossAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(CrossAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)

        addition = self.input_dim if self.self_included else 0

        self.layer = nn.Linear(self.input_dim * self.input_dim + addition, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        # [batch_size, -1, 1, input_dim]
        entity_vectors_a, entity_vectors_b = torch.chunk(entity_vectors, 2, dim=-2)
        entity_vectors_a = entity_vectors_a.view([-1, self.input_dim, 1])
        entity_vectors_b = entity_vectors_b.view([-1, 1, self.input_dim])
        output = torch.matmul(entity_vectors_a, entity_vectors_b)  # [-1, input_dim, input_dim]
        output = output.view([-1, self.input_dim * self.input_dim])  # [-1, input_dim * input_dim]
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim=-1)  # [-1, input_dim * input_dim + input_dim]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)
