import random

from sklearn.utils import compute_class_weight
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, GATConv
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from model_set import encoders


class SmilesModel(nn.Module):
    def __init__(self, args):
        super(SmilesModel, self).__init__()
        self.device = args.device
        self.mask_ratio = args.mask_ratio
        self.smiles_input_dim = args.smiles_input_dim
        self.smiles_hidden_size = args.smiles_hidden_size
        self.smiles_latent_size = args.smiles_latent_size
        self.smiles_input_size_graph = args.smiles_input_size_graph
        self.drop = args.dropout
        self.smiles_encoder = encoders.Encoder(self.smiles_input_dim, self.smiles_hidden_size,
                                               self.smiles_latent_size, self.device,
                                               dropout=self.drop)
        self.smiles_AtomEmbedding = nn.Embedding(self.smiles_input_size_graph,
                                                 self.smiles_hidden_size).to(self.device)
        self.smiles_AtomEmbedding.weight.requires_grad = True
        self.dropout_smiles = nn.Dropout(self.drop)

    def samples_mask(self, samples, dict_=None):
        max_len = 0
        mask_id = torch.tensor(dict_['<mask>'])
        noisy_ratio = self.mask_ratio
        for s in samples:
            max_len = max(len(s), max_len)
        x = np.zeros((len(samples), max_len)).astype('int64')
        x_mask = np.zeros((len(samples), max_len)).astype('float32')
        for idx in range(len(samples)):
            ss = samples[idx]
            mask_index = random.sample(range(len(ss)), int(len(ss) * noisy_ratio))
            x[idx, : len(ss)] = ss
            for i in mask_index:
                x[idx, i] = mask_id
            x_mask[idx, :len(ss)] = 1
        return torch.tensor(x).to(self.device), torch.tensor(x_mask).to(self.device)

    def get_mask(self, data, dict_):
        batch_data = data
        x_s, x_s_mask = self.samples_mask(batch_data, dict_=dict_)
        return x_s, x_s_mask

    def forward(self, samples_seq, node2index):
        x_s, x_s_mask = self.get_mask(samples_seq, node2index)
        x_s_emb = self.dropout_smiles(self.smiles_AtomEmbedding(x_s))
        enc_states_seq = self.smiles_encoder(x_s_emb, x_mask=x_s_mask)
        smiles_output = torch.tanh(torch.mean(enc_states_seq, dim=1))
        return smiles_output

    def pred(self, samples_seq):
        x_s_emb = self.dropout_smiles(self.smiles_AtomEmbedding(samples_seq))  #
        x_s_emb = x_s_emb[None, :, :]
        enc_states_seq = self.smiles_encoder(x_s_emb)
        enc_states = enc_states_seq
        smiles_output = torch.tanh(torch.mean(enc_states, dim=1))
        return smiles_output

class GraphModel(nn.Module):
    def __init__(self, args):
        super(GraphModel, self).__init__()
        self.device = args.device
        self.drop = args.dropout
        self.node_mask_ratio = args.node_mask_ratio
        self.edge_mask_ratio = args.edge_mask_ratio
        self.graph_mask_id = args.graph_mask_id
        self.graph_in_size = args.graph_input_size
        self.graph_hidden_size = args.graph_hidden_size
        self.graph_latent_size = args.graph_latent_size
        self.graph_conv1 = get_convs(args.graph_conv1, self.graph_in_size, self.graph_hidden_size)
        self.graph_conv2 = get_convs(args.graph_conv2, self.graph_hidden_size, self.graph_hidden_size)
        self.dropout_graph = nn.Dropout(self.drop)

    def samples_mask(self, nodes, edges):
        node_mask_ratio = self.node_mask_ratio
        edge_mask_ratio = self.edge_mask_ratio
        num_nodes = nodes.shape[0]
        num_edges = edges.shape[1]
        node_mask_num = max(int(num_nodes * node_mask_ratio), 1)
        node_mask_indices = torch.randperm(num_nodes)[:node_mask_num]
        masked_node_feature = nodes.clone()
        masked_node_feature[node_mask_indices] = self.graph_mask_id
        edge_mask_num = max(int(num_edges * edge_mask_ratio), 1)
        edge_mask_indices = torch.randperm(num_edges)[:edge_mask_num]
        kept_edge_indices = [i for i in range(num_edges) if i not in edge_mask_indices]
        kept_edge_index = edges[:, kept_edge_indices]
        if len(kept_edge_indices) == 0:
            kept_edge_index = edges[:, :1]
        return Data(x=masked_node_feature, edges_index=kept_edge_index)

    def get_output(self, graph_batch, graph_output):
        graph_node_index, cnt, max_len = 0, 0, 0
        graph_cnt = []
        for index, idx in enumerate(graph_batch):
            if idx != graph_node_index:
                graph_cnt.append(cnt)
                if cnt > max_len:
                    max_len = cnt
                graph_node_index = idx
                cnt = 1
            else:
                cnt += 1
                if index == len(graph_batch) - 1:
                    graph_cnt.append(cnt)
                    if cnt > max_len:
                        max_len = cnt
        begin_index = 0
        graph_outputs = torch.zeros(len(graph_cnt), max_len, self.graph_latent_size)
        for index, cnt in enumerate(graph_cnt):
            end_index = begin_index + cnt
            graph_output_ = graph_output[begin_index:end_index, :]
            begin_index = end_index
            graph_outputs[index, :cnt, :] = graph_output_
        return graph_outputs

    def get_mask(self, nodes, edges):
        masked_graph_data = [self.samples_mask(node, edge) for node, edge in zip(nodes, edges)]
        masked_graph_data = Batch.from_data_list(masked_graph_data)
        masked_node_features = masked_graph_data.x.to(self.device)
        kept_edge_index = masked_graph_data.edges_index.to(self.device)
        return masked_node_features, kept_edge_index, masked_graph_data.batch

    def forward(self, samples_node, samples_edge):
        masked_node_features, kept_edge_index, graph_batch = self.get_mask(samples_node, samples_edge)
        graph_output = self.graph_conv1(masked_node_features, kept_edge_index)
        graph_output = F.relu(graph_output)
        graph_output = self.dropout_graph(graph_output)
        graph_output = self.graph_conv2(graph_output, kept_edge_index)
        graph_batch_ = graph_batch.tolist()
        graph_output = self.get_output(graph_batch_, graph_output)
        graph_output = torch.tanh(torch.mean(graph_output, dim=1)).to(self.device)
        return graph_output

    def pred(self, samples_node, samples_edge):
        graph_data = Data(x=samples_node, edge_index=samples_edge)
        graph_output = self.graph_conv1(graph_data.x, graph_data.edge_index)
        graph_output = self.dropout_graph(graph_output)  #
        graph_output = F.relu(graph_output)
        graph_output = self.graph_conv2(graph_output, samples_edge)
        graph_outputs = torch.tanh(torch.mean(graph_output, dim=0)).unsqueeze(0).to(self.device)
        return graph_outputs


def get_convs(graph_conv_type, graph_in_size, graph_out_size):
    if graph_conv_type == 'GCN':
        return GCNConv(graph_in_size, graph_out_size)
    elif graph_conv_type == 'GAT':
        return GATConv(graph_in_size, graph_out_size)
    elif graph_conv_type == 'SAGE':
        return SAGEConv(graph_in_size, graph_out_size)
    else:
        raise NotImplementedError('Graph convolution type {} not implemented'.format(graph_conv_type))


class CombinedModel(nn.Module):
    def __init__(self, args):
        super(CombinedModel, self).__init__()
        self.mito_size = args.mito_size
        self.smiles_latent_size = args.smiles_latent_size
        self.graph_latent_size = args.graph_latent_size
        self.combine_latent_size = args.smiles_latent_size + args.graph_latent_size
        self.graph_model = GraphModel(args)
        self.smiles_model = SmilesModel(args)
        self.device = args.device
        self.sequence = args.sequence
        self.graph = args.graph
        self.mito = args.mito
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_smiles = args.lr_smiles
        self.lr_graph = args.lr_graph
        self.drop = args.dropout

        self.smiles_output_layer = encoders.Classifier(self.smiles_latent_size, self.mito_size, self.device, self.mito)
        self.graph_output_layer = encoders.Classifier(self.graph_latent_size, self.mito_size, self.device, self.mito)
        self.combine_output_layer = encoders.Classifier(self.combine_latent_size, self.mito_size, self.device, self.mito)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.optimizer_smiles = torch.optim.Adam(self.smiles_model.parameters(), lr=self.lr_smiles)
        self.optimizer_graph = torch.optim.Adam(self.graph_model.parameters(), lr=self.lr_graph)
        self.criterion = nn.CrossEntropyLoss()

    def get_mito(self, batch_size, mitos):
        mitos_ = torch.zeros(batch_size, self.mito_size).to(self.device)
        for index, mito in enumerate(mitos):
            for i in range(mitos_.shape[1]):
                mitos_[index, i] = mito.clone().detach().to(self.device)
        return mitos_

    def add_mito(self, batch_size, output, mitos):
        total_output = torch.zeros(batch_size, output.shape[1] + mitos.shape[1]).to(self.device)
        total_output[:, :output.shape[1]] = output
        total_output[:, output.shape[1]:] = mitos
        return total_output

    def forward(self, graph_index, data, labels, mitos, node2index):
        smiles_output, graph_output, total_output = None, None, None
        data_label = labels
        data_mito = mitos
        samples_seq = [data['sequence'][idx] for idx in graph_index]
        samples_node = [data['atom_features'][idx] for idx in graph_index]
        samples_edge = [data['edges'][idx] for idx in graph_index]
        samples_mito = [data_mito[idx] for idx in graph_index]
        # print(samples_seq)

        if self.graph:
            graph_output = self.graph_model(samples_node, samples_edge)
        if self.sequence:
            smiles_output = self.smiles_model(samples_seq, node2index)

        samples_mito = self.get_mito(self.batch_size, samples_mito)
        if self.sequence and self.graph:
            total_output = torch.cat((graph_output, smiles_output), dim=-1)
            if self.mito:
                total_output = self.add_mito(self.batch_size, total_output, samples_mito)
            final_output = self.combine_output_layer(total_output)
        elif self.sequence:
            if self.mito:
                total_output = self.add_mito(self.batch_size, smiles_output, samples_mito)
            else:
                total_output = smiles_output
            final_output = self.smiles_output_layer(total_output)
        else:
            if self.mito:
                total_output = self.add_mito(self.batch_size, graph_output, samples_mito)
            else:
                total_output = graph_output
            final_output = self.graph_output_layer(total_output)
        labels = torch.LongTensor([torch.as_tensor(data_label[idx], dtype=torch.int32) \
                                   for idx in graph_index]).to(self.device)
        if final_output.shape[0] != labels.shape[0]:
            final_output = torch.unsqueeze(final_output, 0)
        loss = self.criterion(final_output, labels)
        self.optimizer.zero_grad()
        if self.sequence:
            self.optimizer_smiles.zero_grad()
        if self.graph:
            self.optimizer_graph.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.sequence:
            self.optimizer_smiles.step()
        if self.graph:
            self.optimizer_graph.step()
        return final_output, loss

    def pred(self, graph_index, data, mitos):
        smiles_output, graph_outputs, total_output = None, None, None
        data_mito = mitos
        samples_seq = data['sequence'][graph_index].to(self.device)
        samples_node = data['atom_features'][graph_index].to(self.device)
        samples_edge = data['edges'][graph_index].to(self.device)
        samples_mito = [torch.tensor(data_mito[graph_index])]
        # print(data['smiles'][graph_index])
        if self.graph:
            graph_outputs = self.graph_model.pred(samples_node, samples_edge)
        if self.sequence:
            smiles_output = self.smiles_model.pred(samples_seq)
        samples_mito_ = self.get_mito(1, samples_mito)
        if self.sequence and self.graph:
            total_output = torch.cat((graph_outputs, smiles_output), dim=-1)
            if self.mito:
                total_output = self.add_mito(1, total_output, samples_mito_)
            final_output = self.combine_output_layer(total_output)
        elif self.sequence:
            if self.mito:
                total_output = self.add_mito(1, smiles_output, samples_mito_)
            else:
                total_output = smiles_output
            final_output = F.softmax(self.smiles_output_layer(total_output), dim=1)
        else:
            if self.mito:
                total_output = self.add_mito(1, graph_outputs, samples_mito_)
            else:
                total_output = graph_outputs
            final_output = self.graph_output_layer(total_output)
        return final_output
