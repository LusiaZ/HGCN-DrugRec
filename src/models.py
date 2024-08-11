import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
import math
from torch.nn.parameter import Parameter

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device("cpu:0")):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)
        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs
        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

"""
DMNC
"""
class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device("cpu:0")):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim)
                for i in range(K)
            ]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [
                DNC(
                    input_size=emb_dim,
                    hidden_size=emb_dim,
                    rnn_type="gru",
                    num_layers=1,
                    num_hidden_layers=1,
                    nr_cells=16,
                    cell_size=emb_dim,
                    read_heads=1,
                    batch_first=True,
                    gpu_id=0,
                    independent_linears=False,
                )
                for _ in range(K - 1)
            ]
        )
        self.decoder = nn.GRU(
            emb_dim + emb_dim * 2, emb_dim * 2, batch_first=True
        )
        self.interface_weighting = nn.Linear(
            emb_dim * 2, 2 * (emb_dim + 1 + 3)
        )
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)
        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)
    def forward(self, input, i1_state=None, i2_state=None, h_n=None, max_len=20):
        i1_input_tensor = self.embeddings[0](
            torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device)
        )
        i2_input_tensor = self.embeddings[1](
            torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device)
        )
        o1, (ch1, m1, r1) = self.encoders[0](
            i1_input_tensor, (None, None, None) if i1_state is None else i1_state
        )
        o2, (ch2, m2, r2) = self.encoders[1](
            i2_input_tensor, (None, None, None) if i2_state is None else i2_state
        )
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)
        predict_sequence = [self.token_start] + input[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)
        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)
        if self.training:
            for item in predict_sequence:
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device)
                )
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(
                    h_n.squeeze(0)
                )
                r1, _ = self.read_from_memory(
                    self.encoders[0],
                    read_keys[:, 0, :].unsqueeze(dim=1),
                    read_strengths[:, 0].unsqueeze(dim=1),
                    read_modes[:, 0, :].unsqueeze(dim=1),
                    i1_state[1],
                )
                r2, _ = self.read_from_memory(
                    self.encoders[1],
                    read_keys[:, 1, :].unsqueeze(dim=1),
                    read_strengths[:, 1].unsqueeze(dim=1),
                    read_modes[:, 1, :].unsqueeze(dim=1),
                    i2_state[1],
                )
                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device)
            )
            for idx in range(max_len):
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(
                    h_n.squeeze(0)
                )
                r1, _ = self.read_from_memory(
                    self.encoders[0],
                    read_keys[:, 0, :].unsqueeze(dim=1),
                    read_strengths[:, 0].unsqueeze(dim=1),
                    read_modes[:, 0, :].unsqueeze(dim=1),
                    i1_state[1],
                )
                r2, _ = self.read_from_memory(
                    self.encoders[1],
                    read_keys[:, 1, :].unsqueeze(dim=1),
                    read_strengths[:, 1].unsqueeze(dim=1),
                    read_modes[:, 1, :].unsqueeze(dim=1),
                    i2_state[1],
                )
                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)
                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device)
                )
        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(
            read_key, read_str, read_mode, m_hidden
        )
        return read_vectors, hidden

    def decode_read_variable(self, input):
        w = 64
        r = 2
        b = input.size(0)
        input = self.interface_weighting(input)
        read_keys = F.tanh(input[:, : r * w].contiguous().view(b, r, w))
        read_strengths = F.softplus(input[:, r * w : r * w + r].contiguous().view(b, r))
        read_modes = F.softmax(input[:, (r * w + r) :].contiguous().view(b, r, 3), -1)
        return read_keys, read_strengths, read_modes

class GAMENet(nn.Module):
    def __init__(
        self,
        vocab_size,
        ehr_adj,
        ddi_adj,
        emb_dim=64,
        device=torch.device("cpu:0"),
        ddi_in_memory=True,
    ):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K - 1)]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K - 1)]
        )
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.ehr_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device
        )
        self.ddi_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device
        )
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )
        self.init_weights()

    def forward(self, input):
        i1_seq = []
        i2_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i2 = mean_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)
        i2_seq = torch.cat(i2_seq, dim=1)
        o1, h1 = self.encoders[0](i1_seq)
        o2, h2 = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(
            dim=0
        )
        queries = self.query(patient_representations)
        query = queries[-1:]
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter
        else:
            drug_memory = self.ehr_gcn()
        if len(input) > 1:
            history_keys = queries[: (queries.size(0) - 1)]
            history_values = np.zeros((len(input) - 1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input) - 1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(
                self.device
            )
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)
        fact1 = torch.mm(key_weights1, drug_memory)
        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()))
            weighted_values = visit_weight.mm(history_values)
            fact2 = torch.mm(weighted_values, drug_memory)
        else:
            fact2 = fact1
        output = self.output(torch.cat([query, fact1, fact2], dim=-1))

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
            return output, batch_neg
        else:
            return output

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        self.inter.data.uniform_(-initrange, initrange)

class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device("cpu:0")):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]
        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.5),
        )
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)
        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        device = self.device
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            input_tmp.extend(
                list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1])
            )
            if len(input_tmp) < max_len:
                input_tmp.extend([self.input_len] * (max_len - len(input_tmp)))

            input_np.append(input_tmp)
        visit_emb = self.embedding(
            torch.LongTensor(input_np).to(device)
        )
        visit_emb = torch.sum(visit_emb, dim=1)
        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0))
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0))
        g = g.squeeze(dim=0)
        h = h.squeeze(dim=0)
        attn_g = F.softmax(self.alpha_li(g), dim=-1)
        attn_h = F.tanh(self.beta_li(h))
        c = attn_g * attn_h * visit_emb
        c = torch.sum(c, dim=0).unsqueeze(dim=0)
        return self.output(c)


class MICRON(nn.Module):
    def __init__(self, vocab_size, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(MICRON, self).__init__()
        self.device = device
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.health_net = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size[2])
        )
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.init_weights()

    def forward(self, input):
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)
        diag_emb = sum_embedding(self.dropout(
            self.embeddings[0](torch.LongTensor(input[-1][0]).unsqueeze(dim=0).to(self.device))))  # (1,1,dim)
        prod_emb = sum_embedding(
            self.dropout(self.embeddings[1](torch.LongTensor(input[-1][1]).unsqueeze(dim=0).to(self.device))))
        if len(input) < 2:
            diag_emb_last = diag_emb * torch.tensor(0.0)
            prod_emb_last = diag_emb * torch.tensor(0.0)
        else:
            diag_emb_last = sum_embedding(self.dropout(
                self.embeddings[0](torch.LongTensor(input[-2][0]).unsqueeze(dim=0).to(self.device))))  # (1,1,dim)
            prod_emb_last = sum_embedding(
                self.dropout(self.embeddings[1](torch.LongTensor(input[-2][1]).unsqueeze(dim=0).to(self.device))))
        health_representation = torch.cat([diag_emb, prod_emb], dim=-1).squeeze(dim=0)  # (seq, dim*2)
        health_representation_last = torch.cat([diag_emb_last, prod_emb_last], dim=-1).squeeze(dim=0)  # (seq, dim*2)
        health_rep = self.health_net(health_representation)[-1:, :]  # (seq, dim)
        health_rep_last = self.health_net(health_representation_last)[-1:, :]  # (seq, dim)
        health_residual_rep = health_rep - health_rep_last
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)
        rec_loss = 1 / self.tensor_ddi_adj.shape[0] * torch.sum(
            torch.pow((F.sigmoid(drug_rep) - F.sigmoid(drug_rep_last + drug_residual_rep)), 2))
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 1 / self.tensor_ddi_adj.shape[0] * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
