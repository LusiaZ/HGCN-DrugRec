import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_dim, out_dim, num_routes):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_routes = num_routes
        self.route_weights = nn.Parameter(torch.randn(num_routes, num_capsules, in_dim, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(x, self.route_weights)
        u_hat = u_hat.squeeze(4)

        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1).to(x.device)
        num_iterations = 3
        for _ in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, tensor):
        tensor_norm = (tensor ** 2).sum(-1, keepdim=True)
        scale = tensor_norm / (1 + tensor_norm)
        return scale * tensor / torch.sqrt(tensor_norm + 1e-9)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = self.linear(x)
        return x
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
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class SimpleHyperGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleHyperGraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, hypergraph_adj):
        x = torch.mm(hypergraph_adj, x)
        x = self.linear(x)
        return x


class HyperGCN(nn.Module):
    def __init__(self, voc_size, emb_dim, hypergraph_adj, graph_adj, device):
        super(HyperGCN, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(voc_size, emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.to(self.device)

        self.hypergraph_adj = hypergraph_adj
        self.graph_adj = graph_adj

        self.hypergcn1 = SimpleHyperGraphConvolution(emb_dim, emb_dim)
        self.hypergcn2 = SimpleHyperGraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embeddings = self.embedding.weight
        node_embeddings = F.relu(self.linear(node_embeddings))

        if self.hypergraph_adj.size(0) != node_embeddings.size(0):
            raise ValueError(
                f"Shape mismatch: hypergraph_adj: {self.hypergraph_adj.size()}, node_embeddings: {node_embeddings.size()}"
            )

        hypergraph_embeddings = self.hypergcn1(node_embeddings, self.hypergraph_adj)
        hypergraph_embeddings = F.relu(hypergraph_embeddings)
        hypergraph_embeddings = self.hypergcn2(hypergraph_embeddings, self.hypergraph_adj)

        graph_embeddings = torch.mm(self.graph_adj, node_embeddings)

        node_embeddings = hypergraph_embeddings + graph_embeddings
        return node_embeddings

class HGCN(nn.Module):
    def __init__(self, vocab_size, ehr_hypergraph_adj, ddi_adj, emb_dim, device, ddi_in_memory=True):
        super(HGCN, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.ddi_in_memory = ddi_in_memory

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(len(vocab_size))]
        )
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(len(vocab_size) - 1)]
        )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.diagnosis_capsule = CapsuleLayer(num_capsules=10, in_dim=emb_dim, out_dim=16, num_routes=9195)
        self.procedure_capsule = CapsuleLayer(num_capsules=10, in_dim=emb_dim, out_dim=16, num_routes=9195)

        if isinstance(ehr_hypergraph_adj, torch.Tensor):
            ehr_hypergraph_adj = ehr_hypergraph_adj.to(device)
        else:
            raise TypeError("Unsupported type for ehr_hypergraph_adj")

        self.ehr_hypergcn = HyperGCN(
            voc_size=vocab_size[2],
            emb_dim=emb_dim,
            hypergraph_adj=ehr_hypergraph_adj[:vocab_size[2], :vocab_size[2]],
            graph_adj=torch.FloatTensor(np.eye(vocab_size[2])).to(device),
            device=device
        )

        self.ddi_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device
        )

        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # Initialize the learnable weight parameter alpha
        self.alpha = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim * 2),  # 输入维度应该是 emb_dim * 2
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

        self.init_weights()

    def forward(self, input):
        i1_seq = []
        i2_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)

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

        if i1_seq.size(1) != i2_seq.size(1):
            max_len = max(i1_seq.size(1), i2_seq.size(1))
            i1_seq = F.pad(i1_seq, (0, 0, 0, max_len - i1_seq.size(1)))
            i2_seq = F.pad(i2_seq, (0, 0, 0, max_len - i2_seq.size(1)))

        o1, h1 = self.encoders[0](i1_seq)
        o2, h2 = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)
        queries = self.query(patient_representations)

        query = queries[-1:]

        if self.ddi_in_memory:
            ehr_memory = self.ehr_hypergcn()
            ddi_memory = self.ddi_gcn()
            drug_memory = ehr_memory - ddi_memory * self.inter
        else:
            drug_memory = self.ehr_hypergcn()

        if len(input) > 1:
            history_keys = queries[: (queries.size(0) - 1)]

            history_values = np.zeros((len(input) - 1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input) - 1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device)

        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)
        fact1 = torch.mm(key_weights1, drug_memory)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()), dim=-1)
            weighted_values = torch.mm(visit_weight, history_values)  # 修复形状问题
            fact2 = torch.mm(weighted_values, drug_memory)
        else:
            fact2 = fact1

        # Compute the weighted sum of fact1 and fact2 using alpha
        weighted_fact = self.alpha * fact1 + (1 - self.alpha) * fact2

        # Concatenate the weighted_fact with query
        output = self.output(torch.cat([query, weighted_fact], dim=-1))

        if self.training:
            neg_pred_prob = torch.sigmoid(output)
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
        self.alpha.data.uniform_(0, 1)

