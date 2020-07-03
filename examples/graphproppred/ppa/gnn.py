import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode, GATLayer

from torch_scatter import scatter_mean

class GAT(torch.nn.Module):
    def __init__(self, num_class, num_layer = 5, emb_dim = 256, num_heads=8):
        super(GAT, self).__init__()

        self.num_class = num_class
        self.emb_dim = emb_dim
        self.num_layer = num_layer

        # node feature dimension 2;
        # edge feature dimension 9;
        self.node_encoder = torch.nn.Embedding(2, emb_dim) # uniform input node embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        self.layer_norm = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

        #GATLayer(self, emb_dim, num_heads):
        self.layers = nn.ModuleList([GATLayer(emb_dim, num_heads) for _ in range(num_layer)])

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)
        
    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        #x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        # node embedding.
        n_emb = self.node_encoder(x)
        e_emb = self.edge_encoder(edge_attr)    

        h = self.layer_norm(n_emb)

        for layer in range(self.num_layer):
            # def forward(self, node_embed, edge_emb, edge_index):
            h = self.layers[layer](h, e_emb, edge_index)

        h_graph = h[-batch_size:]
        return self.graph_pred_linear(h_graph)


class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_class = 10)