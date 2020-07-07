import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree

import math
import torch_scatter

### GIN convolution along the graph structure
# class GATLayer(torch.nn.Module):
#     def __init__(self, emb_dim, num_heads):
#         '''
#             emb_dim (int): node embedding dimensionality
#         '''
#         #super(GATLayer, self).__init__(aggr = "add")
#         super().__init__()
#         #self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
#         #self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
#         #self.edge_encoder = torch.nn.Linear(7, emb_dim)
#         self.q_proj = torch.nn.Linear(emb_dim, emb_dim)
#         self.k_proj = torch.nn.Linear(emb_dim, emb_dim)
#         self.v_proj = torch.nn.Linear(emb_dim, emb_dim)

#         self.num_heads = num_heads
#         self.head_size = (int)(emb_dim / num_heads)

#         self.out_proj = torch.nn.Linear(emb_dim, emb_dim)
#         #self.layer_norm = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

#         self.fc1 = torch.nn.Linear(emb_dim, emb_dim * 2)
#         self.fc2 = torch.nn.Linear(emb_dim * 2, emb_dim)
#         #self.h_dropout = nn.Dropout(hid_dropout)

#         #self.final_layer_norm = torch.nn.LayerNorm(emb_dim, eps=1e-6) 


#     def split_head(self, x):
#         # E * num_heads, head_size
#         new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
#         x = x.view(*new_x_shape)
#         # [E, num_heads, head_size]
#         return x  #x.permute(1, 0, 2).contiguous()

#     def forward(self, node_embed, edge_emb, edge_index):
#         x_i = node_embed[edge_index[0]]
#         x_j = node_embed[edge_index[1]]

#         #edge_embedding = self.edge_encoder(edge_attr)
#         x_j = edge_emb + x_j # torch.cat([edge_emb, x_j], 1)

#         q = self.q_proj(x_i)
#         k = self.k_proj(x_j)
#         v = self.v_proj(x_j)

#         q = self.split_head(q)
#         k = self.split_head(k)
#         #[E, num_heads, head_size]
#         v = self.split_head(v)
#         att = (q * k).sum(dim=2).view(-1, self.num_heads) / math.sqrt(self.head_size)

#         # [E, num_heads]
#         att_probs = torch_scatter.composite.scatter_softmax(att, edge_index[0], dim=0)

#         #print(att_probs.shape)
#         new_v = att_probs.view( *(att_probs.size()[:] + (1,)) ) * v
#         new_v = new_v.view(new_v.shape[0], -1)

#         #print(new_v.shape)

#         att_v = torch_scatter.scatter_add(new_v, edge_index[0], dim=0)
        
#         #print(att_v.shape)

#         att_h = self.out_proj(att_v)

#         #h0 = att_h + node_embed
#         h0 = att_h + node_embed # self.layer_norm(att_h + node_embed)

#         h1 = self.fc1(h0)
#         h1 = F.gelu(h1)
#         h2 = self.fc2(h1)
#         h = h0 + h2 # self.final_layer_norm(h0 + h2)

#         return h

class GATLayer(torch.nn.Module):
    def __init__(self, emb_dim, num_heads):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        #super(GATLayer, self).__init__(aggr = "add")
        super().__init__()
        #self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        #self.eps = torch.nn.Parameter(torch.Tensor([0]))
        #self.edge_encoder = torch.nn.Linear(7, emb_dim)
        self.edge_encoder = torch.nn.Linear(7, emb_dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        #self.h_dropout = nn.Dropout(hid_dropout)
        #self.final_layer_norm = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

    def forward(self, node_embed, edge_index, edge_attr):
        edge_emb = self.edge_encoder(edge_attr)
        #x_i = node_embed[edge_index[0]]
        x_j = node_embed[edge_index[1]]
        #edge_embedding = self.edge_encoder(edge_attr)
        x_j = F.relu(edge_emb + x_j) # torch.cat([edge_emb, x_j], 1)
        _v = torch_scatter.scatter_add(x_j, edge_index[0], dim=0)
        out = self.mlp((1 + self.eps) * node_embed +  _v)
        return out

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)    
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
                
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch


        ### computing input node embedding

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        
        return node_representation


if __name__ == "__main__":
    pass