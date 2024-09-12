import math
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch

from TCN import TemporalConvNet
import config
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = config.parse()


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class STMHGNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, pool=args.pooling_strategy):
        super(STMHGNet, self).__init__()
        self.pool = pool
        self.num_H = args.num_H
        self.time_length = args.num_time_length
        self.dropout = args.dropout_rate
        self.num_layers = args.hgnn_layer
        self.node_num = args.num_node
        self.input_dim = input_dim

        # TCN
        self.use_tcn = args.use_tcn
        if self.use_tcn:
            self.tcn_channel = args.tcn_channel
            self.tcn_outdim = args.tcn_outdim
            self.tcn = TemporalConvNet(1, self.tcn_channel, norm_strategy='batchnorm', kernel_size=args.tcn_kernel)
            self.temporal_lin = nn.Sequential(nn.Linear(self.tcn_channel[-1] * self.time_length, self.tcn_outdim),
                                              nn.ReLU(), nn.Dropout(self.dropout),
                                              nn.Linear(self.tcn_outdim, self.tcn_outdim))
            self.input_dim = self.tcn_outdim

        # Hyperedge weight parameters
        self.weight_all = nn.Parameter(torch.Tensor(self.node_num * self.num_H))
        # self.weight_all = torch.ones((self.node_num * self.num_H))
        self.reset_parameters()

        # HGNNconv layer
        self.convs = nn.ModuleList()
        self.convs.append(HGNN_conv(self.input_dim, hidden_dim1))
        self.convs.append(HGNN_conv(hidden_dim1, hidden_dim2))

        for _ in range(self.num_layers - 2):
            self.convs.append(HGNN_conv(self.hidden_dim2, self.hidden_dim2))

        # Read out layer
        if self.pool == 'graph_mean_max_add_pool':
            self.read_out_in = hidden_dim2 * 3 * self.num_layers
        elif self.pool == 'graph_mean_max_pool' or self.pool == 'graph_mean_add_pool':
            self.read_out_in = hidden_dim2 * 2 * self.num_layers
        else:
            self.read_out_in = hidden_dim2 * self.num_layers
            
        self.read_out = nn.Sequential(
            nn.Linear(self.read_out_in, hidden_dim2), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim2, hidden_dim2), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim2, output_dim)
        )

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight_all)

    def forward(self, data):
        x, H_k, H_lasso, H_elastic, batch, batchsize = data.x, data.H_k, data.H_lasso, data.H_elastic, data.batch, data.y

        if self.use_tcn:
            x = x.view(-1, 1, self.time_length)
            x = self.tcn(x)
            x = x.view(x.size()[0], self.tcn_channel[-1] * self.time_length)
            x = self.temporal_lin(x)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)

        if self.num_H == 3:
            H = torch.cat((H_lasso, H_elastic, H_k), dim=1)
        elif self.num_H == 2:
            H = torch.cat((H_lasso, H_elastic), dim=1)
        else:
            H = H_lasso

        weight_hedge = self.weight_all.to(x.device)
        H = H.view(batchsize.shape[0], self.node_num, self.node_num * self.num_H)
        De = torch.diag_embed(torch.sum(H, dim=1))
        De = De.view(batchsize.shape[0], self.node_num * self.num_H, self.node_num * self.num_H)
        De2 = utils.invert_nonzero_elements(De)
        Dv = torch.diag_embed(torch.matmul(H, torch.abs(weight_hedge)))
        Dv2 = utils.invert_nonzero_elements(Dv) ** 0.5
        W = torch.abs(torch.diag(weight_hedge))
        L = torch.matmul(torch.matmul(Dv2, torch.matmul(H, W)), torch.matmul(torch.matmul(De2, H.transpose(-1, -2)), Dv2))  # Calculate Laplacian Matrix

        x = x.view(batchsize.shape[0], self.node_num, self.input_dim)
        emb = []
        for i in range(self.num_layers):
            x = self.convs[i](x, L)
            out = x.view(batchsize.shape[0] * self.node_num, -1)

            # Pooling
            if self.pool == 'graph_mean_pool':
                emb.append(pyg_nn.global_mean_pool(out, batch))
            elif self.pool == 'graph_max_pool':
                emb.append(pyg_nn.global_max_pool(out, batch))
            elif self.pool == 'graph_add_pool':
                emb.append(pyg_nn.global_add_pool(out, batch))
            elif self.pool == 'graph_mean_max_pool':
                x_mean = pyg_nn.global_mean_pool(out, batch)
                x_max = pyg_nn.global_max_pool(out, batch)
                emb.append(torch.cat([x_mean, x_max], dim=1))
            elif self.pool == 'graph_mean_add_pool':
                x_mean = pyg_nn.global_mean_pool(out, batch)
                x_add = pyg_nn.global_add_pool(out, batch)
                emb.append(torch.cat([x_mean, x_add], dim=1))
            elif self.pool == 'graph_mean_max_add_pool':
                x_mean = pyg_nn.global_mean_pool(out, batch)
                x_max = pyg_nn.global_max_pool(out, batch)
                x_add = pyg_nn.global_add_pool(out, batch)
                emb.append(torch.cat([x_mean, x_max, x_add], dim=1))
            x = F.dropout(F.relu(x), self.dropout, training=self.training)

        x = torch.cat(emb, dim=1)
        x = self.read_out(x)

        return out, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)



