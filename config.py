import argparse


def parse():
    parser = argparse.ArgumentParser()

    # Data_Parameter
    parser.add_argument('--k', type=int, default=5, help='the number of fold')
    parser.add_argument('--data_root', type=str, default='./datasets', help='root directory of the dataset')
    parser.add_argument('--dataset_name', type=str, default='SZ', help='name of dataset')
    parser.add_argument('--num_node', type=int, default=312, help='num of ROIs, i.e. Nodes')
    parser.add_argument('--num_time_length', type=int, default=195, help='length of time series')
    parser.add_argument('--num_class', type=int, default=2, help='num of classification')
    parser.add_argument('--adj_threshold_ratio', type=int, default=20, help='Ratio of adj matrix retention threshold')
    parser.add_argument('--pooling_strategy', type=str, default='graph_mean_max_pool', help='mean/max/add/mean_max/mean_add/mean_max_add')
    parser.add_argument('--num_H', type=int, default=1, help='num of Incidence matrix for Fusion')

    # Hyperparameter
    parser.add_argument('--num_epochs', type=int, default=200, help='max epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--step_size', type=int, default=30, help='scheduler step size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='regularization')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate of GNN and HGNN')
    parser.add_argument('--patience', type=int, default=50, help='patience in the early stop mechanism')

    # HGNN_GNN_Layer_Parameter
    parser.add_argument('--hgnn_layer', type=int, default=2, help='number of MHGNNconv layers')
    parser.add_argument('--input_dim', type=int, default=195, help='input dimension of MHGCNconv')
    parser.add_argument('--hidden_dim1', type=int, default=32, help='hidden dimension of MHGCNconv')
    parser.add_argument('--hidden_dim2', type=int, default=32, help='hidden dimension (output) of MHGCNconv')

    # TCN_Layer_Parameter
    parser.add_argument('--use_tcn', type=bool, default=False, help='whether to use TCN')
    parser.add_argument('--tcn_channel', type=list, default=[8, 16], help='channel of TCN')
    parser.add_argument('--tcn_outdim', type=int, default=32, help='output feature dim of TCN')
    parser.add_argument('--tcn_kernel', type=int, default=7, help='convolutional kernel size of TCN')

    # Save_Model
    parser.add_argument('--save_path', type=str, default='./saved_model/', help='path to save model')
    return parser.parse_args()