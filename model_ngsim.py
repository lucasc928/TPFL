import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq, EncoderRNN
import numpy as np
from layers.fusion import fusion_update


def reputation_computer(vehicle_info):
    vehicle_type = vehicle_info[1]
    vehicle_length = vehicle_info[2]
    vehicle_width = vehicle_info[3]
    vehicle_Vel = vehicle_info[4]
    vehicle_Acc = vehicle_info[5]
    reputation = (vehicle_type * 0.2 + vehicle_length * 0.1 + vehicle_width * 0.3 +
                  vehicle_Vel * 0.1 + vehicle_Acc * 0.3)
    reputation = torch.sigmoid(reputation)
    return reputation.item()


def compute_cosine_similarity(x1, x2):
    return cosine(x1, x2)


class Model(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_args)
        A = np.ones((graph_args['max_hop'] + 1, graph_args['num_node'], graph_args['num_node']))
        self.credibility_mean = np.ones((graph_args['max_hop'] + 1, graph_args['num_node'], graph_args['num_node']))
        spatial_kernel_size = np.shape(A)[0]
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.st_gcn_networks = nn.ModuleList((
            nn.BatchNorm2d(in_channels),
            Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.num_node = num_node = self.graph.num_node
        self.out_dim_per_node = out_dim_per_node = 2  # (x, y) coordinate
        self.seq2seq_car = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                   isCuda=True)
        self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                     isCuda=True)
        self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                    isCuda=True)

    def reshape_for_lstm(self, feature):
        N, C, T, V = feature.size()
        now_feat = feature.permute(0, 3, 2, 1).contiguous()
        now_feat = now_feat.view(N * V, T, C)
        return now_feat

    def reshape_from_lstm(self, predicted):
        # predicted (N*V, T, C)
        NV, T, C = predicted.size()
        now_feat = predicted.view(-1, self.num_node, T,
                                  self.out_dim_per_node)  # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
        now_feat = now_feat.permute(0, 3, 2, 1).contiguous()  # (N, C, T, V)
        return now_feat

    def forward(self, pra_x, pra_A, vehicle_rep, reputation, iteration, pra_pred_length, pra_teacher_forcing_ratio=0,
                pra_teacher_location=None):
        x = pra_x  # trainable_graph(64, 4, NHF, 260)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            if type(gcn) is nn.BatchNorm2d:
                x = gcn(x)  # 对x进行批量归一化，提高训练稳定性
            # if x.shape[2] == 11:
            #     x, _ = gcn(x, pra_A + self.credibility_mean * importance)
            else:
                x, _ = gcn(x, pra_A + importance)
            # else:
            #     x, _ = gcn(x.clone(), pra_A)
        x_modified = x.clone()
        now_epoch = 0
        if x.shape[0] == 64:

            if iteration % 50 == 0 and x.shape[2] == 39 and iteration != 0:
                for t in range(x.shape[2]):
                    features = x[:, :, t, :]
                    for i in range(x.shape[0]):
                        similarities = []
                        for j in range(x.shape[3]):
                            if j != i:
                                similarity = compute_cosine_similarity(
                                    features[:, :, i].flatten().cpu().detach().numpy(),
                                    features[:, :, j].flatten().cpu().detach().numpy())
                                similarities.append((j, similarity))
                        count_similarity = 0
                        for j, similarity in similarities:
                            if similarity <= 0.8 or similarity == 0:
                                reputation[i, 1, t, j] = 0
                                count_similarity += 1
                            else:
                                reputation[i, 1, t, j] = reputation_computer(vehicle_rep[i, :, t, j])
                        x_modified[i, :, t, :] = features[i, :, :]  # 将修改后的特征矩阵保存到 x_modified 中
                if now_epoch % 5 == 0:  # noRL badnode==
                    print("————————star PPO————————")
                    x_modified = fusion_update(x_modified, reputation)  # 将修改后的 x_modified 传递给 fusion_update 函数
            else:
                x_modified = x  # 不需要进行修改时，将 x 赋值给 x_modified

        graph_conv_feature = self.reshape_for_lstm(x_modified)

        last_position = self.reshape_for_lstm(pra_x[:, :2])  # (N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]
        if pra_teacher_forcing_ratio > 0 and type(pra_teacher_location) is not type(None):
            pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

        now_predict_car = self.seq2seq_car(in_data=graph_conv_feature, last_location=last_position[:, -1:, :],
                                           pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio,
                                           teacher_location=pra_teacher_location)
        now_predict_car = self.reshape_from_lstm(now_predict_car)  # (N, C, T, V)

        now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:, -1:, :],
                                               pred_length=pra_pred_length,
                                               teacher_forcing_ratio=pra_teacher_forcing_ratio,
                                               teacher_location=pra_teacher_location)
        now_predict_human = self.reshape_from_lstm(now_predict_human)  # (N, C, T, V)

        now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:, -1:, :],
                                             pred_length=pra_pred_length,
                                             teacher_forcing_ratio=pra_teacher_forcing_ratio,
                                             teacher_location=pra_teacher_location)
        now_predict_bike = self.reshape_from_lstm(now_predict_bike)  # (N, C, T, V)

        now_predict = (now_predict_car + now_predict_human + now_predict_bike) / 3.

        return now_predict


if __name__ == '__main__':
    model = Model(in_channels=4, pred_length=25, graph_args={}, edge_importance_weighting=True)
    print(model)
