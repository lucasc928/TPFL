import os
import sys
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from layers.graph import Graph

import time


class Feeder(torch.utils.data.Dataset):


    def __init__(self, data_path, graph_args={}):

        self.data_path = data_path
        self.load_data()
        self.graph = Graph(**graph_args)  # num_node = 120,max_hop = 1

    def load_data(self):
        with open(self.data_path, 'rb') as reader:
            # Training (N, C, T, V)=(5010, 11, 12, 120), (5010, 120, 120), (5010, 2)
            [self.all_feature, self.all_adjacency, self.all_mean_xy] = pickle.load(reader)

    def __len__(self):
        return len(self.all_feature)

    def __getitem__(self, idx):
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        now_feature = self.all_feature[idx].copy()  # (C, T, V) = (11, 12, 120)
        now_mean_xy = self.all_mean_xy[idx].copy()  # (2,) = (x, y)


        if  np.random.random() > 0.5:
            angle = 2 * np.pi * np.random.random()
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)
            angle_mat = np.array(
                [[cos_angle, -sin_angle],
                 [sin_angle, cos_angle]])
            xy = now_feature[3:5, :, :]
            num_xy = np.sum(xy.sum(axis=0).sum(axis=0) != 0)  # get the number of valid data

            out_xy = np.einsum('ab,btv->atv', angle_mat, xy)
            now_mean_xy = np.matmul(angle_mat, now_mean_xy)
            xy[:, :, :num_xy] = out_xy[:, :, :num_xy]

            now_feature[3:5, :, :] = xy

        now_adjacency = self.graph.get_adjacency(self.all_adjacency[idx])
        now_A = self.graph.normalize_adjacency(now_adjacency)

        return now_feature, now_A, now_mean_xy
