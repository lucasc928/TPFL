import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GRIP'))
import torch
from scipy import spatial
from prediction.model.base.dataloader import DataLoader
from layers.graph import Graph
import numpy as np


class GRIPDataLoader(DataLoader):
    def __init__(self, obs_length=15, pred_length=25, graph_args={}, dataset=None):
        super().__init__(obs_length, pred_length)
        self.max_num_object = graph_args["num_node"]
        self.neighbor_distance = 10
        self.graph = Graph(**graph_args)
        self.dev = 'cuda:0'
        self.dataset = dataset

    def preprocess(self, input_data, rescale_x=669.691 / 2, rescale_y=669.691 / 2):
        rescale_xy = torch.ones((1, 2, 1, 1)).to(self.dev)
        rescale_xy[:, 0] = float(rescale_x)
        rescale_xy[:, 1] = float(rescale_y)

        total_feature_dimension = input_data["feature_dimension"] + 6

        # GRIP maintains invisible objects
        visible_object_id_list = []
        non_visible_object_id_list = []
        for obj_id, obj in input_data["objects"].items():
            if obj["visible"]:
                visible_object_id_list.append(obj_id)
            else:
                non_visible_object_id_list.append(obj_id)
        num_visible_object = len(visible_object_id_list)
        num_non_visible_object = len(non_visible_object_id_list)

        # get object index of perturbation target if any
        obj_index = {_obj_id: index for index, _obj_id in
                     enumerate(visible_object_id_list + non_visible_object_id_list)}

        # compute the mean values of x and y for zero-centralization.
        visible_object_value = np.array([
            np.concatenate((input_data["objects"][obj_id]["observe_trace"][self.obs_length - 1, :],
                            input_data["objects"][obj_id]["observe_feature"][self.obs_length - 1, :]), axis=0) for
            obj_id in visible_object_id_list])
        xy = visible_object_value[:, :2].astype(float)
        mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
        m_xy = np.mean(xy, axis=0)
        mean_xy[:2] = m_xy

        # compute distance between any pair of two objects
        dist_xy = spatial.distance.cdist(xy, xy)
        # if their distance is less than $neighbor_distance, we regard them are neighbors.
        neighbor_matrix = np.zeros((self.max_num_object, self.max_num_object))
        neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy < self.neighbor_distance).astype(int)

        # for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
        object_feature_list = []
        # non_visible_object_feature_list = []
        for frame_ind in range(self.seq_length):
            now_frame_feature_dict = {}
            # we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1]
            # -mean_xy is used to zero_centralize data
            # now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
            for obj_id, obj in input_data["objects"].items():
                if frame_ind < self.obs_length:
                    feature_data = np.concatenate((obj["observe_trace"][frame_ind, :],
                                                   obj["observe_feature"][frame_ind, :]), axis=0)
                    existence = obj["observe_mask"][frame_ind]
                else:
                    feature_data = np.concatenate((obj["future_trace"][frame_ind - self.obs_length, :],
                                                   obj["future_feature"][frame_ind - self.obs_length, :]), axis=0)
                    existence = obj["future_mask"][frame_ind - self.obs_length]

                if existence:
                    if obj_id in visible_object_id_list:
                        now_frame_feature_dict[obj_id] = [frame_ind, int(obj_id), obj["type"]] + list(
                            feature_data - mean_xy) + [1]
                    else:
                        now_frame_feature_dict[obj_id] = [frame_ind, int(obj_id), obj["type"]] + list(
                            feature_data - mean_xy) + [0]
                # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
            now_frame_feature = np.array(
                [now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in
                 visible_object_id_list + non_visible_object_id_list])
            object_feature_list.append(now_frame_feature)

        # object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
        object_feature_list = np.array(object_feature_list)

        # object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
        object_frame_feature = np.zeros((self.max_num_object, self.seq_length, total_feature_dimension))

        # np.transpose(object_feature_list, (1,0,2))
        object_frame_feature[:num_visible_object + num_non_visible_object] = np.transpose(object_feature_list,
                                                                                          (1, 0, 2))

        # result: object_frame_feature, neighbor_matrix, m_xy (function process_data)
        return object_frame_feature, neighbor_matrix, m_xy






        # now_adjacency = self.graph.get_adjacency(all_adjacency_list)
        # now_A = self.graph.normalize_adjacency(now_adjacency)
        #
        # _ori_data, A, mean_xy = all_feature_list, np.array([now_A]), all_mean_list
        # ori_data = torch.from_numpy(_ori_data).cuda()
        # A = torch.from_numpy(A).cuda()
        #
        # feature_id = [3, 4, 9, 10]
        # no_norm_loc_data = ori_data[:, feature_id]
        # data = no_norm_loc_data.clone()
        #
        # new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0)
        # data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
        # data[:, :2, 0] = 0
        # # # small vehicle: 1, big vehicles: 2, pedestrian 3, bicycle: 4, others: 5
        # object_type = ori_data[:, 2:3]
        # data = data.float().to(self.dev)
        # no_norm_loc_data = no_norm_loc_data.float().to(self.dev)
        # object_type = object_type.to(self.dev)  # type
        # data[:, :2] = data[:, :2] / rescale_xy
        # # result: data, no_norm_loc_data, object_type (function main.py:preprocess_data)
        #
        # _input_data = data[:, :, :self.obs_length, :]  # (N, C, T, V)=(N, 4, 6, 120)
        # output_loc_GT = data[:, :2, self.obs_length:, :]  # (N, C, T, V)=(N, 2, 6, 120)
        # output_mask = data[:, -1:, self.obs_length:, :]  # (N, C, T, V)=(N, 1, 6, 120)
        # A = A.float().to(self.dev)
        #
        # return _input_data, A, _ori_data, mean_xy, rescale_xy, no_norm_loc_data, output_loc_GT, output_mask, obj_index
