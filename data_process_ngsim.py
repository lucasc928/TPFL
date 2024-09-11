import numpy as np
import glob
import os
from scipy import spatial
import pickle
import scipy.io
import pandas as pd
# Please change this to your location
from prediction.dataset import ApolloscapeDataset
from prediction.model.GRIP import GRIPDataLoader

data_root = './data/NGSIM'

# Baidu ApolloScape data format:
# Frame Number,Vehicle Id,v_class,Local X,Local Y,Length,width,Lane Id,velocity(feet/second),acceleration(feet/second2), Lateral maneuver,Longitudinal maneuver
total_feature_dimension = 10 + 1  # we add mark "1" to the end of each row to indicate that this row exists

history_frames = 15
future_frames = 25
total_frames = history_frames + future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 260  # maximum number of observed objects is 70
neighbor_distance = 10  # meter




def get_frame_instance_dict(pra_file_path):
    with open(pra_file_path, 'r') as reader:
        content = np.array([x.strip().split(' ') for x in reader.readlines()])
        now_dict = {}
        for row in content:
            n_dict = now_dict.get(row[0], {})
            n_dict[row[1]] = row
            now_dict[row[0]] = n_dict
    return now_dict


def process_data(pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last):
    visible_object_id_list = list(
        pra_now_dict[pra_observed_last].keys())
    num_visible_object = len(visible_object_id_list)
    visible_object_value = np.array(list(pra_now_dict[pra_observed_last].values()))
    xy = visible_object_value[:, 3:5].astype(float)
    mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
    m_xy = np.mean(xy, axis=0)
    mean_xy[3:5] = m_xy
    dist_xy = spatial.distance.cdist(xy, xy)
    neighbor_matrix = np.zeros((max_num_object, max_num_object))
    neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy < neighbor_distance).astype(int)
    now_all_object_id = set([val for x in range(pra_start_ind, pra_end_ind) for val in pra_now_dict[x].keys()])
    non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
    num_non_visible_object = len(non_visible_object_id_list)

    object_feature_list = []
    for frame_ind in range(pra_start_ind, pra_end_ind):
        now_frame_feature_dict = {obj_id: (
            list(pra_now_dict[frame_ind][obj_id] - mean_xy) + [1]
            if obj_id in visible_object_id_list
            else list(pra_now_dict[frame_ind][obj_id] - mean_xy) + [0])
            for obj_id in pra_now_dict[frame_ind]}
        now_frame_feature = np.array(
            [now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in
             visible_object_id_list + non_visible_object_id_list])

        object_feature_list.append(now_frame_feature)
    object_feature_list = np.array(object_feature_list)
    object_frame_feature = np.zeros((max_num_object, pra_end_ind - pra_start_ind, total_feature_dimension))
    object_frame_feature[:num_visible_object + num_non_visible_object] = np.transpose(object_feature_list, (1, 0, 2))
    return object_frame_feature, neighbor_matrix, m_xy


def generate_test_data(pra_file_path):
    now_dict = get_frame_instance_dict(pra_file_path)
    frame_id_set = sorted(set(now_dict.keys()))

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []

    start_frame_id_list = frame_id_set[::history_frames]
    for start_ind in start_frame_id_list:
        start_ind = int(start_ind)
        end_ind = int(start_ind + history_frames)
        observed_last = start_ind + history_frames - 1

        visible_object_id_list = list(
            now_dict[observed_last].keys())
        num_visible_object = len(visible_object_id_list)

        now_all_object_id = set([val for x in range(start_ind, end_ind) for val in now_dict[x].keys()])
        non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
        num_non_visible_object = len(non_visible_object_id_list)
        if num_visible_object + num_non_visible_object > max_num_object:
            continue
        object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)
        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)
        all_mean_list.append(mean_xy)

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    # print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list


def generate_data(pra_file_path_list, pra_is_train=True, pra_is_val=False):
    obs_length = 15
    pred_length = 25
    time_step = 0.2
    skip = 5
    sample_step = 5
    # Create an instance of the ApolloscapeDataset class
    dataset = ApolloscapeDataset(obs_length, pred_length, time_step, sample_step)
    graph_args = {'max_hop': 2, 'num_node': 260}

    all_data = []
    all_adjacency = []
    all_mean_xy = []

    for file_path in pra_file_path_list:
        all_feature_list = []
        all_adjacency_list = []
        all_mean_list = []
        if pra_is_train:
            data_generator = dataset.format_data(file_path)
            for input_data in data_generator:
                # Create an instance of the GRIPDataLoader class
                dataloader = GRIPDataLoader(obs_length, pred_length, graph_args)
                # Preprocess the input data using the data_loader
                object_frame_feature, neighbor_matrix, mean_xy = dataloader.preprocess(
                    input_data)

                all_feature_list.append(object_frame_feature)
                all_adjacency_list.append(neighbor_matrix)
                all_mean_list.append(mean_xy)

            all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
            all_adjacency_list = np.array(all_adjacency_list)
            all_mean_list = np.array(all_mean_list)


        elif pra_is_val:
            data_generator = dataset.format_data(file_path)
            for input_data in data_generator:
                # Create an instance of the GRIPDataLoader class
                dataloader = GRIPDataLoader(obs_length, pred_length, graph_args)
                # Preprocess the input data using the data_loader
                object_frame_feature, neighbor_matrix, mean_xy = dataloader.preprocess(
                    input_data)

                all_feature_list.append(object_frame_feature)
                all_adjacency_list.append(neighbor_matrix)
                all_mean_list.append(mean_xy)
            all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
            all_adjacency_list = np.array(all_adjacency_list)
            all_mean_list = np.array(all_mean_list)
        else:
            data_generator = dataset.format_data(file_path)
            for input_data in data_generator:
                # Create an instance of the GRIPDataLoader class
                dataloader = GRIPDataLoader(obs_length, pred_length, graph_args)
                # Preprocess the input data using the data_loader
                object_frame_feature, neighbor_matrix, mean_xy = dataloader.preprocess(
                    input_data)
                all_feature_list.append(object_frame_feature)
                all_adjacency_list.append(neighbor_matrix)
                all_mean_list.append(mean_xy)
            all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
            all_adjacency_list = np.array(all_adjacency_list)
            all_mean_list = np.array(all_mean_list)

        all_data.extend(all_feature_list)
        all_adjacency.extend(all_adjacency_list)
        all_mean_xy.extend(all_mean_list)

    all_data = np.array(all_data)  # (N, C, T, V)
    all_adjacency = np.array(all_adjacency)
    all_mean_xy = np.array(all_mean_xy)

    # Train (N, C, T, V)=(4701，11，40，260) (4701， 260，260) (4701， 2)
    # Val (N, C, T, V)=(1212, 260， 40，11)(1212, 260， 260) (1212, 2)
    # Test (N, C, T, V)=(3020， 260，40，11) (3020， 260， 260) (3020，2)
    print(np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy))

    # save training_data and trainjing_adjacency into a file.
    if pra_is_train:
        save_path = 'train_data_ng.pkl'
    elif pra_is_val:
        save_path = 'val_data_ng.pkl'
    else:
        save_path = 'test_data_ng.pkl'
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data, all_adjacency, all_mean_xy], writer)


if __name__ == '__main__':
    train_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_train.new/')))
    val_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_val.new/')))
    test_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_test.new/')))

    print('Generating Training Data.')
    generate_data(train_file_path_list, pra_is_train=True)

    print('Generating Val Data.')
    generate_data(val_file_path_list, pra_is_train=False, pra_is_val=True)

    print('Generating Testing Data.')
    generate_data(test_file_path_list, pra_is_train=False)
