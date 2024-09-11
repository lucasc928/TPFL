import numpy as np
import glob
import os
from scipy import spatial
import pickle

data_root = './data/apolloscape/'

# frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
total_feature_dimension = 10 + 1  # we add mark "1" to the end of each row to indicate that this row exists
history_frames = 6  # 3 second * 2 frame/second
future_frames = 6  # 3 second * 2 frame/second
total_frames = history_frames + future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 120  # maximum number of observed objects is 70
neighbor_distance = 10  # meter


def get_frame_instance_dict(pra_file_path):
    '''
    Read raw data from files and return a dictionary in the following format:
        {frame_id:
            {object_id:
                # 10 features
                [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading]
            }
        }
    The dictionary's key represents the frame ID, corresponding to a specific frame.
    For each frame, the value is also a dictionary, where the key is the object ID, representing an individual object.
    For each object, the value is a list of 10 elements that store information such as frame ID, object ID, object type, position coordinates, dimensions, and orientation.
    By using nested dictionaries, the data for each object in a frame can be stored inside a dictionary, which itself is nested within the outer dictionary keyed by the frame ID.
    This structure allows easy access to the object data within a specific frame without the need to iterate through the entire dataset.
    '''
    with open(pra_file_path, 'r') as reader:
        content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float)
        now_dict = {}
        for row in content:
            n_dict = now_dict.get(row[0], {})
            n_dict[row[1]] = row
            now_dict[row[0]] = n_dict
    return now_dict


def process_data(pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last):
    '''
    Using real-time target tracking, each frame detects and tracks objects.
    From the last frame, we determine the number of tracked objects and extract their values. `visible_object_value` is a 2D array where rows represent objects and columns represent features.
    We compute the average x and y coordinates of all visible objects and calculate the Euclidean distances between them, storing the results in `dist_xy` for clustering.
    A neighbor matrix (120*120) is created, marking positions with distances below the threshold as neighbors.
    The IDs of all objects within the time window are stored in a set, and we find objects that disappeared by the last frame.
    For each frame, a feature vector is generated for each object, adjusted by subtracting the mean coordinates (`mean_xy`). Visible objects get a mask of 1, invisible ones get 0.
    These feature vectors are stored in `object_feature_list`, then converted to NumPy arrays.
    Finally, `object_frame_feature`, a zero array, is created to store features across all frames, simplifying the calculation of the node similarity matrix.
    '''

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


def generate_train_data(pra_file_path):
    '''
    Read data from $pra_file_path, and split data into clips with $total_frames length.
    Return: feature and adjacency_matrix
        feture: (N, C, T, V)
            N is the number of training data
            C is the dimension of features, 10raw_feature + 1mark(valid data or not)
            T is the temporal length of the data. history_frames + future_frames
            V is the maximum number of objects. zero-padding for less objects.
    '''
    now_dict = get_frame_instance_dict(pra_file_path)
    frame_id_set = sorted(set(now_dict.keys()))

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    for start_ind in frame_id_set[:-total_frames + 1]:
        start_ind = int(start_ind)
        end_ind = int(start_ind + total_frames)
        observed_last = start_ind + history_frames - 1
        object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)
        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)
        all_mean_list.append(mean_xy)

    all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))  # (N, V, T, C) --> (N, C, T, V)
    all_adjacency_list = np.array(all_adjacency_list)  # (N, V, V)
    all_mean_list = np.array(all_mean_list)  # (N, V, 2)

    return all_feature_list, all_adjacency_list, all_mean_list


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
        object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)
        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)
        all_mean_list.append(mean_xy)

    all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    return all_feature_list, all_adjacency_list, all_mean_list


def generate_data(pra_file_path_list, pra_is_train=True):
    all_data = []
    all_adjacency = []
    all_mean_xy = []
    for file_path in pra_file_path_list:
        if pra_is_train:
            now_data, now_adjacency, now_mean_xy = generate_train_data(file_path)
        else:
            now_data, now_adjacency, now_mean_xy = generate_test_data(file_path)
        all_data.extend(now_data)
        all_adjacency.extend(now_adjacency)
        all_mean_xy.extend(now_mean_xy)

    all_data = np.array(all_data)
    all_adjacency = np.array(all_adjacency)
    all_mean_xy = np.array(all_mean_xy)

    # Train (N, C, T, V)=(5010, 11, 12, 120) (5010, 120, 120) (5010, 2)
    # Test (N, C, T, V)=(415, 11, 6, 120) (415, 120, 120) (415, 2)
    print(np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy))

    if pra_is_train:
        save_path = 'train_data_ap.pkl'
    else:
        save_path = 'test_data_ap.pkl'
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data, all_adjacency, all_mean_xy], writer)


if __name__ == '__main__':
    train_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_train/*.txt')))
    test_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

    print('Generating Training Data.')
    generate_data(train_file_path_list, pra_is_train=True)

    print('Generating Testing Data.')
    generate_data(test_file_path_list, pra_is_train=False)
