import numpy as np
import random
import torch, torch.nn.functional as F

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M), dtype=data_numpy.dtype)
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1)**2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :,
                                                forward_map[t]].transpose(
                                                    1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(
                sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall


def multi_input(data, conn, inputs):
    C, T, V, M = data.shape
    joint = np.zeros((C*2, T, V, M))
    velocity = np.zeros((C*2, T, V, M))
    bone = np.zeros((C*2, T, V, M))
    bone_motion = np.zeros((C*2, T, V, M))
    joint[:C,:,:,:] = data
    data_new = []
    if inputs.isupper():
        if 'J' in inputs:
            for i in range(V):
                joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
            data_new.append(joint)
        if 'V' in inputs:
            for i in range(T-2):
                velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
                velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
            data_new.append(velocity)
        if 'B' in inputs:
            for i in range(len(conn)):
                bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
            bone_length = 0
            for i in range(C):
                bone_length += bone[i,:,:,:] ** 2
            bone_length = np.sqrt(bone_length) + 0.0001
            for i in range(C):
                bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
            data_new.append(bone)
        if 'M' in inputs:
            for i in range(T-2):
                bone_motion[:C,i,:,:] = bone[:C,i+1,:,:] - bone[:C,i,:,:]
                bone_motion[C:,i,:,:] = bone[:C,i+2,:,:] - bone[:C,i,:,:]
            data_new.append(bone_motion)
    elif inputs == 'joint':
        data_new = [joint[:C,:,:,:]]
    elif inputs == 'bone':
        for i in range(len(conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
        data_new = [bone[:C]]
    elif inputs == 'motion':
        for i in range(T-2):
            velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        data_new = [velocity[:C]]
    elif inputs == 'bone_motion':
        for i in range(len(conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
        for i in range(T-2):
            bone_motion[:C,i,:,:] = bone[:C,i+1,:,:] - bone[:C,i,:,:]
        data_new = [bone_motion[:C]]
    
    return data_new

def data_processing(data, graph, processing):
    C, T, V, M = data.shape # 3,300,25,2
    if 'mutual' in graph:
        
        if processing == 'default':
            mutual_data = np.zeros((C,T,V*2,1))
            mutual_data[:,:,:V,0] = data[:,:,:,0]
            mutual_data[:,:,V:,0] = data[:,:,:,1]
        
        elif processing == 'padding':
            mutual_data = np.zeros((C,T,V*2,2))
            mutual_data[:,:,:V,0] = data[:,:,:,0]
            mutual_data[:,:,V:,0] = data[:,:,:,1]
        
        elif processing == 'repeat':
            mutual_data = np.zeros((C,T,V*2,1))
            mutual_data[:,:,:V,0] = data[:,:,:,0]
            if data[:,:,:,1].sum(0).sum(0).sum(0) == 0:
                mutual_data[:,:,V:,0] = data[:,:,:,0]
            else:
                mutual_data[:,:,V:,0] = data[:,:,:,1]
        
        elif processing == 'symmetry':
            mutual_data = np.zeros((C,T,V*2,2))
            mutual_data[:,:,:V,0] = data[:,:,:,0]
            mutual_data[:,:,V:,0] = data[:,:,:,1]
            mutual_data[:,:,:V,1] = data[:,:,:,1]
            mutual_data[:,:,V:,1] = data[:,:,:,0]

        return mutual_data
        
    elif graph == 'physical':
        return data


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    # valid_frame: non-zero frames
    # p_interval: train:[0.5,1] eval:[0.95] 
    # window_size: 64
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data
