import pickle, logging, numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NTU_Feeder(Dataset):
    def __init__(self, phase, dataset, split, root_folder, inputs, num_frame, connect_joint, debug, graph, random_swap=False, processing='default', crop=False, transform=False, **kwargs):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        self.graph = graph
        self.processing = processing
        self.phase = phase
        self.crop = crop

        self.datashape = self.set_datashape()
        
        if transform:
            dataset_path = os.path.join(root_folder,'transformed','-'.join([dataset,split]))
        else:
            dataset_path = os.path.join(root_folder,'-'.join([dataset,split]))
        data_path = '{}/{}_data.npy'.format(dataset_path, phase)
        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)
        try:
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.name, self.label, self.seq_len = pickle.load(f, encoding='latin1')
            if random_swap:
                data = np.zeros_like(self.data)
                for i in range(len(self.label)):
                    if np.random.randint(2) == 1:
                        data[i,:,:,:,[0,1]] = self.data[i,:,:,:,[1,0]]
                    else:
                        data[i] = self.data[i]
                self.data = data
        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.name = self.name[:300]
            self.seq_len = self.seq_len[:300]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.name[idx]
        seq_len = self.seq_len[idx]

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        C = data.shape[0]
        if self.crop:
            if self.phase == 'train':
                data = self.valid_crop_resize(data, valid_frame_num=seq_len, p_interval=[0.5,1], window=64)
            elif self.phase == 'eval':
                data = self.valid_crop_resize(data, valid_frame_num=seq_len, p_interval=[0.95], window=64)
        data = self.data_processing(data)
        joint, joint_motion, bone, bone_motion = self.multi_input(data[:,:self.T,:,:])
        data_new = []
        if self.inputs.isupper():
            if 'J' in self.inputs:
                data_new.append(joint)
            if 'V' in self.inputs:
                data_new.append(joint_motion)
            if 'B' in self.inputs:
                data_new.append(bone)
            if 'M' in self.inputs:
                data_new.append(bone_motion)
        elif self.inputs == 'joint':
            data_new = [joint[:C,:,:,:]]
        elif self.inputs == 'bone':
            data_new = [bone[:C,:,:,:]]
        elif self.inputs == 'joint_motion':
            data_new = [joint_motion[:C,:,:,:]]
        elif self.inputs == 'bone_motion':
            data_new = [bone_motion[:C,:,:,:]]
        else:
            logging.info('')
            logging.error('Error: No input feature!')
            raise ValueError()
        data_new = np.stack(data_new, axis=0)
        assert list(data_new.shape) == self.datashape
        return data_new, label, name

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        joint_motion = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        bone_motion = np.zeros((C*2, T, V, M))
        # joint
        joint[:C,:,:,:] = data
        for i in range(V):
            joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
        # joint_motion
        joint_motion[C:,0] = joint_motion[:C,0] = data[:,0]
        for i in range(1,T):
            joint_motion[:C,i,:,:] = data[:,i,:,:] - data[:,i-1,:,:]
            joint_motion[C:,i,:,:] = joint_motion[:C,i,:,:] - joint_motion[:C,i-1,:,:]
        # bone
        for i in range(len(self.conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
        # bone_motion
        bone_motion[C:,0] = bone_motion[:C,0] = bone[:C,0]
        for i in range(1,T):
            bone_motion[:C,i,:,:] = bone[:C,i,:,:] - bone[:C,i-1,:,:]
            bone_motion[C:,i,:,:] = bone_motion[:C,i,:,:] - bone[:C,i-1,:,:]
        return joint, joint_motion, bone, bone_motion

    def data_processing(self, data):
        C, T, V, M = data.shape # 3,300,25,2
        if 'mutual' in self.graph:
            
            if self.processing == 'default':
                mutual_data = np.zeros((C,T,V*2,1))
                mutual_data[:,:,:V,0] = data[:,:,:,0]
                mutual_data[:,:,V:,0] = data[:,:,:,1]
            
            elif self.processing == 'padding':
                mutual_data = np.zeros((C,T,V*2,2))
                mutual_data[:,:,:V,0] = data[:,:,:,0]
                mutual_data[:,:,V:,0] = data[:,:,:,1]
            
            elif self.processing == 'repeat':
                mutual_data = np.zeros((C,T,V*2,1))
                mutual_data[:,:,:V,0] = data[:,:,:,0]
                if data[:,:,:,1].sum(0).sum(0).sum(0) == 0:
                    mutual_data[:,:,V:,0] = data[:,:,:,0]
                else:
                    mutual_data[:,:,V:,0] = data[:,:,:,1]
            
            elif self.processing == 'symmetry':
                mutual_data = np.zeros((C,T,V*2,2))
                mutual_data[:,:,:V,0] = data[:,:,:,0]
                mutual_data[:,:,V:,0] = data[:,:,:,1]
                mutual_data[:,:,:V,1] = data[:,:,:,1]
                mutual_data[:,:,V:,1] = data[:,:,:,0]
    
            else:
                logging.info('')
                logging.error('Error: Wrong in loading processing configs')
                raise ValueError()
            return mutual_data
        elif self.graph == 'physical':
            return data
        else:
            logging.info('')
            logging.error('Error: Wrong in loading processing configs')
            raise ValueError()

    def valid_crop_resize(self, data_numpy, valid_frame_num, p_interval, window):
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

    def set_datashape(self):
        data_shape = [3,6,300,25,2]
        data_shape[0] = len(self.inputs) if self.inputs.isupper() else 1
        data_shape[1] = 3 if self.inputs in ['joint','joint_motion','bone','bone_motion'] else 6
        data_shape[2] = 64 if self.crop else self.T
        if 'mutual' in self.graph:
            data_shape[3] = data_shape[3]*data_shape[4]
            data_shape[4] = 1
        if self.processing in ['symmetry','padding']: 
            assert data_shape[4] == 1
            data_shape[4] = data_shape[4]*2
        return data_shape
        # if not inputs.isupper():
        #     data_shape = data_shape[1:]

class NTU_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 2, self.T, self.V, self.M))
        for i, name in enumerate(names):
            with open(name, 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        fr.readline()
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')
                            if joint < self.V and person < self.M:
                                location[i,0,frame,joint,person] = float(v[5])
                                location[i,1,frame,joint,person] = float(v[6])
        return location
