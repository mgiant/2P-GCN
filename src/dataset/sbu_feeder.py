import os, pickle, logging, numpy as np
from torch.utils.data import Dataset

class SBU_Feeder(Dataset):
    def __init__(self, phase, fold, root_folder, inputs, num_frame, connect_joint, debug, graph, processing='default', **kwargs):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        self.graph = graph
        self.processing = processing
        self.datashape = self.set_datashape()

        folds = {'train':list({1,2,3,4,5}-{fold}), 'eval':[fold]}

        for i in folds[phase]:
            data_path = '{}/fold{}_data.npy'.format(root_folder,i)
            if os.path.exists(data_path):
                fold = np.load(data_path)
                if i == folds[phase][0]:
                    self.data = fold
                else:
                    self.data = np.concatenate((self.data,fold),axis=0)
            else:
                logging.info('')
                logging.error('Error: Do NOT exist data files: {}!'.format(data_path))
                logging.info('Please generate data first!')
                raise ValueError()

            label_path = '{}/fold{}_label.pkl'.format(root_folder,i)
            if os.path.exists(label_path):
                with open(label_path, 'rb') as f:
                    label = pickle.load(f, encoding='latin1')
                    if i == folds[phase][0]:
                        self.label = label
                    else:
                        self.label = np.concatenate((self.label,label))
            else:
                logging.info('')
                logging.error('Error: Do NOT exist data files: {}!'.format(label_path))
                logging.info('Please generate data first!')
                raise ValueError()

        if debug:
            logging.info(self.data.shape)
            logging.info(self.label.shape)
    
    def __len__(self):
        if len(self.data) == len(self.label):
            return len(self.data)
        else:
            raise ValueError()
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        data = self.data_processing(data)
        joint, motion, bone, bone_motion = self.multi_input(data[:,:self.T,:,:])
        data_new = []
        if self.inputs.isupper():
            if 'J' in self.inputs:
                data_new.append(joint)
            if 'V' in self.inputs:
                data_new.append(motion)
            if 'B' in self.inputs:
                data_new.append(bone)
            if 'M' in self.inputs:
                data_new.append(bone_motion)
        elif self.inputs == 'joint':
            data_new = [joint[:C,:,:,:]]
        elif self.inputs == 'bone':
            data_new = [bone[:C,:,:,:]]
        elif self.inputs == 'motion':
            data_new = [motion[:C,:,:,:]]
        elif self.inputs == 'bone_motion':
            data_new = [bone_motion[:C,:,:,:]]
        else:
            logging.info('')
            logging.error('Error: No input feature!')
            raise ValueError()
        data_new = np.stack(data_new, axis=0)
        assert list(data_new.shape) == self.datashape
        return data_new, label, ""

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        bone_motion = np.zeros((C*2, T, V, M))
        joint[:C,:,:,:] = data
        for i in range(V):
            joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
        for i in range(T-2):
            velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
            velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
        for i in range(len(self.conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        for i in range(T-2):
            bone_motion[:C,i,:,:] = bone[:C,i+1,:,:] - bone[:C,i,:,:]
            bone_motion[C:,i,:,:] = bone[:C,i+2,:,:] - bone[:C,i,:,:]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
        return joint, velocity, bone, bone_motion

    def data_processing(self, data):
        C, T, V, M = data.shape # 3,100,15,2
        if 'mutual' in self.graph:
            
            if self.processing == 'default':
                mutual_data = np.zeros((C,T,V*2,1))
                mutual_data[:,:,:V,0] = data[:,:,:,0]
                mutual_data[:,:,V:,0] = data[:,:,:,1]
            
            elif self.processing == 'padding':
                mutual_data = np.zeros((C,T,V*2,2))
                mutual_data[:,:,:V,0] = data[:,:,:,0]
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
        
    def set_datashape(self):
        data_shape = [3,6,100,15,2]
        data_shape[0] = len(self.inputs) if self.inputs.isupper() else 1
        data_shape[1] = 3 if self.inputs in ['joint','motion','bone'] else 6
        data_shape[2] = self.T
        if 'mutual' in self.graph:
            data_shape[3] = data_shape[3]*data_shape[4]
            data_shape[4] = 1
        if self.processing in ['symmetry','padding']: 
            assert data_shape[4] == 1
            data_shape[4] = data_shape[4]*2
        return data_shape
        # if not inputs.isupper():
        #     data_shape = data_shape[1:]