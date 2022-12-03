import os, logging, numpy as np


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph():
    def __init__(self, dataset, graph, labeling, num_person_out=1, max_hop=10, dilation=1, normalize=True, threshold=0.2, **kwargs):
        self.dataset = dataset
        self.labeling = labeling
        self.graph = graph
        if labeling not in ['spatial','distance','zeros','ones','eye','pairwise0','pairwise1','geometric']:
            logging.info('')
            logging.error('Error: Do NOT exist this graph labeling: {}!'.format(self.labeling))
            raise ValueError()
        self.normalize = normalize
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_person_out = num_person_out
        self.threshold = threshold

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts, self.center = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'kinetics':
            num_node = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14), (8, 11)]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),              # left_arm
                np.array([2, 3, 4]),              # right_arm
                np.array([11, 12, 13]),           # left_leg
                np.array([8, 9, 10]),             # right_leg
                np.array([0, 1, 14, 15, 16, 17])  # torso
            ]
            center = 1
        elif self.dataset in ['ntu','ntu120','ntu_mutual','ntu120_mutual','ntu_original']:
            if self.graph == 'physical':
                num_node = 25
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                (22, 23), (23, 8), (24, 25), (25, 12)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
                parts = [
                    np.array([5, 6, 7, 8, 22, 23]) - 1,     # left_arm
                    np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                    np.array([13, 14, 15, 16]) - 1,         # left_leg
                    np.array([17, 18, 19, 20]) - 1,         # right_leg
                    np.array([1, 2, 3, 4, 21]) - 1          # torso
                ]
                center = 21 - 1
            elif self.graph == 'mutual':
                num_node = 50
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                (22, 23), (23, 8), (24, 25), (25, 12)] + \
                                 [(26, 27), (27, 46), (28, 46), (29, 28), (30, 46),
                                (31, 30), (32, 31), (33, 32), (34, 46), (35, 34),
                                (36, 35), (37, 36), (38, 26), (39, 38), (40, 39),
                                (41, 40), (42, 26), (43, 42), (44, 43), (45, 44),
                                (47, 48), (48, 33), (49, 50), (50, 37)] + \
                                 [(21, 46)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array([1, 1, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 22, 7, 24, 11, 26, 26, 45, 27, 45, 29, 30, 31, 45, 33, 34, 35, 25, 37, 38, 39, 25, 41, 42, 43, 26, 47, 32, 49, 36])
                parts = [
                    # left_arm
                    np.array([5, 6, 7, 8, 22, 23]) - 1,
                    np.array([5, 6, 7, 8, 22, 23]) + 25 - 1,
                    # right_arm
                    np.array([9, 10, 11, 12, 24, 25]) - 1,
                    np.array([9, 10, 11, 12, 24, 25]) + 25 - 1,
                    # left_leg
                    np.array([13, 14, 15, 16]) - 1,
                    np.array([13, 14, 15, 16]) + 25 - 1,
                    # right_leg
                    np.array([17, 18, 19, 20]) - 1,
                    np.array([17, 18, 19, 20]) + 25 - 1,
                    # torso
                    np.array([1, 2, 3, 4, 21]) - 1,
                    np.array([1, 2, 3, 4, 21]) + 25 - 1
                ]
                center = 21 - 1
            elif self.graph == 'mutual-inter':
                num_node = 50
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                (22, 23), (23, 8), (24, 25), (25, 12)] + \
                                 [(26, 27), (27, 46), (28, 46), (29, 28), (30, 46),
                                (31, 30), (32, 31), (33, 32), (34, 46), (35, 34),
                                (36, 35), (37, 36), (38, 26), (39, 38), (40, 39),
                                (41, 40), (42, 26), (43, 42), (44, 43), (45, 44),
                                (47, 48), (48, 33), (49, 50), (50, 37)] + \
                                 [(21, 46)] + \
                                [(23,25), (48,50), (23,48), (25,50)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array([1, 1, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 22, 7, 24, 11, 26, 26, 45, 27, 45, 29, 30, 31, 45, 33, 34, 35, 25, 37, 38, 39, 25, 41, 42, 43, 26, 47, 32, 49, 36]) 
                parts = [
                    # left_arm
                    np.array([5, 6, 7, 8, 22, 23]) - 1,
                    np.array([5, 6, 7, 8, 22, 23]) + 25 - 1,
                    # right_arm
                    np.array([9, 10, 11, 12, 24, 25]) - 1,
                    np.array([9, 10, 11, 12, 24, 25]) + 25 - 1,
                    # left_leg
                    np.array([13, 14, 15, 16]) - 1,
                    np.array([13, 14, 15, 16]) + 25 - 1,
                    # right_leg
                    np.array([17, 18, 19, 20]) - 1,
                    np.array([17, 18, 19, 20]) + 25 - 1,
                    # torso
                    np.array([1, 2, 3, 4, 21]) - 1,
                    np.array([1, 2, 3, 4, 21]) + 25 - 1
                ]
                center = 21 - 1              
        elif self.dataset == 'sbu':
            if self.graph == 'physical':
                num_node = 15
                neighbor_1base = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7),
                                (7, 8), (8, 9), (3, 10), (10, 11), (11, 12),
                                (3, 13), (13, 14), (14, 15)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array([2,3,3,3,4,5,3,7,8,3,10,11,3,13,14]) - 1
                parts = [
                    # left_arm
                    np.array([4,5,6]) - 1,
                    # right_arm
                    np.array([7,8,9]) - 1,
                    # left_leg
                    np.array([10,11,12]) - 1,
                    # right_leg
                    np.array([13,14,15]) - 1,
                    # torso
                    np.array([1,2,3]) - 1,
                ]
                center = 3 - 1
            elif self.graph == 'mutual':
                num_node = 30
                neighbor_1base = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7),
                                (7, 8), (8, 9), (3, 10), (10, 11), (11, 12),
                                (3, 13), (13, 14), (14, 15)]
                neighbor_1base += [(i+15,j+15) for (i,j) in neighbor_1base] + [(2,2+15)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array([2,3,3,3,4,5,3,7,8,3,10,11,3,13,14]) - 1
                parts = [
                    # left_arm
                    np.array([4,5,6]) - 1,
                    np.array([4,5,6]) + 15 - 1,
                    # right_arm
                    np.array([7,8,9]) - 1,
                    np.array([7,8,9]) + 15 - 1,
                    # left_leg
                    np.array([10,11,12]) - 1,
                    np.array([10,11,12]) + 15 - 1,
                    # right_leg
                    np.array([13,14,15]) - 1,
                    np.array([13,14,15]) + 15 - 1,
                    # torso
                    np.array([1,2,3]) - 1,
                    np.array([1,2,3]) + 15 - 1,
                ]
                center = 3 - 1
        elif self.dataset == 'volleyball':
            num_node = 25
            neighbor_link = [  (0,1),(0,15),(0,16),(15,17),(16,18),
                                (1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
                                (1,8),(8,9),(9,10),(10,11),(11,24),(11,22),(22,23),
                                (8,12),(12,13),(13,14),(14,21),(14,19),(19,20)
                            ]
            connect_joint = np.array([1,1,1,2,3,1,5,6,1,8,9,10,8,12,13,0,0,15,16,14,19,14,11,22,11])
            parts = [
                np.array([5, 6, 7]),                 # left_arm
                np.array([2, 3, 4]),                 # right_arm
                np.array([9, 10, 11, 22, 23, 24]),   # left_leg
                np.array([12, 13, 14, 19, 20, 21]),  # right_leg
                np.array([0, 1, 8, 15, 16, 17, 18])  # torso
            ]
            center = 1
            if self.graph == 'multi-person':
                neighbor_link_nperson = []
                connect_joint_nperson = []
                parts_nperson = []
                for i in range(self.num_person_out):
                    for x in connect_joint:
                        connect_joint_nperson.append( x+i*num_node )
                    for x,y in neighbor_link:
                        neighbor_link_nperson.append((x+i*num_node,y+i*num_node))
                    for p in range(len(parts)):
                        parts_nperson.append(parts[p]+i*num_node)
                num_node *= self.num_person_out

                neighbor_link = neighbor_link_nperson
                connect_joint = connect_joint_nperson
                parts = parts_nperson

        else:
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts, center

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        self.oA = A
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):

        if self.labeling == 'distance':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'spatial':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if hop_dis[j, i] == hop:
                            # if hop_dis[j, self.center] == np.inf or hop_dis[i, self.center] == np.inf:
                            #     continue
                            if hop_dis[j, self.center] == hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif hop_dis[j, self.center] > hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
        
        elif self.labeling == 'zeros':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop),self.num_node,self.num_node))
        
        elif self.labeling == 'ones':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.ones((len(valid_hop),self.num_node,self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(A[i])
            
        
        elif self.labeling == 'eye':

            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop),self.num_node,self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(np.eye(self.num_node,self.num_node))
        
        elif self.labeling == 'pairwise0':
            # pairwise0: only pairwise inter-body link             
            assert 'mutual' in self.graph 

            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            v = self.num_node//2
            for i in range(len(valid_hop)):
                A[i,v:,:v] = np.eye(v,v)
                A[i,:v,v:] = np.eye(v,v)
                A[i] = self._normalize_digraph(A[i])


        elif self.labeling == 'pairwise1':
            assert 'mutual' in self.graph
            v = self.num_node//2
            self.edge += [(i,i+v) for i in range(v)]
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'geometric':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            
            geometric_matrix = np.load(os.path.join(os.getcwd(),'src/dataset/a.npy'))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if geometric_matrix[i,j] > self.threshold:
                        adjacency[i,j] += geometric_matrix[i,j]
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
    
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
