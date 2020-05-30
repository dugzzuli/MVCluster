import numpy as np
import linecache
from Utils.utils import *
import os
class Dataset(object):

    def __init__(self, config):
        self.graph_file = config['graph_file']
        self.label_file = config['label_file']
        self.W, self.Y = self._load_data()

        self.num_nodes = self.W.shape[0]
        self.num_classes = self.Y.shape[1]
        self.num_edges = np.sum(self.W) / 2
        print('nodes {}, edges {}, classes {}'.format(self.num_nodes, self.num_edges, self.num_classes))


        self._order = np.arange(self.num_nodes)
        self._index_in_epoch = 0
        self.is_epoch_end = False

    def _load_data(self):
        lines = linecache.getlines(self.label_file)
        lines = [line.rstrip('\n') for line in lines]

        #===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[0]] = idx
            y = []
            for label in line[1:]:
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y)
        num_classes = len(label_map)
        num_nodes = len(node_map)

        L = np.zeros((num_nodes, num_classes), dtype=np.int32)
        for idx, y in enumerate(Y):
            L[idx][y] = 1




        #==========load graph========
        W = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.graph_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            W[idx2, idx1] = 1.0
            W[idx1, idx2] = 1.0



        return W, L
    def transform(self,path):

        with open(path+"/View1.txt", mode='w') as f:
            for i in self._order:
                f.write(str(i))
                for j in self._order:
                    f.write(" "+str(self.W[i][j]))
                f.flush()
                f.write("\n")

if __name__=='__main__':
    path='pubmed'
    dataset_config = { 'label_file': './Database/'+path+'/group.txt',
                      'graph_file': './Database/'+path+'/edges.txt'}
    graph = Dataset(dataset_config)
    graph.transform(path)

