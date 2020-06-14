import time
import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn import datasets
from sklearn.model_selection import train_test_split


class LibSVMData(Dataset):
    def __init__(self, root, dim, normalization, pos=1, neg=-1, out_pos=1, out_neg=-1):
        self.feat, self.label = load_svmlight_file(root)

        self.feat = csr_matrix((self.feat.data, self.feat.indices, self.feat.indptr), shape=(len(self.label), dim))
        self.feat = self.feat.toarray().astype(np.float32)

        self.label = self.label.astype(np.float32)
        idx_pos = self.label == pos
        idx_neg = self.label == neg
        self.label[idx_pos] = out_pos
        self.label[idx_neg] = out_neg

    def __getitem__(self, index):
        arr = self.feat[index, :]
        return arr, self.label[index]
    def __len__(self):
        return len(self.label)

class LibSVMRankData(Dataset):
    def __init__(self, root2data, root2qid, dim):
        self.feat, self.label = load_svmlight_file(root2data)
        self.qid = np.loadtxt(root2qid, dtype='int32')
        self.feat = self.feat.toarray().astype(np.float32)
        self.label = self.label.astype(np.float32)
        self.feat = self.feat[:, ~(self.feat == 0).all(0)]
        print(self.feat.shape[1])

    def __getitem__(self, index):
        return self.feat[index, :], self.label[index], self.qid[index]

    def __len__(self):
        return len(self.label)

class LibSVMRegData(Dataset):
    def __init__(self, root, dim, normalization):
        data = np.load(root)        
        self.feat, self.label = data['features'], data['labels']
        del data
        self.feat = self.feat.astype(np.float32)
        self.label = self.label.astype(np.float32)
        #self.feat = self.feat[:, ~(self.feat == 0).all(0)]
        #import ipdb; ipdb.set_trace()

        print(self.feat.shape[1])

    def __getitem__(self, index):
        return self.feat[index, :], self.label[index]

    def __len__(self):
        return len(self.label)

class LibCSVData(Dataset):
    def __init__(self, root, dim, pos=1, neg=-1):
        self.data = np.loadtxt(root, delimiter=',').astype(np.float32)
        self.feat = self.data[:, 1:]
        self.label = self.data[:, 0]
        self.label[self.label == pos] = 1
        self.label[self.label == neg] = -1

    def __getitem__(self, index):
        #arr = np.log(self.feat[index, :] + 1.0e-5)
        #arr = np.log10(self.feat[index, :] + 1.0e-5)
        arr = self.feat[index, :]
        return arr, self.label[index]

    def __len__(self):
        return len(self.label)
class CriteoCSVData(Dataset):
    def __init__(self, root, dim, normalization, pos=1, neg=-1):
        # Reading the data into panda data frame
        self.data = pd.read_csv(root, header=None, dtype='float32')
        # extracting labels (0, 1) and weights
        self.label = self.data.iloc[:, -2]
        self.weights = self.data.iloc[:, -1]
        self.data = self.data.iloc[:, :-2]
        # transferring labels from {0, 1} to {-1, 1}
        self.label[self.label == pos] = 1
        self.label[self.label == neg] = -1

        # Applying log transformation
        mm = self.data.min().min() # to prevent 0 division
        if normalization:
            # Filling Nan values: Simple approach, mean of the that column or interpolation
            self.data = self.data.transform(lambda x: np.log(x - mm + 1))
            #self.data = self.data.interpolate(method='polynomial', order=2)
            self.data = self.data.fillna(self.data.mean()) # To fill the rest of Nan values left untouched on the corners
            #self.data = (self.data - self.data.mean())/self.data.std()
        #self.feat = self.data.to_numpy('float32')
        self.data = self.data.to_numpy('float32')
    def __getitem__(self, index):
        #arr = np.log(self.feat[index, :] + 1.0e-5)
        #arr = np.log10(self.feat[index, :] + 1.0e-5)
        #arr = self.feat[index, :]
        arr = self.data[index, :]
        return arr, self.label[index], self.weights[index]

    def __len__(self):
        return len(self.label)
