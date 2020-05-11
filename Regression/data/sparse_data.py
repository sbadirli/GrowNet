import time
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file

class LibSVMDataSp(Dataset):
    def __init__(self, root, dim_in, pos=1, neg=-1):
        self.feat, self.label = load_svmlight_file(root)
        self.feat = csr_matrix((self.feat.data, self.feat.indices, self.feat.indptr), shape=(len(self.label), dim_in))
        self.feat = self.feat.astype(np.float32)
        self.label = self.label.astype(np.float32)
        self.label[self.label == pos] = 1
        self.label[self.label == neg] = -1

    def __getitem__(self, index):
        arr = self.feat[index, :]
        return arr, self.label[index]
    def __len__(self):
        return len(self.label)


