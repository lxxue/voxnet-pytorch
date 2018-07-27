import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import sys
import os
import re

sys.path.append(os.path.join(".."))
#from utils import npytar
from utils import npytar

class fewModelNet(Dataset):
    def __init__(self, filedir, filename):
        reader = npytar.NpyTarReader(os.path.join(filedir, filename))
        temp_list = []
        for (arr, name) in reader:
            temp_list.append((arr, name))
        reader.close()
        # in this dataset we have 12 views for each shape
        if len(temp_list) % 12 != 0:
            # assert is a statement in python2/3, not a function
            assert "some shapes might not have 12 views"
        # pick all the examples 
        # b x v x c x d x d x d
        temp_data = np.zeros((len(temp_list)//12, 12, 1, 32, 32, 32), dtype=np.float32)
        # all view share the same label
        temp_label = np.zeros((len(temp_list)//12,), dtype=np.int)

        # sort the file by its name
        # format: classnum.classname_idx.viewidx
        # exception: 001.2h31k8fak.viewidx
        temp_list.sort(key=lambda x: (int(x[1].split(".")[0]), x[1].split(".")[-2].split("_")[-1], int(x[1].split(".")[-1])))
        for idx, (arr, name) in enumerate(temp_list):
            temp_data[idx//12, idx%12, 0] = arr
            if idx % 12 == 0:
                # assign label
                # name: class_idx.fname.view_idx
                temp_label[idx//12] = int(name.split('.')[0])-1
            else:
                # check label consistency
                assert temp_label[idx//12]==(int(name.split('.')[0])-1), "label is inconsistent among different views for file {}, original label{}".format(name, temp_label[idx//12])
        #finish loading all data
        # select 9 shapes for each examples
        # quite a stupid method since we can get 90 directly
        self.data = np.zeros((90, 12, 1, 32, 32, 32), dtype=np.float32)
        self.label = np.zeros((90,), dtype=np.int)
        
        for i in range(10):
            index = (temp_label == i)
            temp_class_len = sum(index)
            #print(temp_class_len)
            temp_class_data = temp_data[index]
            rand_idx = np.random.choice(temp_class_len, 9, replace=False) 
            #print(rand_idx)
            self.data[i*9:i*9+9] = temp_class_data[rand_idx]
            self.label[i*9:i*9+9] = i

    
    def __len__(self):
        return self.label.shape[0]*12

    def __getitem__(self, idx):
        return self.data[idx//12, idx%12], self.label[idx//12]


if __name__ == '__main__':
    modelnet30 = fewModelNet("../data", "shapenet10_test.tar")
    print(len(modelnet30))
    print(modelnet30[80][0].shape, modelnet30[80][1])
    print(np.mean(modelnet30[80][0][np.where(modelnet30[80][0]!=0)]))
