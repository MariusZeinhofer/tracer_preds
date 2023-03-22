# Here is the Dataset implementation. 

from torch.utils.data import Dataset
import os
import numpy as np
import json
from PIL import Image

# idx is inter based index, is fed into __getitem__
#
# self.pat_dict(self.key_list[idx]) 
#
# gives back the dictionary that represents a patient

class MRIDataset(Dataset):
    def __init__(self, root_path, json_path, transform=None):
        # inter-numbered keys. Stupid but I don' know how to do better atm
        self.key_list  = [k for k in json.load(open(json_path))]
        # the dataset as a dict
        self.pat_dict  = json.load(open(json_path)) 
        self.root_path = root_path
        self.transform = transform
        
    def __len__(self):
        return len(self.pat_dict)

    def __getitem__(self, index):
        inpt_path = os.path.join(
            self.root_path, 
            self.pat_dict[self.key_list[index]]["input"]
            )
        target_path = os.path.join(
            self.root_path, 
            self.pat_dict[self.key_list[index]]["target"]
            )
        
        # normalization
        inpt = (1./255.) * np.array(
            Image.open(inpt_path).convert("L"), 
            dtype = np.float32
            )
        target = np.load(target_path)

        if self.transform != None:
            augmentations = self.transform(image=inpt, mask=target)
            inpt   = augmentations["image"]
            target = augmentations["mask"]
        
        return inpt, target
