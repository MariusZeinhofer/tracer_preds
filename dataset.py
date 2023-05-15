# Here is the Dataset implementation. 

from torch.utils.data import Dataset
import pathlib
import numpy as np
from PIL import Image

# Question 1: pathlib.Path(<windows style path>) does not work in this implementation...
# Question 2: list(self.input_dir.iterdir()) seems bad. But PyTorch wants integer based iteration.
# Question 3: What does img.squeeze() do? A: Is a numpy method that removes axis of length 1
#
#
# Logic: Input and Target Data is in two folders. The only requirement for the 
# Dataset to work is that corresponding pairs (Input, Target) have the same name.
#  

def preprocess(input_path, target_path):
    # normalization
    inpt = (1./255.) * np.array(
        Image.open(input_path).convert("L"), 
        dtype = np.float32
        )
    
    # normalization
    target = (1./255.) * np.array(
        Image.open(target_path).convert("L"), 
        dtype = np.float32
        )
    return inpt, target


class MRIDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = pathlib.Path(input_dir)
        self.target_dir = pathlib.Path(target_dir)

        self.input_paths = list(self.input_dir.iterdir())    
        self.transform = transform
        
    def __len__(self):
        return len(self.input_paths)
        
    def __getitem__(self, index):
        input_path = self.input_paths[index]
        target_path = pathlib.Path.joinpath(self.target_dir, input_path.name)
        
        # the preprocessing pipeline, for us just a simple function call.
        # for complex situations it might be useful to do
        # this in a script of its own and safe the preprocessed data to file.
        inpt, target = preprocess(input_path, target_path)

        # data augmentation
        if self.transform != None:
            augmentations = self.transform(image=inpt, mask=target)
            inpt   = augmentations["image"]
            target = augmentations["mask"]
        
        return inpt, target
    
    def get_pat_name(self, index):
        return str(self.input_paths[index].stem)


# We provide some visualization of the dataset and the preprocessed data.
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = MRIDataset(input_dir='Data/Input', target_dir=pathlib.Path('Data/Target'))

    figure = plt.figure(figsize=(8, 8))
    rows, columns = 2, 4
    for i in range(1, columns + 1):
        
        img, label = dataset[i-1]
        figure.add_subplot(rows, columns, i)
        plt.title(dataset.get_pat_name(i-1) + ' Input')
        plt.axis("off")
        plt.imshow(img, cmap="gray")

        figure.add_subplot(rows, columns, i+columns)
        plt.title(dataset.get_pat_name(i-1) + ' Target')
        plt.axis("off")
        plt.imshow(label, cmap="gray")
    
    plt.show()