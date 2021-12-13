''' load cots dataset
    Note: input bboxes must be list format(not dict)
'''
from torch.utils.data import Dataset
import cv2

class Dataset(Dataset):
     
    def __init__(self, file_paths, bboxes, transform=None):
        self.file_paths = list(file_paths)
        self.bboxes = list(bboxes)
        self.transform = transform
         
    def __len__(self):
        return len(self.file_paths)
 
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        bboxes = self.bboxes[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image, bboxes = self.transform(image=image, bboxes=bboxes)
        return image, bboxes