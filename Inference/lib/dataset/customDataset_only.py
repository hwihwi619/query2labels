from PIL import Image
import numpy as np
import os
import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import cv2
import torchvision
# category_map = {'정상':0, '홍계':1, '배꼽':2, '피부손상F':3, '피부손상C':4, '피부손상S':5, '골절C':6, '가슴멍':7, '날개멍':8, '다리멍':9}
################ 0      1        2          3          4          5        6       7       8
category_map = ('홍계', '배꼽', '피부손상F', '피부손상C', '피부손상S', '골절C', '가슴멍', '날개멍', '다리멍')
target_map = ['가슴멍']

def image_resize(img):
    output_size = 448
    (h, w) = img.shape[:2]
    shape = img.shape[:2]  # current shape [height, width]
    r = output_size / max(h, w)
    if r != 1:
        img = cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_LINEAR)

    # Scale ratio (new / old)
    r = min(output_size / h, output_size / w)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = output_size - new_unpad[0], output_size - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))  # add border

    return img

class CustomDataset_csv_multiLabel(Dataset): 
    def __init__(self, sourcePath, transform=None):
        self.sourcePath = sourcePath
        self.images = os.listdir(sourcePath)
        self.transform = transform

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.images)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        img_path = os.path.join(self.sourcePath, self.images[idx])
        img = cv2.imread(img_path)
        label = self.images[idx].split('*')[:-1]
        
        label = self.get_multi_label(label)
        
        #  img should be PIL Image. Got <class 'numpy.ndarray'> : ndarray -> PIL
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.ToTensor()
            img = transform(img)

        # if target_map[0] in img_path:
        #     print()
        #     pass
        
        return img, label
    
    def get_multi_label(self,label):
        result = np.zeros(len(target_map))
        for name in label:
            if name in target_map:
                result[target_map.index(name)] = 1.0
        return result
    
if __name__=='__main__':
    # ig = '/home/hwi/Downloads/VQIS-POC train dataset/VQIS_FOR_TRAIN original/1_000000_B_SAM.JPG'
    ig = '/home/hwi/github/dataset/Chess Pieces.v24-416x416_aug.multiclass/train/IMG_0318_JPG.rf.60de6e8f2c9365770ed11379dfd1cb55.jpg'
    img = cv2.imread(ig, cv2.IMREAD_COLOR)
    img = image_resize(img)

    cv2.imshow('img', img)
    cv2.waitKey(0)