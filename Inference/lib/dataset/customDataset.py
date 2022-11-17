import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2

category_map = ('홍계', '배꼽', '피부손상F', '피부손상C', '피부손상S', '골절C', '가슴멍', '날개멍', '다리멍')

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
    def __init__(self, sourcePath, target:list, transform=None):
        self.sourcePath = sourcePath
        self.images = os.listdir(sourcePath)
        self.transform = transform
        self.target = target
        
    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.sourcePath, self.images[idx])
        img = cv2.imread(img_path)
        img = image_resize(img)
        label = self.images[idx].split('*')[:-1]
        
        label = self.get_multi_label(label)
        
        #  img should be PIL Image. Got <class 'numpy.ndarray'> : ndarray -> PIL
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.ToTensor()
            img = transform(img)
        
        return img, label, img_path
    
    def get_multi_label(self,label):
        result = np.zeros(len(self.target))
        for name in label:
            if name in self.target:
                result[self.target.index(name)] = 1.0
        return result
    