from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch

class Text2ImgDataSet(Dataset):

    def __init__(self,txtDataFile,transform=None,split=0):
        fh = open(txtDataFile,'r')
        raw_samples = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(' ')
            labels = words[2].split(',')
            embed = []
            for label in labels:
                embed.append(int(label))
            raw_samples.append((words[0],words[1],embed))
        self.txtDataFile = txtDataFile
        self.transform = transform
        self.raw_samples = raw_samples
    def __getitem__(self,index):
        right_image_path, wrong_image_path, right_embed = self.raw_samples[index]
        #right_image = Image.open(right_image_path).resize((64, 64))
        #wrong_image = Image.open(wrong_image_path).resize((64, 64))
        right_image = Image.open(right_image_path).convert("RGB")
        wrong_image = Image.open(wrong_image_path).convert("RGB")

        right_image = self.validate_image(right_image)
        wrong_image = self.validate_image(wrong_image)

        right_embed = np.array(right_embed, dtype=float)
        right_image = np.array(right_image, dtype=float)
        wrong_image = np.array(wrong_image, dtype=float)
        #right_embed = torch.from_numpy(right_embed)
        #right_image = torch.from_numpy(right_image)
        #wrong_image = torch.from_numpy(wrong_image)
        sample = {
            'right_images': torch.FloatTensor(right_image),
            'right_embed': torch.FloatTensor(right_embed),
            'wrong_images': torch.FloatTensor(wrong_image)
        }
        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] = sample['wrong_images'].sub_(127.5).div_(127.5)
        return sample
    def __len__(self):
        return len(self.raw_samples)

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb
        return img.transpose(2, 0, 1)