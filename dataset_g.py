from torch.utils.data import Dataset
from PIL import Image
import csv
import torch


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset_G(Dataset):
    def __init__(self, csv_path, image_path, lbp_path, dataset='', data_transforms=None, target_transform=None,
                 loader=default_loader):

        imgs = []

        csvFile = open(csv_path, "r")
        reader = csv.reader(csvFile)

        for item in reader:
            if reader.line_num == 1:
                continue
            temp = 0
            if item[1] == '1.0':
                temp = 1
            if item[2] == '1.0':
                temp = 2
            imgs.append((image_path + "/" + item[0], reader.line_num - 1, temp))

        self.imgs = imgs
        self.lbp_path = lbp_path
        self.data_transforms = data_transforms
        self.target_transform = target_transform
        self.loader = loader
        self.dataset = dataset

    def __getitem__(self, index):
        fn, num, label = self.imgs[index]
        orin_img = self.loader(fn + ".jpg")
        img = orin_img.resize((224, 224))

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(fn))

        csvFile = open(self.lbp_path, "r")
        reader = csv.reader(csvFile)

        feature = []
        for row in reader:
            if reader.line_num == num:
                feature = [float(i) for i in row]
                feature = torch.Tensor(feature)
                break

        return img, feature, label

    def __len__(self):
        return len(self.imgs)
