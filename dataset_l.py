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
        img_temp = list()
        labels = []
        orin_img1 = self.loader(fn + "1.jpg")
        img1 = orin_img1.resize((224, 224))

        orin_img2 = self.loader(fn + "2.jpg")
        img2 = orin_img2.resize((224, 224))

        orin_img3 = self.loader(fn + "3.jpg")
        img3 = orin_img3.resize((224, 224))

        orin_img4 = self.loader(fn + "4.jpg")
        img4 = orin_img4.resize((224, 224))

        for i in range(4):
            labels.append(label)

        if self.data_transforms is not None:
            try:
                img1 = self.data_transforms[self.dataset](img1)
                img2 = self.data_transforms[self.dataset](img2)
                img3 = self.data_transforms[self.dataset](img3)
                img4 = self.data_transforms[self.dataset](img4)
            except:
                print("Cannot transform image: {}".format(fn))
        img_temp.append(img1)
        img_temp.append(img2)
        img_temp.append(img3)
        img_temp.append(img4)
        csvFile = open(self.lbp_path, "r")
        reader = csv.reader(csvFile)

        feature = []

        for row in reader:
            if reader.line_num == ((num - 1) * 4 + 1):
                fea = [float(i) for i in row]
                fea = torch.Tensor(fea)
                feature.append(fea)

            if reader.line_num == ((num - 1) * 4 + 2):
                fea = [float(i) for i in row]
                fea = torch.Tensor(fea)
                feature.append(fea)

            if reader.line_num == ((num - 1) * 4 + 3):
                fea = [float(i) for i in row]
                fea = torch.Tensor(fea)
                feature.append(fea)

            if reader.line_num == ((num - 1) * 4 + 4):
                fea = [float(i) for i in row]
                fea = torch.Tensor(fea)
                feature.append(fea)
        return img_temp, feature, labels

    def __len__(self):
        return len(self.imgs)
