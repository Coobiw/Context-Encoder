
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg
import random
import matplotlib.pyplot as plt
import math

class MyDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）

        img = Image.open(img).convert('L')
        label = Image.open(label).convert('L')



        img, label = self.center_crop(img, label, self.crop_size)

        img, label ,mask , weight = self.img_transform(img, label)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        sample = {'img': img, 'label': label ,'mask' :mask , 'weight' : weight}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        label = np.array(label)  # 以免不是np格式的数据
        label = Image.fromarray(label.astype('uint8'))
        img = img.resize((128 ,128))



        transform_img = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        transform_label = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        img = np.array(img)
        random.seed(729608)
        p1 = random.choice([0,0,0,0,1])
        p2 = random.choice([0,0,0,0,0,0,0,0,0,1])
        poly = random.uniform(0.2,1)

        # print(p1)
        # print(p2)
        # print(poly)

        if p1 == 1:
            img = img * poly
            img = img.astype('uint8')
        if p2 == 1:
            img = 255-img
        img = Image.fromarray(img)
        img = transform_img(img)
        label = img.clone()
        random_erase = transforms.RandomErasing(p=1, scale=(0.05, 0.5) ,ratio=(0.33 ,3), value=1)
        img = random_erase(img)
        mask = (img==1)
        normalization = transforms.Normalize([0.5] ,[0.5])
        img = normalization(img)
        label = normalization(label)

        mask_tensor = mask.clone()
        mask_numpy = mask_tensor.numpy()
        weight = np.zeros((128,128))
        for ih in range(4,124): #为什么右侧不是128，因为不能取图像的四个边界
            for iw in range(4,124):
                if mask_numpy[0,ih,iw]:
                    if (mask_numpy[0,ih-4,iw]) and (mask_numpy[0,ih+4,iw]) and (mask_numpy[0,ih,iw+4]) and (mask_numpy[0,ih,iw-4]):
                        weight[ih,iw] = 1
                    else:
                        weight[ih,iw]= 10

        weight = transforms.ToTensor()(weight)
        # print(weight)


        return img, label ,mask,weight



if __name__ == "__main__":

    TRAIN_ROOT = cfg.TRAIN_ROOT
    TRAIN_LABEL = cfg.TRAIN_LABEL
    # VAL_ROOT = cfg.VAL_ROOT
    # VAL_LABEL = cfg.VAL_LABEL
    # TEST_ROOT = cfg.TEST_ROOT
    # TEST_LABEL = cfg.TEST_LABEL
    crop_size = cfg.crop_size
    Cam_train = MyDataset([TRAIN_ROOT, TRAIN_LABEL], crop_size)
    # Cam_val = MyDataset([VAL_ROOT, VAL_LABEL], crop_size)
    # Cam_test = MyDataset([TEST_ROOT, TEST_LABEL], crop_size)
    A = Cam_train[0]
    B = Cam_train[1]
    C = Cam_train[2]
    a1 = A['img']
    a2 = A['label']
    a3 = A['mask']
    a_w = A['weight']
    a4 = B['img']
    a5 = B['label']
    a6 = B['mask']
    a1 = ((a1 *0.5 ) +0.5) * 255
    a2 = ((a2 *0.5 ) +0.5 )* 255
    a3 = a3 * 255
    a4 = ((a4 *0.5 ) +0.5) * 255
    a5 = ((a5 *0.5 ) +0.5 )* 255
    a6 = a6 * 255
    plt.figure(figsize=(8 ,8))
    plt.subplot(2 ,4 ,1)
    plt.imshow(a1.squeeze().data.cpu().numpy().astype('uint8'), cmap='gray', vmin=0, vmax=255)
    plt.subplot(2, 4, 2)
    plt.imshow(a2.squeeze().data.cpu().numpy().astype('uint8'), cmap='gray', vmin=0, vmax=255)
    plt.subplot(2, 4, 3)
    plt.imshow(a3.squeeze().data.cpu().numpy().astype('uint8'), cmap='gray', vmin=0, vmax=255)
    plt.subplot(2, 4, 4)
    plt.imshow(a_w.squeeze().data.cpu().numpy().astype('uint8'), cmap='gray', vmin=0, vmax=10)
    plt.subplot(2, 4, 5)
    plt.imshow(a4.squeeze().data.cpu().numpy().astype('uint8'), cmap='gray', vmin=0, vmax=255)
    plt.subplot(2, 4, 6)
    plt.imshow(a5.squeeze().data.cpu().numpy().astype('uint8'), cmap='gray', vmin=0, vmax=255)
    plt.subplot(2, 4, 7)
    plt.imshow(~(a6.squeeze().data.cpu().numpy().astype('uint8')), cmap='gray', vmin=0, vmax=255)

    plt.show()
