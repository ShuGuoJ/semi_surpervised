from torchvision import transforms
import os, torch
from PIL import Image
from torch.utils.data import Dataset
import random
from random import shuffle
import glob
'''自定义数据集'''

class MyDataset(Dataset):
    def __init__(self, path, transform=None, train=True, unlabel_ratio=0.5, semi_surpervised=True):
        super(MyDataset, self).__init__()
        self.transform = transform
        cats = glob.glob(os.path.join(path, 'cat.*.jpg'))
        dogs = glob.glob(os.path.join(path, 'dog.*.jpg'))
        self.unlabel_datas = None
        if train:
            train_cats, train_dogs = cats[:int(len(cats)*0.8)], dogs[:int(len(dogs)*0.8)]
            self.unlabel_datas = train_cats[:int(unlabel_ratio*len(train_cats))] + train_dogs[:int(unlabel_ratio*len(train_dogs))]
            # semi_surpervised
            if semi_surpervised:
                self.datas = train_cats + train_dogs
            # surpervised
            else:
                self.datas = train_cats[int(unlabel_ratio*len(train_cats)):] + train_dogs[int(unlabel_ratio*len(train_dogs)):]
        else:
            self.datas = cats[int(len(cats)*0.8):] + dogs[int(len(dogs)*0.8):]

        shuffle(self.datas)

    def __len__(self):
        return len(self.datas)


    def __getitem__(self, index):
        label = self.datas[index].split('\\')[1][:3]
        label = 1 if label=='dog' else 0
        path = self.datas[index]
        if self.unlabel_datas is not None and path in self.unlabel_datas:
            label = -1
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)

# from torch.utils.data import DataLoader
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.ToTensor()
# ])
# train_dataset = MyDataset('../data/cat_vs_dog', transform=transform)
# test_dataset = MyDataset('../data/cat_vs_dog', transform=transform, train=False)
# train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)
# print(next(iter(train_loader))[-1])
# print(next(iter(test_loader))[-1])
