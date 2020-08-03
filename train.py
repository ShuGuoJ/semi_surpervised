import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from visdom import Visdom
import numpy as np
from resnet18 import Resnet18
from utils import *

def train(learning_rate=1e-1, batch_size=10, epochs=50):
    '''模型训练'''

    # batch_size = 10
    # epochs = 50
    # learning_rate = 5e-6
    # cbam: learning_rate = 1e-5
    # se: learning_rate = 1e-4
    seed = 123456
    torch.manual_seed(seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

    data_path = "../data/cat_vs_dog"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomCrop((256, 256), 5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MyDataset(data_path, transform,semi_surpervised=True, unlabel_ratio=0.7)
    test_dataset = MyDataset(data_path, transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    net = Resnet18(3,2)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # (optimizer, step_size, gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    net.to(device)
    criterion.to(device)
    total_loss = []
    total_label_loss = []
    total_unlabel_loss = []
    viz = Visdom()
    viz.line([[0., 0.]], [0.], win="train", opts=dict(title="train&&val loss",
                                                      legend=['train', 'val']))
    viz.line([0.], [0.], win="acc", opts=dict(title="accuracy",
                                              legend=['acc']))
    for epoch in range(epochs):
        net.train()
        total_loss.clear()
        total_label_loss.clear()
        total_unlabel_loss.clear()
        for batch, (input, label) in enumerate(train_loader):
            # input, label = torch.rand(2,3,256,256), torch.LongTensor([0,1])
            input, label = input.to(device), label.to(device)
            logits = net(input)
            # 取出label=-1的样本索引
            unlabel_samples = label== -1
            label = label.masked_select(~unlabel_samples)
            unlabel_samples = unlabel_samples.unsqueeze(-1)
            label_logits = logits.masked_select(~unlabel_samples)
            unlabel_logits = logits.masked_select(unlabel_samples)
            if label_logits.numel() != 0:
                label_logits = label_logits.view(-1, 2)
            if unlabel_logits.numel() != 0:
                unlabel_logits = unlabel_logits.view(-1, 2)

            # compute lable loss
            label_loss = criterion(label_logits, label) if label_logits.numel()!=0 else torch.tensor(0., device=device)
            # compute unlable loss
            unlabel_loss = CountEntropy(unlabel_logits) if unlabel_logits.numel()!=0 else torch.tensor(0., device=device)
            # total_loss = label_loss + unlabel_loss
            loss = label_loss + 0.5 * unlabel_loss
            total_loss.append(loss.item())
            total_label_loss.append(label_loss.item())
            total_unlabel_loss.append(unlabel_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                print("epoch:{} batch:{} loss:{} label_loss:{} unlabel_loss:{}".format(epoch, batch, loss.item(), label_loss.item(),
                                                                                       unlabel_loss.item()))

        net.eval()
        correct = 0
        test_loss = 0
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            logits = net(input)

            '''crossentropy'''
            test_loss += criterion(logits, label).item() * input.shape[0]
            pred = logits.argmax(dim=1)

            correct += pred.eq(label).float().sum().item()
        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), acc))
        viz.line([[float(np.mean(total_loss)), test_loss]], [epoch], win="train", update="append")
        viz.line([acc], [epoch], win='acc', update='append')
        torch.save(net.state_dict(), "models/semi_surpervised/resnet18_{}.pkl".format(epoch))
        scheduler.step()


if __name__ == '__main__':
    train(1e-5)