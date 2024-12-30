# -*- coding: utf-8 -*-
import torch
torch.manual_seed(0)
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



def prediction():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    criterion = nn.CrossEntropyLoss()
    singletestset = torchvision.datasets.ImageFolder(root='./save', transform=transform)
    singletestloader = torch.utils.data.DataLoader(singletestset, batch_size=4,
                                                   shuffle=False, num_workers=0)
    classes = (' meningioma', 'glioma', 'pituitary tumor')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.fc1 = nn.Linear(64 * 14 * 14, 40)
            self.fc2 = nn.Linear(40, 3)
            # self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # output size: [batch_size, 32, 255, 255]
            x = self.pool(F.relu(self.conv2(x)))  # output size: [batch_size, 64, 126, 126]

            x = x.view(-1, 64 * 14 * 14)  # output size: [batch_size, 64126126]
            print(x.shape)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    model = Net()
    state_dict = torch.load('last_brain1.pth')['state_dict']
    model.load_state_dict(state_dict)
    # testing
    loss = 0.0
    correct = 0
    total = 0.0
    itr = 0
    testaccuracy = []
    model.eval()
    for images, labels in singletestloader:
        images = Variable(images)
        labels = Variable(labels)
        # CUDA=torch.cuda.is_available()
        # if CUDA:
        # images=images.cuda()
        # labels=labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        itr += 1
        i=predicted.item()
        print(i)
        if predicted.item()==0:
            print("CAD predicted:Meningioma")
        if predicted.item() == 1:
            print("CAD Predicted:Glioma")
        if predicted.item() == 2:
            print("CAD Predicted :pituitry Tumor")




