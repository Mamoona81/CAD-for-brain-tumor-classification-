# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:13:57 2019

@author: MS
"""
import torch
torch.manual_seed(0)
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.ImageFolder(root='./train', transform=transform)    # take samples
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testset = torchvision.datasets.ImageFolder(root='./test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)


classes = (' meningioma', 'glioma', 'pituitary tumor')

import torch.nn as nn          #nn is base class for all neural networks modules. our model is subclass to this class
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()    #inherit
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 previous filter 64 new filter and 3 is 3x3 filter size
        self.fc1 = nn.Linear(12544,40) # input layers ,hidden layers
                                                      # self.fc3=nn.Linear(40,40)
        self.fc2 = nn.Linear(40,3)          # hidden layers ,output layers

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # output size: [batch_size, 32, 255, 255]
        x = self.pool(F.relu(self.conv2(x)))  # output size: [batch_size, 64, 126, 126]
        
        x = x.view(-1,12544)  # output size: [batch_size, 64*126*126]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x=self.softmax(x)
        return x


x = torch.randn(4, 3, 64, 64) #(batch size or #of images,channels RGB,width,height)
model = Net()
output = model(x)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9) #SGD:Implements stochastic gradient descent
# (optionally with momentum).

#optimizer = optim.Adam(model.parameters(), lr=0.001)#Implements Adam algorithm.

#It has been proposed in Adam: A Method for Stochastic Optimization.


trainloss=[]
testloss=[]
trainaccuracy=[]
testaccuracy=[]
itr=0;
for epoch in range(10):
    correct=0
    itr=0
    itrloss=0
    model.train()
    
    for i,(images,labels)in enumerate(trainloader):
        itr+=1;
        images=Variable(images)
        labels=Variable(labels)

        optimizer.zero_grad() #Clears the gradients of all optimized torch.Tensor
        outputs=model(images)
        loss=criterion(outputs, labels)
        itrloss+=loss.item()
        loss.backward()#once the gradients are computed using e.g. backward().
        optimizer.step()#method, that updates the parameters
        _,predicted=torch.max(outputs,1)  # 1 dim outputs are tensors
        correct+=(predicted==labels).sum()
        itr+=1
    trainloss.append(itrloss/itr)
    trainaccuracy.append(100*correct/len(trainset))
    torch.save({'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),  # Returns the state of the optimizer as a dict.
                }, 'last_brain1.pth')
    
    #testing
    loss=0.0
    correct=0
    total=0.0
    itr=0
    model.eval()
    for images,labels in testloader:
        images=Variable(images)
        labels=Variable(labels)
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss+=loss.item()
        _,predicted=torch.max(outputs,1)
        total += labels.size(0)
        correct+=(predicted==labels).sum()
        itr+=1
    testloss.append(loss/itr)
    testaccuracy.append((100*correct/len(testset)))
    

    print('training loss:%f %%' %(itrloss/itr))
    print('training accuracy:%f %%'%(100*correct/len(trainset)))
    print('test loss:%f %%'%(loss/itr))
    print('test accuracy:%f %%'%((100*correct/len(testset))))
print('Accuracy of the network on the  test images: %d %%' % (
    100 * correct / total))


class_correct = list(0 for i in range(3))                  # cpred
class_total = list(0 for i in range(3))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        #class_total=[]
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(3):
    print('Accuracy of %5s : %2f %%' % ( classes[i], 100 * class_correct[i] / class_total[i]))
    
    
#plotting

import matplotlib.pyplot as plt
#
f=plt.figure(figsize=(5,5))
#
plt.plot(trainaccuracy,label='training accuracy')
plt.plot(testaccuracy,label='testing accuracy')
plt.legend()
plt.show()