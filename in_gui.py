# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'in_gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from predict_brain import prediction
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import os
import torch.optim as optim
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap,QIcon
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog,QMessageBox
from PyQt5.uic import loadUi
from PyQt5 import QtCore
import torch
torch.manual_seed(0)
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt




import torch.nn as nn
import torch.nn.functional as F
class Design(QDialog):
    def __init__(self):
        super(Design, self).__init__()
        loadUi('in_gui.ui', self)
        self.image = None
        self.processedImage = None
        self.imageLoaded = True
        #....................Button Actions......................................#
        self.load.clicked.connect(self.loadmodel)
        self.load.clicked.connect(self.loading)
        self.cnntest.clicked.connect(self.accuracy)
        #self.accu.clicked.connect(self.accloading)
        self.browse.clicked.connect(self.loadClicked)
        #self.prep.clicked.connect(self.preprocess)
        self.predic.clicked.connect(self.predict)
        self.predic.clicked.connect(self.preloading)
        self.cls.clicked.connect(self.clearclicked)
        self.mask.clicked.connect(self.maskclicked)





#..............................Browse Clear button................................................#
    @pyqtSlot()
    def clearclicked(self):

        try:
            self.debugPrint("Clear Image\n")
            self.label.clear()
            self.label3.clear()


        except Exception:
            #print(traceback.format_exc())
            pass
#...............................................tumor mask...................................................
    @pyqtSlot()
    def maskclicked(self):
        img=cv2.imread('./save/singleImage/.png')
        #print(img)
        ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        self.processedImage=thresh1
        self.displayImage(3)

#............................................Message printing...........................#
    def debugPrint(self, msg):
        self.textBrowser.append(msg)

#.............................................Load CNN ,odel.....................................#
    def loadmodel(self):
        model=torch.load("last_brain1.pth")
        print(model)
        self.debugPrint("Loading Trained CNN Model.............")
#...........................................Load model Bar.............................................#
    def loading(self):
        self.completed = 0
        while self.completed < 100:
            self.completed += 0.00001
            self.loadbar.setValue(self.completed)
        self.debugPrint("Model Loaded.....")
    #.........................................................CNN Testing Button..............................#
    def accuracy(self):
        try:


            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            trainset = torchvision.datasets.ImageFolder(root='./train', transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                      shuffle=True)

            testset = torchvision.datasets.ImageFolder(root='./test', transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                     shuffle=False)
            classes = (' meningioma','glioma', 'pituitary tumor')

            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.conv2 = nn.Conv2d(32, 64, 3)
                    self.fc1 = nn.Linear(64 * 14 * 14, 40)
                    self.fc2 = nn.Linear(40, 3)

                def forward(self, x):
                    x = self.pool(F.relu(self.conv1(x)))  # output size: [batch_size, 32, 255, 255]
                    x = self.pool(F.relu(self.conv2(x)))  # output size: [batch_size, 64, 126, 126]

                    x = x.view(-1, 64 * 14 * 14)  # output size: [batch_size, 64*126*126]
                    x = F.relu(self.fc1(x))
                    x = self.fc2(x)

                    return x

            x = torch.randn(4, 3, 64, 64)  # (batch size or #of images,channels RGB,width,height)
            model = Net()
            output = model(x)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001,
                                  momentum=0.9)  # SGD:Implements stochastic gradient descent


            trainloss = []
            testloss = []
            trainaccuracy = []
            testaccuracy = []
            itr = 0;
            self.debugPrint("testing start.......")

            for epoch in range(10):
                correct = 0
                itr = 0
                itrloss = 0
                model.train()

                for i, (images, labels) in enumerate(trainloader):
                    itr += 1;
                    images = Variable(images)
                    labels = Variable(labels)

                    optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    itrloss += loss.item()
                    loss.backward()  # once the gradients are computed using e.g. backward().
                    optimizer.step()  # method, that updates the parameters
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum()
                    itr += 1
                trainloss.append(itrloss / itr)
                trainaccuracy.append(100 * correct / len(trainset))

                # testing
                loss = 0.0
                correct = 0
                total = 0.0
                itr = 0
                model.eval()
                for images, labels in testloader:
                    images = Variable(images)
                    labels = Variable(labels)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    itr += 1
                testloss.append(loss / itr)
                testaccuracy.append((100 * correct / len(testset)))
                # print('training loss:%f %%' % (itrloss / itr))
                print('training accuracy:%f %%' % (100 * correct / len(trainset)))
                # print('test loss:%f %%' % (loss / itr))
                print('test accuracy:%f %%' % ((100 * correct / len(testset))))
            print('Accuracy of the network on the  test images: %d %%' % (
                100 * correct / total))

            class_correct = list(0 for i in range(3))
            class_total = list(0 for i in range(3))
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = Variable(images)
                    labels = Variable(labels)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    # class_total=[]
                    for i in range(labels.size(0)):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
            for i in range(3):
                print('Accuracy of %5s : %2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

        except Exception:
            print("....")
        self.debugPrint("Testing Done Sucessfully")



#......................................................................Browse Button...............................

    @pyqtSlot()
    def loadClicked(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open FIle', 'C:\\', "Image Files(*.png)")
        if fname:
            self.loadImage(fname)
            self.debugPrint("Browse Image.")
        # self.loadImage('adn.jpg')
        else:
            print("kindly input the image")

#.................................................... image Loading...................................#
    def loadImage(self, fname):
        self.image = cv2.imread(fname, cv2.IMREAD_COLOR)
        self.processedImage = self.image.copy()
        cv2.imwrite(os.path.join('./save/singleImage', '.png'),self.image)
        self.displayImage(1)

#................................................predict button.................................................#
    def predict(self):

        try:
            prediction()
            self.debugPrint("Prediction Done")

        except:
            self.debugPrint("enter only Valid size 64 x 64 of image")

#.............................................................prediction bar  ................................,,#
    def preloading(self):
        self.completed = 0
        while self.completed < 100:
            self.completed += 0.001
            self.predbar.setValue(self.completed)

#....................................Disply image in labels................................>>>>>>>>>>>>>>>>>#
    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.processedImage.shape) == 3:  # rows[0],col[1],chann[2]
            if (self.processedImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.processedImage, self.processedImage.shape[1], self.processedImage.shape[0],
                     self.processedImage.strides[0], qformat)
        # BGR > RBG convert
        img = img.rgbSwapped()
        if window == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window == 2:
            self.label3.setPixmap(QPixmap.fromImage(img))
            self.label3.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window == 3:
            self.label3.setPixmap(QPixmap.fromImage(img))
            self.label3.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)




app = QApplication(sys.argv)
window=Design()
window.setWindowTitle('CAD For Brain Tumor Classification')
window.setWindowIcon(QtGui.QIcon('brainimage.jpg'))
window.show()
sys.exit(app.exec_())


