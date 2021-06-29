from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataloader as dl
import matplotlib.pyplot as plt
import seaborn as sb ; sb.set()
import torchvision.models as models
from sklearn.metrics import confusion_matrix

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Res18(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self.make_layer(64, 2, 1)
        self.layer2 = self.make_layer(128, 2, 2)
        self.layer3 = self.make_layer(256, 2, 2)
        self.layer4 = self.make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 5)

    def make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
                )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        layers.append(BasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class BottleneckBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Res50(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, (7,7), stride=(2,2), padding=(3,3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self.make_layer(64, 3, 1)
        self.layer2 = self.make_layer(128, 4, 2)
        self.layer3 = self.make_layer(256, 6, 2)
        self.layer4 = self.make_layer(512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=2048, out_features=5)
   
    def make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*4, stride),
                    nn.BatchNorm2d(planes*4)
                    )
        layers = []
        layers.append(BottleneckBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x      

def main():
    pre_model50 = models.resnet50(pretrained=True)
    pre_model50.fc = nn.Linear(2048, 5)
    pre_model18 = models.resnet18(pretrained=True)
    pre_model18.fc = nn.Linear(512, 5)
    model18 = Res18()
    model50 = Res50()
    nets18 = [model18, pre_model18]
    nets50 = [model50, pre_model50]
    nets_text18 = ['model18', 'pre_model18']
    nets_text50 = ['model50', 'pre_model50']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_train = dl.RetinopathyLoader('data/', 'train')
    data_test = dl.RetinopathyLoader('data/', 'test')
    
    batch_size = 4
    #run_times = 28099 // batch_size
    run_times = 2000
    l_rate = 1e-3
    epochs = 3
    
    for i in range(2):
        torch.cuda.empty_cache()
        #net = nets18[i]
        net = nets50[i]
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=l_rate)
        Xtest, Ytest = list(), list()
        Xtrain, Ytrain = list(), list()

        for epoch in range(epochs):
            print(epoch)
            net.train()
            for run_time in range(run_times):
                rnd_idx = np.random.randint(0, 28099, batch_size)
                img_train, label_train = data_train[rnd_idx]
                img_train, label_train = torch.from_numpy(img_train).float().to(device), torch.from_numpy(label_train).long().to(device) 
                optimizer.zero_grad()
                outputs = net(img_train)
                loss = criterion(outputs, label_train)
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                correct = 0.
                for index in range(0, 6):
                    #print("index : ", index)
                    print(index)
                    idx = np.random.randint(0,28099,50)
                    #img_test, label_test = data_train[range(index, min(index+25,28099))]
                    img_test, label_test = data_train[idx]
                   
                    img_test, label_test = torch.from_numpy(img_test).float().to(device), torch.from_numpy(label_test).long().to(device)
                    outputs = net(img_test)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == label_test).squeeze()
                    c = c.cpu().numpy()
            
                    for cnt in range(c.size):
                        if(c[cnt] == 1):
                            correct += 1
                acc = correct / 300
                Xtrain.append(epoch)
                Ytrain.append(acc)
                print(nets_text50[i],' Train Acc = : ',acc,'\n')
                acc = 0
                correct = 0

            with torch.no_grad():
                correct = 0.
                for index in range(0, 6):
                    idx = np.random.randint(0,7025, 50)
                    #img_test, label_test = data_test[range(index, min(index+50,7025))]
                    img_test, label_test = data_test[idx]

                    img_test, label_test = torch.from_numpy(img_test).float().to(device), torch.from_numpy(label_test).long().to(device)  
                    outputs = net(img_test)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == label_test).squeeze()
                    c = c.cpu().numpy()
            
                    for cnt in range(c.size):
                        if(c[cnt] == 1):
                            correct += 1
                acc = correct / 300
                record = acc
                Xtest.append(epoch)
                Ytest.append(acc)
                print(nets_text50[i],' Test Acc = : ',acc,'\n')
                acc = 0
                correct = 0

        path = "./model/" + nets_text50[i] + str(record)  + ".pkl"
        torch.save(net.state_dict(), path)
        train_name = nets_text50[i] + '_train'
        test_name = nets_text50[i] + '_test'
        plt.plot(Xtrain, Ytrain, label=train_name)
        plt.plot(Xtest,Ytest, label=test_name)

    plt.title('Result Comparison(ResNet50)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.legend(loc='best')
    name = './result/Res50' + str(record) + '.png'
    plt.savefig(name)
    return

if __name__ == '__main__':
    main()
