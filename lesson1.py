import torch
import torchvision#torchvision包 包含了目前流行的数据集，模型结构和常用的图片转换工具。
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  #数据可视化
import numpy as np  #支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

transform = transforms.Compose(           #将多个transform组合起来使用。
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#class torchvision.transforms.ToTensor把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
#转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensorclass。
#torchvision.transforms.Normalize(mean, std)给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
#即：Normalized_image=(image-mean)/std。

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
''' - root : cifar-10-batches-py 的根目录 - train : True = 训练集, 
False = 测试集 - download : True = 从互联上下载数据，并将其放在root目录下。
如果数据集已经下载，什么都不干。'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
#该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor,
# num_workers读取样本的线程数
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize反归一化
    npimg = img.numpy()      #转numpy类
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  #transpose交换维度（0，1，2）转化为（1，2，0），imshow画图
    plt.show()    #显示图片


# get some random training images
dataiter = iter(trainloader)      #生成迭代器。iter(object[, sentinel])object -- 支持迭代的集合对象。
#sentinel -- 如果传递了第二个参数，则参数 object 必须是一个可调用的对象（如，函数），此时，iter 创建了一个迭代器对象，每次调用这个迭代器对象的__next__()方法时，都会调用 object。
#返回迭代器对象。
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
images, labels= images.to(device), labels.to(device)

#给定 4D mini-batch Tensor， 形状为 (B x C x H x W),或者一个a list of image，做成一个size为(B / nrow, nrow)的雪碧图。

#normalize=True ，会将图片的像素值归一化处理

#如果 range=(min, max)， min和max是数字，那么min，max用来规范化image

#scale_each=True ，每个图片独立规范化，而不是根据所有图片的像素最大最小值来规范化
# print labels
print(' '.join('%5s' % labels[j] for j in range(4)))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):# 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__()# 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(3, 6, 5)# 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.pool = nn.MaxPool2d(2, 2)# 使用2x2的窗口进行最大池化Max pooling。
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1 = nn.Linear(16 * 5 * 5, 120)# 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2 = nn.Linear(120, 84) # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3 = nn.Linear(84, 10) # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。

    def forward(self, x): # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
        x = self.pool(F.relu(self.conv1(x))) # 输入x经过卷积conv1之后，经过激活函数ReLU ，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = self.pool(F.relu(self.conv2(x))) # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, 16 * 5 * 5)# view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x)) # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))# 输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x) # 输入x经过全连接3，然后更新x
        return x

net = Net()
net.cuda()

import torch.optim as optim   #建立损失函数和优化器

criterion = nn.CrossEntropyLoss()#定义loss函数
optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9)#基于SGD的优化器
for epoch in range(2):  # 对数据集进行多次循环，epoch：迭代次数，1个epoch等于使用训练集中的全部样本训练一次。

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):#enumerate(sequence, [start=0]),sequence -- 一个序列、迭代器或其他支持迭代对象。start -- 下标起始位置。
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)#forward
        outputs=outputs.to(device)
        loss = criterion(outputs, labels)
        loss.backward()#backward
        optimizer.step()#optimize

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
'''总结一下，该部分代码总共做了以下几件事
定义优化器与代价函数
执行网络训练
执行网络训练部分，每次迭代包括以下操作

1.初始化梯度
2.执行前馈计算代价函数
3.执行反馈计算梯度并更新权值'''

dataiter = iter(testloader)
images, labels = dataiter.next()
labels = labels.to(device)


# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
images=images.to(device)
outputs = net(images)
outputs=outputs.to(device)
_, predicted = torch.max(outputs, 1)#不需要最大概率，只需要最大概率的索引，使用_,predicted = torch.max(outputs.data,1)
# 在第一维看取出最大的数（丢弃）和最大数的位置（保留）后再与label相比即可进行测试。

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


#测试正确率
correct = 0 #统计预测正确个数
total = 0 #总样本数
with torch.no_grad():#测试不需要修改参数
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        outputs=outputs.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#每个class的表现情况
class_correct = list(0. for i in range(10))#快速创建列表，[0,0,0,0,0,0,0,0,0,0]
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:#读入测试集中所有的资料
        images, labels = data#在data中分别读出images和label信息
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)#用训练出的模型来测试images
        outputs = outputs.to(device)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()#squeeze（）函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        #生成如：c=tensor([1, 0, 1, 1], device='cuda:0', dtype=torch.uint8)
        print(c)
        for i in range(4):                 #由于testloader是个集合数据集，故采集出来也为多维Tensor，需要使用循环计算每个图片的预测结果
            label = labels[i]
            class_correct[label] += c[i].item()#item()方法取出tensor中第i个元素
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

