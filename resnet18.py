import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transform.Compose([transform.ToTensor(),transform.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
trainset, validset = torch.utils.data.random_split(
    torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform = transform),
    lengths=[40000, 10000])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
validloader = torch.utils.data.DataLoader(validset, batch_size=16, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR100(root = './data', train=False, download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)

classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel,out_channel, kernel_size=3, stride=stride, padding=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel),
                                   )
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride),nn.BatchNorm2d(out_channel))
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.shortcut(x) + x1
        x2 = F.relu(x2)
        return x2
    
class ResNet(nn.Module):
    def __init__(self,ResBlock,num_classes):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self.make_layer(ResBlock,2,64,1)
        self.layer2 = self.make_layer(ResBlock,2,128,2)
        self.layer3 = self.make_layer(ResBlock,2,256,2)
        self.layer4 = self.make_layer(ResBlock,2,512,2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
        
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
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
        
    def make_layer(self, ResBlock, num_of_blocks, out_channel, stride):
        strides = [stride] + [1] * (num_of_blocks-1)
        layers = []
          
        for stride in strides:
            layers.append(ResBlock(self.in_channel,out_channel,stride))
            self.in_channel = out_channel
        
        return nn.Sequential(*layers)
        
net = ResNet(ResBlock,num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

if __name__ == '__main__':
    progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc='Training' )
    for epoch in range(1):  # epoch
        running_loss = 0.0
        for i, data in (enumerate(trainloader, 0)):

            inputs, labels = data[0].to(device), data[1].to(device)

            # zero gradient
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            running_loss += loss.item()
            if i % 250 == 0:
                # validation
                net.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in validloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                progress_bar.set_postfix(accuracy=100 * correct / total)
            running_loss = 0.0
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # calculate accuracy
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(100):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))