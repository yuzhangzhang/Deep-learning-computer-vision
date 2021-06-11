# -*- coding: utf-8 -*-
# @Author  : Youquan Liu
# @FileName: training.py
# @Software: PyCharm
# @Description: This Code is for training LeNet5 model on GTSRB dataset by using GPU/CPU



import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image
import PIL
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# traffic sign pre-processing
def process_data(image):
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in image])
    # histogram equalization
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])
    X = np.float32(X)

    # standardize features
    mean_img = np.mean(X, axis=0)
    std_img = (np.std(X, axis=0) + np.finfo('float32').eps)
    X -= mean_img
    X /= std_img
    return X

# Read pictures and their category information from files and pre-processing
def load_traffic_sign_data(training_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_train = process_data(X_train)
    return X_train, y_train # images, targets

# data augmentation
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply([
        transforms.RandomRotation(15, resample=PIL.Image.BICUBIC),
        transforms.RandomAffine(0, translate=(0.1, 0.1), resample=PIL.Image.BICUBIC),
        transforms.RandomAffine(0, scale=(0.9, 1.1), resample=PIL.Image.BICUBIC)
    ]),
    transforms.ToTensor()
])


# create custom dataloader
class MyDataset(Dataset):
    def __init__(self, images=None, labels=None, transform=train_tf):
        super(MyDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


# Creat LeNet5 Model
class LeNet(nn.Module):
    def __init__(self):
        # Define the required layers(Convolutional layer, pooling layer, fully connection layer)
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5) #28x28
        self.pool1 = nn.MaxPool2d(2,2) #14x14
        self.conv2 = nn.Conv2d(6,16,5) #10x10
        self.pool2 = nn.MaxPool2d(2,2) #5x5
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,43)

    # Model forward calculation process
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.5)
        x = self.fc3(x)
        return x


# training model
def train_model(model,train_loader,optimizer,epochs,device):
    model = model.to(device=device)   # if device is GPU then send model to GPU, otherwise CPU
    for epoch in range(epochs):
        correct = 0
        for index, (data,target) in enumerate(train_loader):
            data = data.to(device=device)  # send images to GPU or CPU
            out = model(data) # Pass the image into the model
            target = target.type(torch.LongTensor) # Change data type
            target = target.to(device=device)  # send targets to GPU or CPU

            loss = criterion(out,target) # Use cross entropy to calculate loss
            pred = out.data.max(1)[1] # Get predicted category
            correct += pred.eq(target.data).sum().item() # Compare the real category and the predicted category, and accumulate the correct number
            optimizer.zero_grad() # Gradient clear
            loss.backward()  # Compute gradient
            optimizer.step() # Gradient update

        print("epoch:%s,loss:%s,train_acc:%s" % (epoch+1, loss.item(), correct / len(train_loader.dataset)))
    # save trained model
    torch.save(model.state_dict(), '/content/drive/MyDrive/trained_model5.pth')
    print('trained_model5.pth was saved')

if __name__=='__main__':

    a, b = load_traffic_sign_data('/content/drive/MyDrive/train.p')
    # goes throgh the data augmentation and pack the images and targets into batches
    trainloader = DataLoader(
        dataset=MyDataset(images=a, labels=b),
        batch_size=16,
        shuffle=True
    )

    model = LeNet()

    # load the trained model which has trained 700 epochs and continue to train
    train_weights = '/content/drive/MyDrive/trained_model4.pth'
    state_dict = torch.load(train_weights)
    model.load_state_dict(state_dict) # Load pre-training weights
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # initialize loss and epoch
    loss = 10000000000000
    epochs =300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model,trainloader,optimizer,epochs,device)