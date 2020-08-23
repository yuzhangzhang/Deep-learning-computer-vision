import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import model
import data
import loss
import torch.optim as optim

# model = model.resnet18(pretrained=False)
model = torch.load('bottle_resnet18.pt')
learning_rate = 0.00005
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_model(images,gts,labels):
    for epoch in range(10):  # loop over the dataset multiple times
        loss_value=0.0
        for image,gt,label in zip(images,gts,labels):
            optimizer.zero_grad()
            outputs = model(image)
            # print(outputs.shape)
            gt = gt.squeeze(1)
            # print(gt.shape)
            loss_value = loss.calculate_loss(outputs,gt,label)
            loss_value.backward()
            optimizer.step()
        train_acc = get_acc(images, labels, 0, 70)
        print("epoch=%s,loss=%s,train_acc=%s" % (epoch,loss_value.item(),train_acc))

    # 保存模型参数
    torch.save(model, 'bottle_resnet18.pt')
    print('bottle_resnet18.pt saved')

def get_acc(images,labels,start,end):
    correct = 0
    for image, label in zip(images[start:end], labels[start:end]):
        for img, lab in zip(image, label):
            img = img.unsqueeze(0)
            y = model(img)
            pred = torch.argmax(y).item()  # y是非0即1的向量，1所在的位置对应它的标签，所以取（y-1）最小值，其所在位置即为标签
            if lab == pred:
                correct += 1
    correct = float(correct / float(end - start))
    return correct  # 正确的个数/总数


if __name__=='__main__':
    gen = data.train_image_gen(2)
    images, gts, labels = next(gen)
    train_model(images, gts, labels)


