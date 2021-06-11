import model_Le
from Data_loader import *
import torch.nn as nn
import torch


a,b = load_traffic_sign_data('E:\ex_python\sign_detection\\backup\\test.p')
testloader = DataLoader(
    dataset=MyDataset(images=a,labels=b),
    batch_size=1,
    shuffle=True
    )

# train_weights = 'E:\ex_python\sign_detection\\backup\\trained_model2.pth'

model = model_Le.LeNet()
model.load_state_dict(torch.load('E:\ex_python\sign_detection\\backup\\trained_model5.pth', map_location=lambda storage, loc: storage))
model.eval()
test_correct = 0
with torch.no_grad():
    for image,target in testloader:
        out = model(image)
        target = target.type(torch.LongTensor)
        pred = out.data.max(1)[1]
        test_correct += pred.eq(target.data).sum().item()

    print('test_acc:{}'.format(test_correct/len(testloader.dataset)))


