import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import yolo_model
import data
import Loss
import ResNet
import decoder
'''
'百岁山':1,'怡宝':2,'百事可乐':3,'景甜':4,'娃哈哈':5,'康师傅':6,'苏打水':7,
'天府可乐':8,'可口可乐':9,'农夫山泉':10,'恒大冰泉':11,'其它':12,'冰露':13
'''
#分类
classes_list = ["今麦郎", "冰露", "百岁山", "怡宝", "百事可乐", "景甜", "娃哈哈", "康师傅", "苏打水", "天府可乐",
                "可口可乐", "农夫山泉", "恒大冰泉", "其他"]
Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


#模型
# yolov0_model = yolo_model.YOLOV0()
model = torch.load('bottle_resnet5.pt')
model.eval()


image_path, label_path,label_name = data.make_list()
test_image_path = image_path[130:137]
# print(len(test_image_path))
test_label_path = label_path[130:137]
test_label_name = label_name[130:137]

label_list = data.make_label(test_label_name)
bboxes = data.get_bbox(test_label_path,test_image_path)

batch_size = 1

test_dataset = data.myDataset(image_path=test_image_path, labels_list =label_list ,bboxes=bboxes, train=False, transform=[transforms.ToTensor()])
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

criterion = Loss.yoloLoss(14, 2, 5, 0.5)#权重参数，某些要占大的比重，某些占小的比重


for i,(image,target) in enumerate(test_loader):
    pred = model(image)
    # print(pred)
    loss = criterion(pred, target)
    image_name = test_label_name[i]
    bbox = bboxes[i]
    print(bbox)
    img_path = test_image_path[i]
    cv_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  ## 读取图像，解决imread不能读取中文路径的问题
    h,w,c = cv_img.shape
    # cv_img = cv2.resize(cv_img, (448,448))  # 将所有图片都resize到指定大小
    # bbox = torch.FloatTensor(bbox)
    # bbox /= torch.FloatTensor([w, h, w, h]).expand_as(bbox)  # 坐标归一化处理，为了方便训练
    # print(bbox)
    # bbox *= torch.LongTensor([448/w, 448/h, 448/w, 448/h]).expand_as(bbox)
    # bbox *= 448
    # print(bbox)
    for box in bbox:
        cv2.rectangle(cv_img, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])), (255,0, 0), 8)
        cv2.putText(cv_img, 'target', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    boxes, cls_indexs, probs = decoder.get_decoder(pred)
    result = []
    n = len(bbox)
    for j,box in enumerate(boxes[0:n]):
        x1 = int(box[0] * w)
        x2 = int(box[2] * h)
        y1 = int(box[1] * w)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[j]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[j]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), cls_index, prob])
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
        cv2.putText(cv_img, 'pred',(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # print([(x1, y1), (x2, y2)])
    # print(bbox)
    # cls_index = int(cls_indexs)  # convert LongTensor to int
    # prob = float(probs)
    # cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
    # cv2.putText(cv_img, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    print('**************************')
    print('第 %d 张图片' % (i))
    print('loss:{}'.format(loss))
    print('H:{}, W:{}'.format(h,w))
    print('target_bbox:{}, target_name:{}'.format(bbox,image_name))
    # print('pred_bbox:{}, pred_class:{}, pred_prob:{}'.format(boxes,cls_index,prob))
    # print('transformed_bbox:{}'.format([x1,y1,x2,y2]))
    print('result:{}'.format(result))
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image', cv_img)
    Img_Name = "D:\Python\CV\week12\week12\output\\" + str(i) + ".jpg"

    cv2.imwrite(Img_Name, cv_img)
    cv2.waitKey(1000)

# predict
#
# result = []
# for i in range(7):
#     bbox = bboxes[i]
#     img_path = test_image_path[i]
#     cv_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  ## 读取图像，解决imread不能读取中文路径的问题
#     h,w,c = cv_img.shape
#     # print(h,w,c)
#     image_name = test_label_name[i]
#     image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (448, 448))
#     image = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / (255.0 / 2) - 1
#     image = torch.from_numpy(image)
#     image = image.unsqueeze(0)
#     # print(image.shape)
#     pred = model(image)
#     loss = criterion(pred, target)
#
#     boxes, cls_indexs, probs = decoder.get_decoder(pred)
#
#     # for i, box in enumerate(boxes):
#     x1 = int(boxes[0] * w)
#     x2 = int(boxes[2] * w)
#     y1 = int(boxes[1] * h)
#     y2 = int(boxes[3] * h)
#     # print([(x1, y1), (x2, y2)])
#     # print(bbox)
#     cls_index = int(cls_indexs)  # convert LongTensor to int
#     prob = float(probs)
#     result.append([(x1, y1), (x2, y2),bbox,cls_index,image_name, prob])
#     cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
#     # cv2.putText(cv_img, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#     cv2.imshow('image', cv_img)
#     plt.show()
#     cv2.waitKey(10000)




# for left_up, right_bottom, class_name, img_path, prob in result:
#     # image = cv2.imread(img_name)
#
#     # print(img_path)
#     image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)  ## 读取图像，解决imread不能读取中文路径的问题
#     label = class_name
#     cv2.rectangle(image, left_up, right_bottom, (0,255,0), 2)
#     cv2.putText(image, label, left_up, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#     cv2.imshow(label, image)
#     plt.show()
#     cv2.waitKey(10000)

# print(result)
print('finish')




