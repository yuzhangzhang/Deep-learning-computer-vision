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
model = torch.load('bottle_resnet18.pt')
model.eval()

image_path, label_path,label_name = data.make_list()
test_image_path = image_path[140:147]
test_label_path = label_path[140:147]
test_label_name = label_name[140:147]

label_list = data.make_label(test_label_name)
bboxes = data.get_bbox(test_label_path,test_image_path)

batch_size = 1

# predict

result = []
for i in range(7):
    bbox = bboxes[i]
    cv_img = cv2.imdecode(np.fromfile(test_image_path[i], dtype=np.uint8), -1)  ## 读取图像，解决imread不能读取中文路径的问题
    h,w,_ = cv_img.shape
    image_name = test_label_name[i]
    image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (448, 448))
    image = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / (255.0 / 2) - 1
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    pred = model(image)
    # print(pred)
    boxes, cls_indexs, probs = decoder.get_decoder(pred)
    x1 = int(boxes[0] * w)
    x2 = int(boxes[2] * w)
    y1 = int(boxes[1] * h)
    y2 = int(boxes[3] * h)
    cls_index = int(cls_indexs)  # convert LongTensor to int
    prob = float(probs)
    result.append([(x1, y1), (x2, y2),bbox,cls_index,image_name, prob])


    img = cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    class_name = str(cls_index)
    cv2.putText(img, class_name,(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imshow(class_name, img)
    plt.show()
    cv2.waitKey(1000)
print(result)
print('finish')




