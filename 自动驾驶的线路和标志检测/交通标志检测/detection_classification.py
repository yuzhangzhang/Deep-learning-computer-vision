# -*- coding: utf-8 -*-
# @Author  : Youquan Liu
# @FileName: detection_classification.py
# @Software: PyCharm
# @Description: This Code achieves traffic sign detection and classification

# Import required libraries
import cv2
import imutils
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import os

# ------------------------------ classififcation part ------------------------------#
# Create a dictionary containing all category information
label_dict = {0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
              3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)',
              5: 'Speed limit (80km/h)', 6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
              8: 'Speed limit (120km/h)', 9: 'No passing',
              10: 'No passing for vechiles over 3.5 metric tons', 11: 'Right-of-way at the next intersection',
              12: 'Priority road', 13: 'Yield',
              14: 'Stop', 15: 'No vechiles', 16: 'Vechiles over 3.5 metric tons prohibited', 17: 'No entry',
              18: 'General caution',
              19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right', 21: 'Double curve',
              22: 'Bumpy road', 23: 'Slippery road',
              24: 'Road narrows on the right', 25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
              28: 'Children crossing', 29: 'Bicycles crossing',
              30: 'Beware of ice/snow', 31: 'Wild animals crossing', 32: 'End of all speed and passing limits',
              33: 'Turn right ahead', 34: 'Turn left ahead',
              35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
              39: 'Keep left', 40: 'Roundabout mandatory',
              41: 'End of no passing', 42: 'End of no passing by vechiles over 3.5 metric tons'}


# Creat LeNet5 Model
class LeNet(nn.Module):
    def __init__(self):  # Define the required layers(convolutional layer, pooling layer, fully connection layer)
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14
        self.conv2 = nn.Conv2d(6, 16, 5)  # 10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):  # Model forward calculation process
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.dropout(x,p=0.5)
        x = self.fc3(x)
        return x


# traffic sign pre-processing
def process_data(image):
    # convert from RGB to YUV
    X = np.array(np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[:, :, 0], 2))
    # histogram equalization
    X = np.array(np.expand_dims(cv2.equalizeHist(np.uint8(X)), 2))
    X = np.float32(X)

    # standardize features
    mean_img = np.mean(X, axis=0)  # Mean
    std_img = (np.std(X, axis=0) + np.finfo('float32').eps)  # variance
    X -= mean_img
    X /= std_img
    return X


# traffic signs input to the trained model and make classification
def get_probability(img):
    img = np.asanyarray(img)  # Convert array form
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)  # Resize the image
    # cv2.imwrite('E:\ex_python\sign_detection\image1\\resize.jpg', img)
    img = process_data(img)  # call process_data function

    img = torch.from_numpy(img.transpose((2, 0, 1)))  # Adjust the dimensions of the image
    img = img.float().unsqueeze(0)  # Ascending dimension

    with torch.no_grad():  # Do not calculate derivatives in network parameters
        out = model(img)  # Pass the image into the model
        # Calculate the probability distribution of model prediction results
        probability = torch.nn.functional.softmax(out, dim=1)
        # Take the value with the highest probability
        probabilityValue = probability.data.max(1)[0]
        # Take the index of the value with the highest probability and use it as the category predicted by the model
        class_pred = probability.data.max(1)[1]
        # Take out the category corresponding to the predicted category number in the dictionary storing all categories
        label = label_dict[class_pred.item()]

    return probabilityValue, label


# ------------------------------ detection part ------------------------------#
# Convert grayscale image(after histogram equalization) to hsv color space
def img_to_hsv(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Since many believe that the RGB color space is very fragile with regards to changes in lighting
    img_hsv = cv2.cvtColor(img_output, cv2.COLOR_BGR2HSV)
    return img_hsv


def further_process(thresh_img):
    r_thresh = cv2.GaussianBlur(thresh_img, (5, 5), 0)  # Gaussian filtering

    kernel_1 = np.ones((3, 3), np.uint8)  # Define a 3*3 array of all ones
    kernel_2 = np.ones((5, 5), np.uint8)
    # Corrosion, expansion and opening operations in morphology
    erosion = cv2.erode(r_thresh, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)

    # Do MSER
    mser_red = cv2.MSER_create(8, 100, 10000)
    regions, _ = mser_red.detectRegions(np.uint8(opening))
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    blank_im = np.zeros_like(r_thresh)  # create a mask
    cv2.fillPoly(np.uint8(blank_im), hulls, (255, 255, 255))  # fill a blank image with the detected hulls
    # close operations in morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 6))
    closed = cv2.morphologyEx(blank_im, cv2.MORPH_CLOSE, kernel)
    # find contours which are traffic signs possibly
    cnts = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def red_color_select(img):
    h_img, w_img, c_img = img.shape

    img_hsv = img_to_hsv(img)  # call function
    # Choose color threshold
    lower_red_1 = np.array([0, 70, 60])
    upper_red_1 = np.array([10, 255, 255])
    # Take out the part within the threshold
    mask_1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
    lower_red_2 = np.array([170, 70, 60])
    upper_red_2 = np.array([180, 255, 255])
    mask_2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    # create a red color mask
    red_mask = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

    # separating channels
    r_channel = red_mask[:, :, 2]
    g_channel = red_mask[:, :, 1]
    b_channel = red_mask[:, :, 0]

    # filtering
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)
    # create a red gray space
    filtered_r = 4 * filtered_r - 0.5 * filtered_b - 2 * filtered_g

    # Threshold processing
    _, r_thresh = cv2.threshold(np.uint8(filtered_r), 20, 255, cv2.THRESH_BINARY)

    cnts = further_process(r_thresh)  # call function
    cnts_all = []
    for cnt in cnts:
        # Get the coordinate information of the smallest bounding rectangle of contours
        x_, y_, weight, height = cv2.boundingRect(cnt)
        if x_ < 0.5 * w_img:  # traffic signs are on the right hand
            continue
        # The red traffic sign is generally at the position of 0 to 3/4 image width of the picture coordinate system
        if (y_ > 0.75 * h_img):
            continue

        # contour area
        area = cv2.contourArea(cnt)

        if 500 < area < 10000:  # further filter contours by area
            # roundness
            Roundness = (area * 4 * math.pi) / (cv2.arcLength(cnt, True) ** 2)
            ((cx, cy), (w, h), theta) = cv2.minAreaRect(cnt)
            (x, y, w, h) = cv2.boundingRect(cnt)
            area_rect = w * h
            # rectangularity
            Rectangularity = area / area_rect
            # elongation
            Elongation = min(w, h) / max(w, h)

            # Through roundness,rectangularity,elongation to find red circular traffic sign
            if Roundness > 0.85 and Rectangularity > 0.70 and Elongation > 0.75:  # possible circle sign
                cnts_all.append(cnt)

            # Through roundness,rectangularity,elongation to find red triangle traffic sign
            elif 0.35 < Roundness < 0.75 and 0.4 < Rectangularity < 0.65 and Elongation > 0.70:
                cnts_all.append(cnt)

    return cnts_all


# ------------------------------ main function ------------------------------#
def main(img):

    img = img.copy()  # protect image
    h_img, w_img, c = img.shape
    # 2 optionals of image size: 906x533; 1360x800
    # img = cv2.resize(img, (906, 533), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (1360, 800), interpolation=cv2.INTER_CUBIC)

    cnts_all = red_color_select(img)

    for cnt in cnts_all:
        # Get the coordinate information of traffic signs
        x, y, w, h = cv2.boundingRect(cnt)
        out = img[y:y + h, x:x + w]  # crop the region of traffic sign
        # load classfier
        probability, label = get_probability(out)
        if probability > 0.75:   # set a threshold for further filter traffic signs
            cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)   # draw bounding box

            cv2.putText(img, str(label), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)  # draw predicted label

    # img = cv2.resize(img, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    return img


if __name__ == "__main__":
    # test image
    model = LeNet()
    # load trained model in cpu
    model.load_state_dict(torch.load('E:\ex_python\sign_detection\\backup\\trained_model5.pth',map_location=lambda storage, loc: storage))
    model.eval()


    '''test signle image'''
    img = cv2.imread('E:\ex_python\sign_detection\image1\\20201216_131445.jpg')
    img = main(img)
    cv2.imshow('original', img)
    cv2.waitKey(1000)

    ''' for loop read images and test '''
    # for file_img in os.listdir('E:\ex_python\sign_detection\image1'):
    #     img = cv2.imread('E:\ex_python\sign_detection\image1' + '\\' + file_img)
    #     img = main(img)
    #     cv2.imshow('original', img)
    #     cv2.waitKey(600)

    '''   # video test'''
    # # Define the codec and create VideoWriter object
    # cap = cv2.VideoCapture('E:\ex_python\sign_detection\\videos\\new\\test_videos_and_results\\test_videos_15fps\\test7.mp4')  # read video from video address
    # ret, frame = cap.read()
    # height = frame.shape[0]
    # width = frame.shape[1]
    # fps = cap.get(cv2.CAP_PROP_FPS)  # return video fps
    # # address for write output video
    # video = cv2.VideoWriter('E:\ex_python\sign_detection\\videos\\new\\test_result_8.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),fps, (width, height))
    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         frame = main(frame)
    #         # write the flipped frame
    #         # video.write(frame)
    #         cv2.imshow('frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    # # Release everything if job is finished
    # cap.release()
    # video.release()
    # cv2.destroyAllWindows()
