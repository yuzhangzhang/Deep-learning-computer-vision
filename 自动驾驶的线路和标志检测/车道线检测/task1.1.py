import numpy as np
import cv2
import os
import argparse
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def roi(image, roi_range):
    mask = np.zeros_like(image)
    imshape = image.shape
    if len(image.shape) > 2:
        channel_out = image.shape[2]
        cv2.fillPoly(mask, roi_range, (255,) * channel_out)
    else:
        cv2.fillPoly(mask, roi_range, 255)

    img_masked = cv2.bitwise_and(image, mask)

    return img_masked



def process_line(image, lines):
    positive_slop_points = []  # positive 右
    negative_slop_points = []  # negtive  左
    positive_slop_intercept = []
    negative_slop_intercept = []
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    if lines is None:
        return line_img
    for line in lines:
        for x1, y1, x2, y2 in line:
            slop = np.float((y2 - y1)) / np.float((x2 - x1))
            if slop > 0:
                positive_slop_points.append([x1, y1])
                positive_slop_points.append([x2, y2])
                positive_slop_intercept.append([slop, y1 - slop * x1])
            elif slop < 0:
                negative_slop_points.append([x1, y1])
                negative_slop_points.append([x2, y2])
                negative_slop_intercept.append([slop, y1 - slop * x1])
    if (len(positive_slop_points) == 0 or len(negative_slop_points) == 0):
        return line_img
    positive_slop, positive_intercept = filter_line(positive_slop_intercept)
    negative_slop, negative_intercept = filter_line(negative_slop_intercept)
    ymin = 325
    ymax = image.shape[0]

    draw_line(line_img, positive_slop, positive_intercept, ymin, ymax)
    draw_line(line_img, negative_slop, negative_intercept, ymin, ymax)
    return line_img


def draw_line(image, slop, intercept, ymin, ymax):
    xmin = int((ymin - intercept) / slop)
    xmax = int((ymax - intercept) / slop)
    cv2.line(image, (xmin, ymin), (xmax, ymax), [255, 0, 0], 5)


def filter_line(slop_intercept):
    legal_slop = []
    legal_intercept = []
    slopes = [pair[0] for pair in slop_intercept]
    slop_mean = np.mean(slopes)
    slop_std = np.std(slopes)
    for pair in slop_intercept:
        if pair[0] - slop_mean < 3 * slop_std:
            legal_slop.append(pair[0])
            legal_intercept.append(pair[1])
    if not legal_slop:
        legal_slop = slopes
        legal_intercept = [pair[1] for pair in slop_intercept]
    slop = np.mean(legal_slop)
    intercept = np.mean(legal_intercept)
    return slop, intercept


def process_picture(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img_gray',img_gray)
    # cv2.imwrite('adjust/solidWhiteRight_test/img_gray.jpg',img_gray)
    # cv2.waitKey(5000)

    blur_image = cv2.GaussianBlur(img_gray, (5,5), 0)
    # cv2.imwrite('adjust/solidWhiteRight_test/blur_image.jpg',blur_image)
    # cv2.imshow('blur_image', blur_image)
    # cv2.waitKey(5000)

    canny_image = cv2.Canny(blur_image, 250, 300)
    # cv2.imwrite('adjust/solidWhiteRight_test/canny_image.jpg',canny_image)
    # cv2.imshow('canny_image', canny_image)
    # cv2.waitKey(5000)

    img_shape = canny_image.shape
    roi_range = np.array([[(0, img_shape[0]), (465, 320), (475, 320), (img_shape[1], img_shape[0])]], dtype=np.int32)
    img_masked = roi(canny_image, roi_range)
    # cv2.imwrite('adjust/solidWhiteRight_test/img_masked.jpg',img_masked)
    # cv2.imshow('img_masked', img_masked)
    # cv2.waitKey(5000)

    lines = cv2.HoughLinesP(img_masked, rho=1, theta=np.pi / 180, threshold=15,lines=np.array([]), minLineLength=25, maxLineGap=20)
    line_image = process_line(img_masked, lines)
    # cv2.imwrite('adjust/solidWhiteRight_test/line_image.jpg',line_image)
    # cv2.imshow('line_image', line_image)
    # cv2.waitKey(5000)

    res_image = cv2.addWeighted(image, 0.7, line_image, 1, 0)
    # cv2.imwrite('adjust/solidWhiteRight_test/res_image.jpg',res_image)
    # cv2.imshow('res_image', res_image)
    # cv2.waitKey(5000)
    return res_image


def main(config):
    if config.mode == 'picture':
        for filename in os.listdir(config.picture_dir):
            print(filename)
            img = cv2.imread(config.picture_dir+'/'+filename)
            res_img = process_picture(img)
            cv2.imshow('res_img',res_img)
            cv2.waitKey(1000)

    if config.mode == 'video':
        cap = cv2.VideoCapture(config.video_path)
        print('cap.isOpened(){}'.format(cap.isOpened()))
        while (cap.isOpened()):
            # get a frame
            ret, frame = cap.read()
            # show a frame
            if ret == True:
                res_img = process_picture(frame)
                cv2.imshow('res_img', res_img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    print('exit')
                    break
            else:
                cap.release()
                cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()

        video = VideoFileClip(config.video_path)
        video_after_process = video.fl_image(process_picture)
        video_after_process.write_videofile("./output.mp4", audio=False)
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='video', choices=['picture', 'video'])
    parser.add_argument('--video_path', type=str, default='solidWhiteRight.mp4',choices=['solidWhiteRight.mp4', 'challenge.mp4'])
    parser.add_argument('--picture_dir', type=str, default='./test_img')
    config = parser.parse_args()
    print(config)
    main(config)
    # img = cv2.imread('test_img/solidWhiteRight.jpg')
    # res_img = process_picture(img)