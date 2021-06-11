import numpy as np
import cv2

# 去除文字信息
def move_characters(img):
    ret, mask1 = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow('mask of characters', cv2.WINDOW_NORMAL)
    # cv2.imshow('mask of characters', mask1)

    dst = cv2.inpaint(img, mask1, 20, cv2.INPAINT_NS)
    # cv2.namedWindow('Characters removed', cv2.WINDOW_NORMAL)
    # cv2.imshow('Characters removed', dst)

    return dst

# 提取所有贝壳
def get_shell(img):
    # 去除文字干扰
    ret, mask_of_characters = cv2.threshold(img, 190, 255, cv2.THRESH_TOZERO_INV)
    # cv2.namedWindow('mask_of_characters',cv2.WINDOW_NORMAL)
    # cv2.imshow('mask_of_characters', mask_of_characters)
    # 中值滤波
    median_blur = cv2.medianBlur(mask_of_characters,7)
    # cv2.namedWindow('median_blur',cv2.WINDOW_NORMAL)
    # cv2.imshow('median_blur', median_blur)

    # if img[0:1000,0:1000].sum() > 100000000:
    # 提取贝壳掩膜
    ret1, mask_of_shell = cv2.threshold(median_blur, 55, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow('mask_of_shell',cv2.WINDOW_NORMAL)
    # cv2.imshow('mask_of_shell', mask_of_shell)
    # 均值滤波
    blur = cv2.blur(mask_of_shell, (7, 7))
    # cv2.namedWindow('blur', cv2.WINDOW_NORMAL)
    # cv2.imshow('blur', blur)

    ret2, mask1 = cv2.threshold(blur, 190, 255, cv2.THRESH_TOZERO)
    # cv2.namedWindow('mask1', cv2.WINDOW_NORMAL)
    # cv2.imshow('mask1', mask1)

    for i in range(2):  # 循环滤波三次，尽可能的去除噪音
        blur1 = cv2.blur(mask1,(7,7))
        # cv2.namedWindow('blur1',cv2.WINDOW_NORMAL)
        # cv2.imshow('blur1', blur1)
        ret3, mask1 = cv2.threshold(blur1, 190,255 , cv2.THRESH_TOZERO)
        # cv2.namedWindow('mask2',cv2.WINDOW_NORMAL)
        # cv2.imshow('mask2', mask1)

    # 腐蚀 将沙砾去除留下贝壳的掩膜
    kernel = np.ones((7,7),np.uint8)
    erosion = cv2.erode(mask1,kernel,iterations = 1) # 腐蚀两次为了去掉中间的噪点
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    # cv2.namedWindow('after_erosion',cv2.WINDOW_NORMAL)
    # cv2.imshow('after_erosion', erosion)
    kernel1 = np.ones((7,7),np.uint8)
    dilation = cv2.dilate(erosion,kernel1,iterations = 1)# 膨胀三次为了填补中间腐蚀出来的黑洞
    dilation = cv2.dilate(dilation,kernel1,iterations = 1)
    dilation = cv2.dilate(dilation,kernel1,iterations = 1)
    # cv2.namedWindow('after_dilate',cv2.WINDOW_NORMAL)
    # cv2.imshow('after_dilate', dilation)

    # 提取贝壳
    shells = cv2.bitwise_and(img, img, mask=dilation)
    # cv2.imwrite('shells.jpg',shells)
    # cv2.namedWindow('shells',cv2.WINDOW_NORMAL)
    # cv2.imshow('shells', shells)
    # cv2.waitKey(0)
    # else:
    #     pass

    return shells,dilation

# 提取沙子
def get_sand(img):

    blur = cv2.blur(img, (7, 7))
    # cv2.namedWindow('blur', cv2.WINDOW_NORMAL)
    # cv2.imshow('blur', blur)

    ret, mask = cv2.threshold(blur, 130, 255, cv2.THRESH_TOZERO_INV)
    ret, mask = cv2.threshold(mask, 125, 255, cv2.THRESH_TOZERO)
    # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    # cv2.imshow('mask', mask)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    # erosion = cv2.erode(erosion, kernel, iterations=1)
    # cv2.namedWindow('after erosion', cv2.WINDOW_NORMAL)
    # cv2.imshow('after erosion', erosion)


    # 增大特征的对比
    equalizeHist = cv2.equalizeHist(erosion)
    # cv2.namedWindow('equalizeHist', cv2.WINDOW_NORMAL)
    # cv2.imshow('equalizeHist', equalizeHist)

    return equalizeHist

# 分割出每个贝壳
def every_shell(mask):
    canny_output = cv2.Canny(mask, 10, 200, 3)
    img_edge,contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.namedWindow('img_edge', cv2.WINDOW_NORMAL)
    # cv2.imshow('img_edge', img_edge)

    print(len(contours))  # 六个边缘
    print(len(hierarchy))
    print(contours[0]) # 第0个边缘的坐标信息

    shell_count = cv2.drawContours(img, contours[0], -1, (0, 255, 255), 3)
    # cv2.namedWindow('shell_count', cv2.WINDOW_NORMAL)
    # cv2.imshow('shell_count', shell_count)

    h,w = mask.shape
    shells=[]
    for i in range(len(contours)):
        label = np.zeros((h, w),dtype='uint8')
        shell = cv2.fillPoly(label, [contours[i]], 255)
        shells.append(shell)
        # cv2.namedWindow('shell_contours', cv2.WINDOW_NORMAL)
        # cv2.imshow('shell_contours', shell)
        # cv2.waitKey(1000)

    return shells

if __name__=='__main__':
    import os

    ''' step 01 去除所有图片的文字'''
    for filename in os.listdir(r"./image"):  # listdir的参数是文件夹的路径
        print(filename)

        # 读入图片
        img = cv2.imread('image/'+str(filename))
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)

        # 灰度化图片
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # 去除文字
        img_charactermoved = move_characters(img_gray)
        cv2.namedWindow('img_charactermoved', cv2.WINDOW_NORMAL)
        cv2.imshow('img_charactermoved', img_charactermoved)
        cv2.imwrite('character_removed/'+ str(filename),img_charactermoved)
        cv2.waitKey(1000)

    ''' step 02 提取每张图片的所有贝壳'''
    for filename in os.listdir(r"./character_removed"):  # listdir的参数是文件夹的路径
        print(filename)

        img = cv2.imread('character_removed/' + str(filename))
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # 提取出所有贝壳
        shells,mask = get_shell(img_gray)
        cv2.namedWindow('shells', cv2.WINDOW_NORMAL)
        cv2.imshow('shells', shells)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.imshow('mask', mask)
        cv2.imwrite('shells/'+ str(filename),mask)
        cv2.waitKey(1000)

        # WeChat Image_20200221164050.jpg 单张图片曝光过高，效果不好，所以去掉


    ''' step 03 分别提取图片中的每一个贝壳并对其提取沙子特征、计算特征点数量'''
    for filename in os.listdir(r"./shells"):  # listdir的参数是文件夹的路径
        print(filename)

        img = cv2.imread('image/' + str(filename),0)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)


        img_mask = cv2.imread('shells/' + str(filename),0)
        cv2.namedWindow('img_mask', cv2.WINDOW_NORMAL)
        cv2.imshow('img_mask', img_mask)

        # 分割出每一个贝壳
        all_shells = every_shell(img_mask)
        print(len(all_shells)) # 贝壳数量
        # 提取特征
        for i in range(len(all_shells)):
            shell = cv2.bitwise_and(img, img, mask=all_shells[i])
            cv2.namedWindow('shell', cv2.WINDOW_NORMAL)
            cv2.imshow('shell', shell)
            cv2.imwrite('./output/'+str(filename.split('.jpg')[0])+'/'+str(i)+'.jpg',shell)
            cv2.waitKey(1000)
            shell_sand = get_sand(shell)
            # SIFT提特征r
            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(shell_sand,None)
            kp,des = sift.compute(shell_sand,kp)
            print(len(kp))  # 特征点数量
            output = cv2.drawKeypoints(shell,kp,shell_sand)
            output = cv2.putText(output, str(len(kp)),(200,100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            cv2.namedWindow('output',cv2.WINDOW_NORMAL)
            cv2.imshow('output',output)
            cv2.imwrite('./output/'+str(filename.split('.jpg')[0])+'/'+str(i)+'_feature.jpg',output)
            cv2.waitKey(1000)




