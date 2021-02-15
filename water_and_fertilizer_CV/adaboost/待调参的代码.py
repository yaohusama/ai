import cv2
import math
# 导入所需要的库：
import numpy as np
from datetime import datetime
import  argparse
import matplotlib.pyplot as plt


# 图像的读取
img = cv2.imread("dujuan_1.jpg") 

# 灰度图像的读取
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

# 修改原图的尺寸
fx = 0.3
fy = 0.3
image = cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation = cv2.INTER_AREA)

## 图像预处理部分

# 直方图均衡化
(b, g, r) = cv2.split(image)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
imageH = cv2.merge((bH, gH, rH))

## 图像分割

#基于H分量的双阈值分割

HSV_img = cv2.cvtColor(imageH, cv2.COLOR_BGR2HSV)
hue = HSV_img[:, :, 0]


lower_gray = np.array([1, 0,0])
upper_gray = np.array([99,255,255])


mask = cv2.inRange(HSV_img, lower_gray, upper_gray)
# Bitwise-AND mask and original image
res2 = cv2.bitwise_and(imageH,imageH, mask= mask)

# 先将这张图进行二值化

ret1, thresh1 = cv2.threshold(res2,0, 255, cv2.THRESH_BINARY)

# 然后是进行形态学操作
# 我们采用矩形核（10，10）进行闭运算

kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel_1)

# 对原图进行RGB三通道分离

(B,G,R) = cv2.split(image)#之前得到的图

# 三个通道分别和closing这个模板进行与运算

mask=cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)

and_img_B = cv2.bitwise_and(B,mask)
and_img_G = cv2.bitwise_and(G,mask)
and_img_R = cv2.bitwise_and(R,mask)

# 多通道图像进行混合

zeros = np.zeros(res2.shape[:2], np.uint8)

img_RGB = cv2.merge([and_img_R,and_img_G,and_img_B])

# 下面我先用closing的结果，进一步进行处理

# 这个颜色空间转来转去的，要小心

img_BGR=cv2.cvtColor(img_RGB,cv2.COLOR_RGB2BGR) #这个颜色空间转来转去的，要小心
HSV_img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
hue = HSV_img[:, :, 0]


lower_gray = np.array([1, 0,0])
upper_gray = np.array([99,255,255])


mask = cv2.inRange(HSV_img, lower_gray, upper_gray)
# Bitwise-AND mask and original image
result = cv2.bitwise_and(img_BGR,img_BGR, mask= mask)

## 特征提取

# hu 不变矩

seg = result
seg_gray = cv2.cvtColor(seg,cv2.COLOR_BGR2GRAY)
moments = cv2.moments(seg_gray)
humoments = cv2.HuMoments(moments)
humoments = np.log(np.abs(humoments)) # 同样建议取对数
print(humoments)

# 纹理特征&灰度共生矩阵

#定义最大灰度级数
gray_level = 16

def maxGrayLevel(img):
    max_gray_level=0
    (height,width)=img.shape
    #print(height,width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level+1

def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = input.shape
    
    max_gray_level=maxGrayLevel(input)
    
    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height-d_y):
        for i in range(width-d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i+d_x]
            ret[rows][cols]+=1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j]/=float(height*width)

    return ret

def feature_computer(p):
    Con=0.0
    Eng=0.0
    Asm=0.0
    Idm=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            Asm+=p[i][j]*p[i][j]
            Idm+=p[i][j]/(1+(i-j)*(i-j))
            if p[i][j]>0.0:
                Eng+=p[i][j]*math.log(p[i][j])
    return Asm,Con,-Eng,Idm

def test(img):
    try:
        img_shape=img.shape
    except:
        print ('imread error')
        return

    img=cv2.resize(img,(int(img_shape[1]/2),int(img_shape[0]/2)),interpolation=cv2.INTER_CUBIC)

    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0=getGlcm(img_gray, 1,0)
    #glcm_1=getGlcm(src_gray, 0,1)
    #glcm_2=getGlcm(src_gray, 1,1)
    #glcm_3=getGlcm(src_gray, -1,1)

    asm,con,eng,idm=feature_computer(glcm_0)

    return [asm,con,eng,idm]

result= test(seg)
print(result)

# 颜色矩&颜色特征

def color_moments(img):
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average 
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature

yanse_ju=color_moments(seg)
print(yanse_ju)





















































































