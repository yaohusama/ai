#coding=utf-8
import cv2
import numpy as np
from cnocr import CnOcr

def get_horizontal_projection(image):
    '''
    统计图片水平位置白色像素的个数
    '''
    #图像高与宽
    height_image, width_image = image.shape 
    height_projection = [0]*height_image
    for height in range(height_image):
        for width in range(width_image):
            if image[height, width] == 255:
                height_projection[height] += 1
    return height_projection

def get_vertical_projection(image): 
    '''
    统计图片垂直位置白色像素的个数
    '''
    #图像高与宽
    height_image, width_image = image.shape 
    width_projection = [0]*width_image
    for width in range(width_image):
        for height in range(height_image):
            if image[height, width] == 255:
                width_projection[width] += 1
    return width_projection

def get_text_lines(projections):
    text_lines = []
    start = 0
    for index, projection in enumerate(projections):
        if projection>0 and start==0:
            start_location = index
            start = 1
        if projection==0 and start==1:
            end_location = index
            start = 0
            text_lines.append((start_location,end_location))
    return text_lines

def get_text_word(projections):
    text_word = [ ]
    start = 0
    for index, projection in enumerate(projections):
        if projection>0 and start==0:
            start_location = index
            start = 1
        if projection==0 and start==1:
            end_location = index
            start = 0
            if len(text_word)>0 and start_location-text_word[-1][1]<3:
                text_word[-1] = (text_word[-1][0],end_location)
            else:
                text_word.append((start_location,end_location))
    return text_word  

def orc_text(path_image):
    ocr = CnOcr()
    image = cv2.imread(path_image,cv2.IMREAD_GRAYSCALE)
    height_image, width_image = image.shape
    _, binary_image = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)   
    height_projection = get_horizontal_projection(binary_image)
    text_lines = get_text_lines(height_projection)
    text_list = []
    for line_index, text_line in enumerate(text_lines):
        text_line_image = binary_image[text_line[0]:text_line[1], 0:width_image]
        vertical_projection = get_vertical_projection(text_line_image)
        text_words = get_text_word(vertical_projection)
        text_line_word_image = image[text_line[0]:text_line[1], text_words[0][0]:text_words[-1][1]]         
        res = ocr.ocr_for_single_line(text_line_word_image) 
        text_list.append(''.join(res))
    return text_list

if __name__ == '__main__':
    ocr = CnOcr()
    image = cv2.imread(r'try.png',cv2.IMREAD_GRAYSCALE)
    print(image)
    print(image.shape)
    cv2.imshow('gray_image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    height_image, width_image = image.shape
    _, binary_image = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('binary_image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    height_projection = get_horizontal_projection(binary_image)
    text_lines = get_text_lines(height_projection)
    for line_index, text_line in enumerate(text_lines):
        text_line_image = binary_image[text_line[0]:text_line[1], 0:width_image]
        vertical_projection = get_vertical_projection(text_line_image)
        text_words = get_text_word(vertical_projection)
        text_line_word_image = image[text_line[0]:text_line[1], text_words[0][0]:text_words[-1][1]]  
        cv2.imshow('text_line_word_image', text_line_word_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
        res = ocr.ocr_for_single_line(text_line_word_image) 
        print(''.join(res))