import sys
import xlrd
import xlwt

#from sklearn import preprocessing
# from sklearn.externals import joblib
import joblib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from Window_dujuan import Ui_MainWindow


#from pca_model import principal_component_analysis_model as pcam
#from pls_model_ import pls_model as plsm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import seaborn as sns
from datetime import datetime
import  argparse
import math

class FS_window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):#初始化，没有父类
        super(FS_window, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle('基于机器视觉的智慧农药化肥喷洒平台')#给一个窗口标题
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")
        self.train_data.triggered.connect(self.load_traindata)#链接训练数据
        self.test_data.triggered.connect(self.load_testdata)#链接测试数据，在界面左上角
        self.picture_data.triggered.connect(self.load_picturedata)#链接图片数据

        #按钮空间对应界面的不同页面
        self.button_reading.clicked.connect(self.topage_1)#一个按钮可以与一个页面（page）连接
        self.button_preprocessing.clicked.connect(self.topage_2)
        self.button_segemation.clicked.connect(self.topage_3)
        self.button_feature_extraction.clicked.connect(self.topage_4)
        self.button_classification.clicked.connect(self.topage_8)
        self.button_color_feature.clicked.connect(self.topage_5)
        self.button_shape_feature.clicked.connect(self.topage_6)
        self.button_texture_feature.clicked.connect(self.topage_7)

        #先让button灰调，也就是按钮按不了。等它前面的步骤都做完了，再亮回来。
        #self.button_preprocessing.setEnabled(False)
        #self.button_segemation.clicked.setEnabled(False)
        #self.button_feature_extraction.setEnabled(False)
        #self.button_classification.clicked.setEnabled(False)
        #self.button_color_feature.clicked.setEnabled(False)
        #self.button_shape_feature.clicked.setEnabled(False)
        #self.button_texture_feature.clicked.setEnabled(False)


        #按钮控件对应函数
        self.button_get_picture.clicked.connect(self.get_picture)#和后面的函数连接，也就是我们写的函数
        self.button_histogram_equalization.clicked.connect(self.histogram_equalization)
        self.button_color_segemation.clicked.connect(self.color_segemation)
        self.button_color_moment.clicked.connect(self.color_moment)#这写函数还未命名
        self.button_Hu_invariant_moment.clicked.connect(self.Hu_invariant_moment)
        self.button_gray_level_co_occurance_matrix.clicked.connect(self.gray_level_co_occurance_matrix)
        self.button_classifier.clicked.connect(self.adaboost_classifier)

        ## 画布——对应image(原图)
        self.fig_image = Figure((7, 5))  # 15, 8这里应该只确定了figsize
        self.canvas_image = FigureCanvas(self.fig_image)
        #self.canvas_pca.setParent(self.pca_gongxiantu)
        self.graphicscene_image = QGraphicsScene()
        self.graphicscene_image.addWidget(self.canvas_image)
        self.toolbar_image = NavigationToolbar(self.canvas_image, self.picture_dujuan_1)
        
        ## 画布——对应imageH（均衡化后的图）
        self.fig_imageH = Figure((7, 5))  # 15, 8这里应该只确定了figsize
        self.canvas_imageH = FigureCanvas(self.fig_imageH)
        #self.canvas_pca.setParent(self.pca_gongxiantu)
        self.graphicscene_imageH = QGraphicsScene()
        self.graphicscene_imageH.addWidget(self.canvas_imageH)
        self.toolbar_imageH = NavigationToolbar(self.canvas_imageH, self.picture_imageH)
        
        ## 画布——对应img_RGB
        self.fig_img_RGB = Figure((7, 5))  # 15, 8这里应该只确定了figsize
        self.canvas_img_RGB = FigureCanvas(self.fig_img_RGB)
        #self.canvas_pca.setParent(self.pca_gongxiantu)
        self.graphicscene_img_RGB = QGraphicsScene()
        self.graphicscene_img_RGB.addWidget(self.canvas_img_RGB)
        self.toolbar_img_RGB = NavigationToolbar(self.canvas_img_RGB, self.picture_img_RGB)



    #界面切换
    def topage_1(self):
        self.stackedWidget.setCurrentWidget(self.page_1)
    def topage_2(self):
        self.stackedWidget.setCurrentWidget(self.page_2)
    def topage_3(self):
        self.stackedWidget.setCurrentWidget(self.page_3)    
    def topage_4(self):
        self.stackedWidget.setCurrentWidget(self.page_4)
    def topage_5(self):
        self.stackedWidget_2.setCurrentWidget(self.page_5)
    def topage_6(self):
        self.stackedWidget_2.setCurrentWidget(self.page_6)
    def topage_7(self):
        self.stackedWidget_2.setCurrentWidget(self.page_7)    
    def topage_8(self):
        self.stackedWidget.setCurrentWidget(self.page_8)


    #导入训练数据
    def load_traindata(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "选择训练数据")
            table = xlrd.open_workbook(datafile).sheets()[0]
            nrows = table.nrows
            ncols = table.ncols
            self.trainWidget.setRowCount(nrows)#确定行数
            self.trainWidget.setColumnCount(ncols)#确定列数
            self.train_data = np.zeros((nrows, ncols))#设初始值，零矩阵
            self.dataArr = np.zeros((nrows, ncols-1))
            self.LabelArr = np.zeros((nrows, 1))

            for i in range(nrows):
                for j in range(ncols):
                    self.trainWidget.setItem(i, j, QTableWidgetItem(str(table.cell_value(i, j))))#这里的trainWidget是界面的东西
                    self.train_data[i, j] = table.cell_value(i, j)#把数据一个一个导入进去
            for i in range(nrows):
                for j in range(ncols-1):
                    self.dataArr[i,j]=self.train_data[i, j]
            for i in range(nrows): 
                self.LabelArr[i]=self.train_data[i, -1]
            #print(self.dataArr)
            #print(np.shape(self.dataArr))
            #print(self.LabelArr)
            #print(np.shape(self.LabelArr))
            #print(self.LabelArr.T)
            #print(np.shape(self.LabelArr.T))
            self.statusbar.showMessage('训练数据已导入')
        except:
            QMessageBox.information(self, 'Warning', '数据为EXCEL表格', QMessageBox.Ok)


    #导入测试数据
    def load_testdata(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "选择测试数据")
            table = xlrd.open_workbook(datafile).sheets()[0]
            nrows = table.nrows
            ncols = table.ncols
            self.testWidget.setRowCount(nrows)
            self.testWidget.setColumnCount(ncols)
            self.test_data = np.zeros((nrows, ncols))
            self.testArr = np.zeros((nrows, ncols-1))
            self.testLabelArr = np.zeros((nrows, 1))
            
            for i in range(nrows):
                for j in range(ncols):
                    self.testWidget.setItem(i, j, QTableWidgetItem(str(table.cell_value(i, j))))
                    self.test_data[i, j] = table.cell_value(i, j)
            self.statusbar.showMessage('测试数据已导入')
            for i in range(nrows):
                for j in range(ncols-1):
                    self.testArr[i,j]=self.test_data[i, j]
            for i in range(nrows): 
                self.testLabelArr[i]=self.test_data[i, -1]
            #print(self.testArr)
            #print(np.shape(self.testArr))
            #print(self.testLabelArr)
            #print(np.shape(self.testLabelArr))
            #print((np.shape(self.testLabelArr.T)))
        except:
            QMessageBox.information(self, 'Warning', '数据为EXCEL表格', QMessageBox.Ok)

    #选择图片
    def load_picturedata(self):
        try:
        #选择图片
            imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "img", "*.jpg;*.tif;*.png;;All Files(*)")
            if imgName=="":
                return 0
            #qt5读取图片
            #self.jpg = QPixmap(imgName).scaled(self.picture_dujuan_1.width(), self.picture_dujuan_1.height())
        except:
            QMessageBox.information(self, 'Warning', '导入失败', QMessageBox.Ok)


    def get_picture(self):
        try:
            img=cv2.imread("dujuan_1.jpg")
            #img=cv2.cvtColor(self.jpg,cv2.COLOR_RGB2BGR)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
            # 修改原图的尺寸
            fx = 0.3
            fy = 0.3
            self.image = cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation = cv2.INTER_AREA)
            
            self.fig_image.clear()
            plt = self.fig_image.add_subplot(111)
            plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            self.canvas_image.draw()
            self.picture_dujuan_1.setScene(self.graphicscene_image)
            self.picture_dujuan_1.show()
            #self.button_preprocessing.setEnabled(True)
        except:
            QMessageBox.information(self, 'Warning', '绘制图片的时候出错', QMessageBox.Ok)

    def histogram_equalization(self):
        (b, g, r) = cv2.split(self.image)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        self.imageH = cv2.merge((bH, gH, rH))
        self.fig_imageH.clear()
        plt = self.fig_imageH.add_subplot(111)
        plt.imshow(cv2.cvtColor(self.imageH, cv2.COLOR_BGR2RGB))
        self.canvas_imageH.draw()
        self.picture_imageH.setScene(self.graphicscene_imageH)
        self.picture_imageH.show()
        #self.button_segemation.clicked.setEnabled(True)

    def color_segemation(self):
        #基于H分量的双阈值分割

        HSV_img = cv2.cvtColor(self.imageH, cv2.COLOR_BGR2HSV)
        hue = HSV_img[:, :, 0]


        lower_gray = np.array([1, 0,0])
        upper_gray = np.array([99,255,255])


        mask = cv2.inRange(HSV_img, lower_gray, upper_gray)
        # Bitwise-AND mask and original image
        res2 = cv2.bitwise_and(self.imageH,self.imageH, mask= mask)

        # 先将这张图进行二值化

        ret1, thresh1 = cv2.threshold(res2,0, 255, cv2.THRESH_BINARY)

        # 然后是进行形态学操作
        # 我们采用矩形核（10，10）进行闭运算

        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel_1)

        # 对原图进行RGB三通道分离

        (B,G,R) = cv2.split(self.image)#之前得到的图

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
        self.result = cv2.bitwise_and(img_BGR,img_BGR, mask= mask)
        
        self.fig_img_RGB.clear()
        plt = self.fig_img_RGB.add_subplot(111)
        plt.imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
        self.canvas_img_RGB.draw()
        self.picture_img_RGB.setScene(self.graphicscene_img_RGB)
        self.picture_img_RGB.show()        
        #self.button_feature_extraction.setEnabled(True)
        #self.button_color_feature.clicked.setEnabled(True)
        #self.button_shape_feature.clicked.setEnabled(True)
        #self.button_texture_feature.clicked.setEnabled(True)


    def color_moment(self):
        try:
            # Convert BGR to HSV colorspace
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            # Split the channels - h,s,v
            h, s, v = cv2.split(hsv)
            # Initialize the color feature
            self.color_feature = []
            # N = h.shape[0] * h.shape[1]
            # The first central moment - average 
            h_mean = np.mean(h)  # np.sum(h)/float(N)
            s_mean = np.mean(s)  # np.sum(s)/float(N)
            v_mean = np.mean(v)  # np.sum(v)/float(N)
            self.color_feature.extend([h_mean, s_mean, v_mean])
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
            self.color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

                
            self.lineEdit_H_first.setText(str(color_feature[0]))
            self.lineEdit_H_second.setText(str(color_feature[1]))            
            self.lineEdit_H_third.setText(str(color_feature[2]))
            self.lineEdit_S_first.setText(str(color_feature[3]))  
            self.lineEdit_S_second.setText(str(color_feature[4]))    
            self.lineEdit_S_third.setText(str(color_feature[5]))
            self.lineEdit_V_first.setText(str(color_feature[6]))            
            self.lineEdit_V_second.setText(str(color_feature[7]))
            self.lineEdit_V_third.setText(str(color_feature[8])) 
        except:
            QMessageBox.information(self, 'Warning', '提取颜色矩出错', QMessageBox.Ok)        
        
    def Hu_invariant_moment(self):
        try:
            seg = self.result
            seg_gray = cv2.cvtColor(seg,cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(seg_gray)
            self.humoments = cv2.HuMoments(moments)
            self.humoments = np.log(np.abs(self.humoments)) # 同样建议取对数  
                
            self.fai_1.setText(str(humoments[0]))
            self.fai_2.setText(str(humoments[1]))         
            self.fai_3.setText(str(humoments[2]))
            self.fai_4.setText(str(humoments[3]))
            self.fai_5.setText(str(humoments[4]))   
            self.fai_6.setText(str(humoments[5]))
            self.fai_7.setText(str(humoments[6]))         


        except:
            QMessageBox.information(self, 'Warning', '提取颜色矩出错', QMessageBox.Ok)

            
    def gray_level_co_occurance_matrix(self):
        try:
            img_shape=self.result.shape

            resized_img=cv2.resize(self.result,(int(img_shape[1]/2),int(img_shape[0]/2)),interpolation=cv2.INTER_CUBIC)

            img_gray=cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            gray_level = 16

            
            #之前的getGlcm(self,img_gray,d_x,d_y)
            d_x=0
            d_y=1
            srcdata=img_gray.copy()
            p=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
            (height,width) = img_gray.shape
            
            #以前的maxGrayLevel(img)
            max_gray_level=0
            (height,width)=img_gray.shape
            #print(height,width)
            for y in range(height):
                for x in range(width):
                    if img_gray[y][x] > max_gray_level:
                        max_gray_level = img_gray[y][x]
                        
            max_gray_level=max_gray_level+1
    
            #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
            if max_gray_level > gray_level:
                for j in range(height):
                    for i in range(width):
                        srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

            for j in range(height-d_y):
                for i in range(width-d_x):
                    rows = srcdata[j][i]
                    cols = srcdata[j + d_y][i+d_x]
                    p[rows][cols]+=1.0

            for i in range(gray_level):
                for j in range(gray_level):
                    p[i][j]/=float(height*width)
            

            #之前的feature_computer()
            con=0.0
            eng=0.0
            asm=0.0
            idm=0.0
            for i in range(gray_level):
                for j in range(gray_level):
                    con+=(i-j)*(i-j)*p[i][j]
                    asm+=p[i][j]*p[i][j]
                    idm+=p[i][j]/(1+(i-j)*(i-j))
                    if p[i][j]>0.0:
                        eng-=p[i][j]*math.log(p[i][j])
            self.glcm=[asm,eng,con,idm]            
            self.energy_1.setText(str(asm))
            self.entrophy_1.setText(str(eng))         
            self.contrast_ratio.setText(str(con))
            self.inverse_variance.setText(str(idm))
        except:
            QMessageBox.information(self, 'Warning', '提取灰度共生矩阵出错', QMessageBox.Ok)

    def adaboost_classifier(self):
        try:
            weakClassArr = []
            m = np.shape(self.dataArr)[0]
            D = np.mat(np.ones((m, 1)) / m)
            #初始化权重
            #print(D)
            aggClassEst = np.mat(np.zeros((m,1)))
            numIt=4
            for i in range(numIt):
                #bestStump, error, classEst = buildStump(self.dataArr, self.LabelArr, D) 	#构建单层决策树

                dataMatrix = np.mat(self.dataArr); labelMat = np.mat(self.LabelArr)
                #print(dataMatrix)
                #print(labelMat)
                #print(type(dataMatrix))
                #print(type(labelMat))
                #print(np.shape(dataMatrix))
                #print(np.shape(labelMat))
                
                m,n = np.shape(dataMatrix)
                numSteps = 1.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
                minError = float('inf')
                #最小误差初始化为正无穷大
                #print("good")
                for j in range(n):															#遍历所有特征
                    rangeMin = dataMatrix[:,j].min(); rangeMax = dataMatrix[:,j].max()		#找到特征中最小的值和最大值
                    stepSize = (rangeMax - rangeMin) / numSteps
                    #计算步长
                    #print("goodjob")
                    for k in range(-1, int(numSteps) + 1):
                        for inequal in ['lt', 'gt']:
                            #print("good")
                            #大于和小于的情况，均遍历。lt:less than，gt:greater than
                            threshVal = (rangeMin + float(k) * stepSize) 					#计算阈值
                            #predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#计算分类结果
                            dimen=j
                            threshIneq=inequal
                            retArray = np.ones((np.shape(dataMatrix)[0],1))				#初始化retArray为1
                            if threshIneq == 'lt':
                                
                                retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#如果小于阈值,则赋值为-1
                            else:
                                retArray[dataMatrix[:,dimen] > threshVal] = -1.0
                                    
                            predictedVals=np.mat(retArray)
                            #print(predictedVals)
                            #print(type(predictedVals))
                            #print(np.shape(predictedVals))
                            
                            errArr = np.mat(np.ones((m,1)))#初始化误差矩阵
                            #print(np.shape(errArr))
                            
                            errArr[predictedVals == labelMat] = 0#分类正确的,赋值为0
                            #print(errArr)
                            #print(np.shape(errArr))
                            #print(D)
                            #print(np.shape(D))
                            weightedError = D.T * errArr#计算误差
                            #print(weightedError)
                            #print(np.shape(weightedError))
                            
                            # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                            if weightedError < minError:
                                #找到误差最小的分类方式
                                minError = weightedError
                                bestClasEst = predictedVals.copy()
                                bestStump['dim'] = j
                                bestStump['thresh'] = threshVal
                                bestStump['ineq'] = inequal
                        
                #print("good")
                #print(bestStump)
                error=minError
                #print(minError)
                classEst=bestClasEst
                #print(bestClasEst)
                alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
                #计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
                #print(alpha)
                bestStump['alpha'] = alpha  										#存储弱学习算法权重 
                weakClassArr.append(bestStump)
                #存储单层决策树
                #print(weakClassArr)
                # print("classEst: ", classEst.T)
                expon = np.multiply(-1 * alpha * np.mat(self.LabelArr), classEst) 	#计算e的指数项
                #print(np.shape(expon))
                D = np.multiply(D, np.exp(expon))                           		   
                D = D / D.sum()
                #print(np.shape(D))#根据样本权重公式，更新样本权重
                #计算AdaBoost误差，当误差为0的时候，退出循环
                aggClassEst += alpha * classEst  									#计算类别估计累计值								
                # print("aggClassEst: ", aggClassEst.T)
                aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(self.LabelArr), np.ones((m,1))) 	#计算误差
                errorRate = aggErrors.sum() / m
                #print(errorRate)
                # print("total error: ", errorRate)
                if errorRate == 0.0: break 	

            print("good")
            #predictions = adaClassify(dataArr, weakClassArr)
            datToClass=self.dataArr
            print("good")
            classifierArr=weakClassArr
            print("good")
            dataMatrix = np.mat(datToClass)
            print("good")
            m = np.shape(dataMatrix)[0]
            print("good")
            aggClassEst = np.mat(np.zeros((m,1)))
            print("good")
            for i in range(len(classifierArr)):										#遍历所有分类器，进行分类
                #classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
                #stumpClassify(dataMatrix,dimen,threshVal,threshIneq)
                dimen=classifierArr[i]['dim']
                threshVal=classifierArr[i]['thresh']
                threshIneq=classifierArr[i]['ineq']
                retArray = np.ones((np.shape(dataMatrix)[0],1))				#初始化retArray为1
                if threshIneq == 'lt':
                        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#如果小于阈值,则赋值为-1
                else:
                        retArray[dataMatrix[:,dimen] > threshVal] = -1.0 		#如果大于阈值,则赋值为-1
                classEst=retArray
                aggClassEst += classifierArr[i]['alpha'] * classEst
                # print(aggClassEst)
            print("good")
            predictions=np.sign(aggClassEst)
            print("good")
            errArr = np.mat(np.ones((len(self.dataArr), 1)))
            print("good")
            print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(self.LabelArr)].sum() / len(self.dataArr) * 100))
            #predictions = adaClassify(testArr, weakClassArr)
            datToClass=self.testArr
            classifierArr=weakClassArr      
            dataMatrix = np.mat(datToClass)
            m = np.shape(dataMatrix)[0]
            aggClassEst = np.mat(np.zeros((m,1)))
            for i in range(len(classifierArr)):										#遍历所有分类器，进行分类
                #classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
                dimen=classifierArr[i]['dim']
                threshVal=classifierArr[i]['thresh']
                threshIneq=classifierArr[i]['ineq']
                retArray = np.ones((np.shape(dataMatrix)[0],1))				#初始化retArray为1
                if threshIneq == 'lt':
                        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#如果小于阈值,则赋值为-1
                else:
                        retArray[dataMatrix[:,dimen] > threshVal] = -1.0 		#如果大于阈值,则赋值为-1
                classEst=retArray                
                aggClassEst += classifierArr[i]['alpha'] * classEst
                # print(aggClassEst)
            predictions=np.sign(aggClassEst)
            
            errArr = np.mat(np.ones((len(self.testArr), 1)))
            print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(self.testLabelArr)].sum() / len(self.testArr) * 100))
####################################################################################################################################################
            test=[]
            test.extend(self.color_feature)
            test.extend(self.humoments)
            test.extend(self.glcm)
            datToClass=self.test
            classifierArr=weakClassArr      
            dataMatrix = np.mat(datToClass)
            m = np.shape(dataMatrix)[0]
            aggClassEst = np.mat(np.zeros((m,1)))
            for i in range(len(classifierArr)):										#遍历所有分类器，进行分类
                #classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
                dimen=classifierArr[i]['dim']
                threshVal=classifierArr[i]['thresh']
                threshIneq=classifierArr[i]['ineq']
                retArray = np.ones((np.shape(dataMatrix)[0],1))				#初始化retArray为1
                if threshIneq == 'lt':
                        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#如果小于阈值,则赋值为-1
                else:
                        retArray[dataMatrix[:,dimen] > threshVal] = -1.0 		#如果大于阈值,则赋值为-1
                classEst=retArray                
                aggClassEst += classifierArr[i]['alpha'] * classEst
                # print(aggClassEst)
            predictions=np.sign(aggClassEst)
            self.classification_result.setText(str(predictions))        
        except:
            QMessageBox.information(self, 'Warning', '构建分类器出错', QMessageBox.Ok)        
        
#这是运行的代码
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FS_window()
    win.show()
    sys.exit(app.exec_())

































