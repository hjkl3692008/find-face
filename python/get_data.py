# -*- coding:UTF-8 -*-
import cv2 as cv
import random
import glob
import os
import re
 
#加载负样本
#folder = 'C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm'
#savePath = 'C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm/Img/neg_250x250/'
def get_neg_samples(foldername,savePath):
    count = 0
    imgs = []
    labels = []
    f = open('neg.txt','w')
    # 查找到两层之下的所有jpg文件
    filenames = glob.iglob(os.path.join(foldername,'./Img/sample-neg/*/*.jpg'))
    for filename in filenames:
        # print('filename = ',filename)

        #-1：imread按解码得到的方式读入图像
        #0：imread按单通道的方式读入图像，即灰白图像
        #1：imread按三通道方式读入图像，即彩色图像
        src = cv.imread(filename,1)
        
        # 随机生成坏取样
        #****这里可能存在问题******
        # 还真的有问题：src is 'numpy.ndarray' which has no attribute 'cols'
        # y:shape[0]   x:shape[1]
        try:
            if((src.shape[1] >= 250) & (src.shape[0] >= 250)):
                x = int(random.uniform(0,src.shape[1] - 250))
                y = int(random.uniform(0,src.shape[0] - 250))
                print('src.shape:{}'.format(src.shape))
                print('x:{}'.format(x))
                print('y:{}'.format(y))
                # ***********************************************************************
                # 这里也存在问题 ：cv2.cv2' has no attribute 'Rect'
                # imgRoi = src(cv.Rect(x,y,250,250))
                #在这里就直接截取250^2像素
                imgRoi = src[y:y+250,x:x+250]
                print('imgRoi.shape:{}'.format(imgRoi.shape))
                imgs.append(imgRoi)
                # 图片路径
                saveName = savePath + 'neg' + str(count) + '.jpg'
                # 保存图片
                cv.imwrite(saveName,imgRoi)
                
                # 某文件夹下面所有的图片共用一个标签
                label = 'neg' + re.findall(r'.*\\(.*?)\\.*.jpg',filename)[0]
                if labels.count(label) == 0:
                    labels.append(label)
                label = saveName + ' '+ label +'\n'
                f.write(label)
                count += 1
        except Exception as e:
            print(e)
            print('读取文件错误filename = ',filename)
            continue
    print('处理负样本完成')
    return imgs,labels
 
 
#读取负样本
def read_neg_samples(foldername):
    imgs = []
    labels = []
    neg_count = 0;
    filenames = glob.iglob(os.path.join(foldername,'*'))
    for filename in filenames:
        src = cv.imread(filename,1)
        imgs.append(src)
         #负样本全部打上标签-1
        labels.append(-1)
        neg_count += 1
    return imgs,labels
        
        
 
#加载正样本
#folder:C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm
#savePath:C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm/Img/pos_250x250/
def get_pos_samples(foldername,savePath):
    count = 0
    imgs = []
    labels = []
    f = open('pos.txt','w')
    # 查找到两层之下的所有jpg文件
    filenames = glob.iglob(os.path.join(foldername,'./Img/sample-pos/*/*.jpg'))
    for filename in filenames:
        try:
            # print('filename = ',filename)
            src = cv.imread(filename)
            # imgRoi = src(cv.Rect(0,0,250,250))
            imgRoi  = src[0:250,0:250]
            imgs.append(imgRoi)
            saveName = savePath + 'pos' + str(count) + '.jpg'
            cv.imwrite(saveName,imgRoi)
            
            label = 'pos' + re.findall(r'.*\\(.*?)\\.*.jpg',filename)[0]
            if labels.count(label) == 0:
                    labels.append(label)
            label = saveName + ' '+ label +'\n'
            f.write(label)
            count += 1
        except Exception as e:
            print(e)
            print('读取文件错误filename = ',filename)
    print('处理正样本完成')
    return imgs,labels

#folder:C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm
#savePath:C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm/Img/neg/
def get_neg_hard_samples(foldername,savePath):
    count = 0
    imgs = []
    labels = []
    f = open('neg_hard.txt','w')
    # 查找到两层之下的所有jpg文件
    filenames = glob.iglob(os.path.join(foldername,'./Img/sample-neg/*/*.jpg'))
    for filename in filenames:
        try:
            # print('filename = ',filename)
            src = cv.imread(filename)
            if((src.shape[1] >= 500) & (src.shape[0] >= 500)):
                x = int(random.uniform(0,src.shape[1] - 500))
                y = int(random.uniform(0,src.shape[0] - 500))
                imgRoi  = src[y:y+500,x:x+500]
                imgs.append(imgRoi)
                saveName = savePath + 'neg' + str(count) + '.jpg'
                cv.imwrite(saveName,imgRoi)
                
                label = 'neg' + re.findall(r'.*\\(.*?)\\.*.jpg',filename)[0]
                if labels.count(label) == 0:
                        labels.append(label)
                label = saveName + ' '+ label +'\n'
                f.write(label)
                count += 1
        except Exception as e:
            print(e)
            print('读取文件错误filename = ',filename)
    print('处理强负样本完成')
    return imgs,labels
 
 
#读取正样本
def read_pos_samples(foldername):
    imgs = []
    labels = []
    pos_count = 0
    filenames = glob.iglob(os.path.join(foldername,'*'))
    
    for filename in filenames:
        src = cv.imread(filename)
        imgs.append(src)
        #正样本全部打上标签1
        labels.append(1)
        pos_count += 1
    
    return imgs,labels
