# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
 
from get_data import read_pos_samples,read_neg_samples
from svm_train import svm_config,svm_train,svm_save,svm_load
 
#计算hog特征
def computeHog(imgs,features,wsize = (250,250)):
    # 这里默认为HOGDescriptor myHOG(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9)
    # hog = cv.HOGDescriptor()
    # 第一个 窗口 大小 设置 为 上面的图片大小 64 x 64 。 
    # 第二个 块大小 是 16 x 16 的话 
    # 第三个 块block的步进 stride 8 x 8
    # 第四个是 胞元cell大小 8 x 8
    # 第五个是 cell的直方图的 bin = 9 每个 cell 有 9 个向量
    # 每个block 有 (16 / 8 ) * (16 / 8) = 2 * 2 = 4 个 cell， 那么现在就有 4 * 9 = 36 个向量啦
    # 窗口 有多少个 block：(window_size - block_size)/block_stride + 1  对两个方向进行计算：
    # ( 64 - 16) / 8 + 1 = 7
    # 两个方向  7 * 7 = 49
    # 共有  49* 36 = 1764
    hog = cv.HOGDescriptor((250,250),(25,25),(5,5),(5,5),9)
    count = 0
    
    for i in range(len(imgs)):
        # 仅使用图片尺寸大于250的图，并截取250*250的尺寸
        if imgs[i].shape[1] >= wsize[1] and imgs[i].shape[0] >= wsize[0]:
            y = imgs[i].shape[0] - wsize[0]
            x = imgs[i].shape[1] - wsize[1]
            h = imgs[i].shape[0]
            w = imgs[i].shape[1]
            roi = imgs[i][y : y + h, x : x + w]
            features.append(hog.compute(roi))
            count += 1
    
    print ('count = ',count)
    return features
 
#获取svm参数
def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv,[[-rho]],0)        
 
#加载hardexample
def get_hard_samples(svm,hog_features,labels):
    #hog = cv.HOGDescriptor()
    hog = cv.HOGDescriptor((250,250),(25,25),(5,5),(5,5),9)

    hard_examples = []
    hog.setSVMDetector(get_svm_detector(svm))
    negs,hardlabel= read_neg_samples('C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm/Img/neg/')
    
    for i in range(len(negs)):
        rects,wei = hog.detectMultiScale(negs[i],0,winStride = (5,5),padding = (0,0),scale = 1.05)
        for (x,y,w,h) in rects:
            hardexample = negs[i][y : y + h, x : x + w]
            hard_examples.append(cv.resize(hardexample,(250,250)))
            
    computeHog(hard_examples,hog_features)
    [labels.append(-1) for _ in range(len(hard_examples))]
    svm_train(svm,hog_features,labels)
    hog.setSVMDetector(get_svm_detector(svm))
    print ('save myHogDector1.bin...')
    hog.save('myHogDector1.bin')
    print ('save myHogDector1.bin complete...')
    #svm.train(np.array(hog_features),cv.ml.ROW_SAMPLE,np.array(labels))

#获取所有的hog特征
def get_features(features,labels):
    print('get pos hog...')
    pos_imgs,pos_labels = read_pos_samples('C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm/Img/pos_250x250/')
    print('compute pos hog...')
    computeHog(pos_imgs,features)
    print('append pos labels...')
    [labels.append(1) for _ in range(len(pos_imgs))]
    
    print('get neg hog...')
    neg_imgs,neg_labels = read_neg_samples('C:/Users/fyyk_whua/Desktop/python demo/MachineLearn/hog+svm/Img/neg_250x250/')
    print('compute neg hog...')
    computeHog(neg_imgs,features)
    print('append neg labels...')
    [labels.append(-1) for _ in range(len(neg_imgs))]

    return features,labels
 
#hog训练
def hog_train(svm):
    features = []
    labels = []
    
    # hog = cv.HOGDescriptor()
    hog = cv.HOGDescriptor((250,250),(25,25),(5,5),(5,5),9)
    
    #get hog features
    print ('get hog features...')
    get_features(features,labels)
    print ('get hog features complete...')
    
    #svm training
    print ('svm training...')
    svm_train(svm,features,labels)
    print ('svm training complete...')
    
    hog.setSVMDetector(get_svm_detector(svm))
    print ('save myHogDector.bin...')
    hog.save('myHogDector.bin')
    print ('save myHogDector.bin complete...')
    
    print('hard samples training...')
    get_hard_samples(svm,features,labels)
    print('hard samples complete...')
    
    
if __name__ == '__main__':
    #svm config
    svm = svm_config()
    
    #hog training
    hog_train(svm)
