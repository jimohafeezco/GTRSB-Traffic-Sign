#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 00:16:01 2019

@author: hafeez
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import PredefinedSplit

import numpy as np
import csv
import pandas as pd
from skimage import io
import os
import cv2
from skimage.transform import rotate
import numpy as np
test = pd.read_csv('GTSRB_Final_Test_GT/GT-final_test.csv',sep=';')
  
import time
import scikitplot as skplt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
i = 0


X_test = []
y_test = []


#test data
for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/',file_name)
    X_test.append(io.imread(img_path))
    y_test.append(class_id)

    
X_test = np.array(X_test)
y_test = np.array(y_test)

#training data
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader)
#        skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels


trainImages, trainLabels = readTrafficSigns('GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')

trainLabels = [int(i) for i in trainLabels]


padded_img =[]
sizes = []
def paddingImage(image):
    paddedImg = []
    for i in range(len(image)):
        w, h = image[i].shape[0], image[i].shape[1]
        p = w-h
        sizes.append(image[i].shape)
        if w>h:
            image[i] = np.lib.pad(image[i], ((0,0), (0,p), (0,0)), 'constant', constant_values = (0,))
        elif w<h:
            image[i] = np.lib.pad(image[i], ((0,abs(p)), (0,0), (0,0)), 'constant', constant_values = (0,))
        paddedImg.append(image[i])
    return paddedImg
        
train_paddedImg = paddingImage(trainImages)
test_paddedImg = paddingImage(X_test)


sizespadded= []
for i in range(len(train_paddedImg)):
    w, h = train_paddedImg[i].shape[0], train_paddedImg[i].shape[1]
    sizespadded.append(train_paddedImg[i].shape)       
        
randImg = [1, 5000, 11000, 20000]
   
for i in randImg:
    plt.figure()
    plt.imshow(train_paddedImg[i])
    plt.show()


def resizeImage(image, x,y):
    for j in range(len(image)):
        image[j]=cv2.resize(image[j],(x,y))
    return image


x_train_resize = resizeImage(train_paddedImg, 30, 30)
x_test_resize = resizeImage(test_paddedImg, 30, 30)

for i in randImg:
    plt.figure()
    plt.imshow(x_train_resize[i])
    plt.show()

#np.random.randint(1000, size =(4))

sizes= []
for i in range(len(trainImages)):
    w, h = trainImages[i].shape[0], trainImages[i].shape[1]
    sizes.append(trainImages[i].shape)       


for i in randImg:
    plt.figure()
    plt.imshow(trainImages[i])
    plt.show()


#np.random.randint(1000, size =(4))
for i in randImg:
    plt.figure()
    plt.imshow(x_train_resize[i])
    plt.show()



from random import seed
from random import randrange
 
# Split a dataset into a train and test set
def train_test_split(dataset,label, split):
    train = list()
    trainlabels=list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    trainlabels_copy = list(label)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
        trainlabels.append(trainlabels_copy.pop(index))
    return train, dataset_copy, trainlabels, trainlabels_copy


X_train, X_validation, Y_train, Y_validation =train_test_split(x_train_resize, trainLabels, 0.80)
print(len(X_train), len(X_validation), len(Y_train), len(Y_validation))


randImg = np.random.randint(1000, size =(4))
for i in randImg:
    plt.figure()
    plt.imshow(x_train_resize[i])
    plt.show()


#def translate_image(image, max_trans = 5, height=30, width=30):
#    translate_x = max_trans*np.random.uniform() - max_trans/2
#    translate_y = max_trans*np.random.uniform() - max_trans/2
#    translation_mat = np.float32([[1,0,translate_x],[0,1,translate_y]])
#    trans = cv2.warpAffine(image, translation_mat, (height,width))
#    return trans
    
def translate_image(img, shift=np.random.randint(5,15), direction='right', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img

def rotate_image(image, max_angle =15):
    rotate_out = rotate(image, np.random.uniform(-max_angle, max_angle), mode='edge')
    return rotate_out




plt.Figure()

label_count= pd.value_counts(Y_train)

xxx= pd.DataFrame(pd.value_counts(trainLabels))


dataxx= pd.DataFrame(pd.value_counts(Y_train))
plt.Figure()
plt.bar(dataxx.index, dataxx[0], color = 'g') # hist for  train split
plt.title('Label distribution after split')
plt.show()
plt.bar(xxx.index, xxx[0], color = 'g') # hist for whole training before split
plt.title('Label distribution before split')

plt.show()


X_train_Unaug = X_train.copy()
Y_train_Unaug = Y_train.copy()

label_max =pd.value_counts(Y_train).max()
#Data augmentation starts herex
XtrainAug=[]
YtrainAug =[]
def data_augment(data, data_label):        
    for i in range(len(data_label)):
        while data_label.count(data_label[i])< label_max:
            data_label.append(data_label[i])
            data.append(translate_image(rotate_image(data[i])))
    return data,data_label


XtrainAug, YtrainAug = data_augment(X_train,Y_train)

dataxxxx= pd.DataFrame(pd.value_counts(YtrainAug))
plt.Figure()
plt.bar(dataxxxx.index, dataxxxx[0], color ='g')


def norm_ravel(data):
    train_norm = []
    for i in range(len(data)):
    #    train_
    #   train_norm[i]= train_aug[i]/255
       norm = data[i]/255
       train_norm.append(norm.ravel())
    return train_norm

XtrainNorm = norm_ravel(XtrainAug)
xValiNorm = norm_ravel(X_validation)
xTestNorm = norm_ravel(x_test_resize)


XtrainDF =np.stack(XtrainNorm,axis=0)
ytrainDF= np.stack(Y_train, axis=0)
yvaliDF = np.stack(Y_validation, axis =0)
xvaliDF = np.stack(xValiNorm, axis =0)
#XtestDF = np.stack(X_test, axis =0)
ytestDF = np.stack(y_test, axis =0)
xtestDF = np.stack(xTestNorm, axis =0)
#
#def convertDF(data):
#    df0 = pd.DataFrame(data[0:10000])
#    df1= pd.DataFrame(data[10000:20000])
#    df2 = pd.DataFrame(data[20000:30000])
#    df3 = pd.DataFrame(data[30000:40000])
#    df4 = pd.DataFrame(data[40000:50000])
#    df5 = pd.DataFrame(data[50000:60000])
#    df6 = pd.DataFrame(data[60000:70000])
#    df7 = pd.DataFrame(data[70000:len(data)])
#    frames = [df0, df1, df2, df3, df4, df5, df6, df7]
#    return pd.concat(frames)
  





def null_columns(data):
  cols = list(data.columns[np.where(data.isnull().any())])
  counts = data.isnull().sum()
  return {c:counts[c] for c in cols}
null_columns(XtrainDF)
null_columns(ytrainDF)





#resize


#cross validation starts here

val_prec= []
val_accur = []
val_recall = []
times = []  
estimators = [10, 50, 100, 200]

for estimator in estimators:
    rfCV = RandomForestClassifier(n_estimators=estimator, random_state=42)
    rfCV.fit(XtrainDF , ytrainDF)
    
    y_train_pred=rfCV.predict(XtrainDF)
#    ytestPred = rf.predict(xtestDF)
    
    yValPred =rfCV.predict(xvaliDF)
        
    score= accuracy_score(yvaliDF, yValPred)
    recall = recall_score(yvaliDF, yValPred, average='weighted')
    precision = precision_score(yvaliDF, yValPred, average='weighted')
    
    val_prec.append(score)
    val_accur.append(recall)
    val_recall.append(precision)
#print('time it takes for each n is', times)
print('accuracy for CV is', val_accur)
print('recall for CV is', val_recall)
print('precision for CV is',val_prec)





rf = RandomForestClassifier(n_estimators=300, random_state=42)

rf.fit(XtrainDF , ytrainDF)
ytestPred = rf.predict(xtestDF)
ytestPred_prob = rf.predict_proba(xtestDF)

incorrects_indices = np.nonzero(ytestPred.reshape((-1,)) != ytestDF)

yValPred =rf.predict(xvaliDF)


val_score= accuracy_score(yvaliDF, yValPred)
vali_recall = recall_score(yvaliDF, yValPred, average='weighted')
vali_precision = precision_score(yvaliDF, yValPred, average='weighted')

test_recall = recall_score(ytestDF, ytestPred, average='weighted')
test_precision = precision_score(ytestDF, ytestPred, average='weighted')
test_score = accuracy_score(ytestDF, ytestPred)


skplt.metrics.plot_roc(ytestDF, ytestPred_prob)
skplt.metrics.plot_precision_recall(ytestDF, ytestPred_prob)
skplt.metrics.plot_confusion_matrix(ytestDF, ytestPred)
print('accuracy is', val_score, test_score)
print('recall is', vali_recall, test_recall)
print('precision is',vali_precision, test_precision)





# training on unaugmented data
print('training on unaugmented data')
XtrainNormU = norm_ravel(X_train_Unaug)
XtrainDFU =np.stack(XtrainNormU)
ytrainDFU= np.stack(Y_train_Unaug, axis=0)
#XtestDF = np.stack(X_test, axis =0)

rfU = RandomForestClassifier(n_estimators=300, random_state=42)
rfU.fit(XtrainDFU, ytrainDFU)
y_trainPredU=rfU.predict(XtrainDFU)

ytestPredU = rfU.predict(xtestDF)

yValPredU =rfU.predict(xvaliDF)

train_scoreU=accuracy_score(ytrainDFU, y_trainPredU)
train_recallU = recall_score(ytrainDFU, y_trainPredU, average='weighted')
train_precisionU = precision_score(ytrainDFU, y_trainPredU, average='weighted')


val_scoreU= accuracy_score(yvaliDF, yValPredU)
vali_recallU = recall_score(yvaliDF, yValPredU, average='weighted')
vali_precisionU = precision_score(yvaliDF, yValPredU, average='weighted')

test_recallU = recall_score(ytestDF, ytestPredU, average='weighted')
test_precisionU = precision_score(ytestDF, ytestPredU, average='weighted')
test_scoreU = accuracy_score(ytestDF, ytestPredU)

print('accuracy is', train_scoreU, val_scoreU, test_scoreU)
print('recall is', train_recallU, vali_recallU, test_recallU)  
print('precision is', train_precisionU,vali_precisionU, test_precisionU)
#
#
#sizes = [10, 15, 20, 40, 60]
#for i sizes:
#    x_train_resize = resizeImage(train_paddedImg, i, i)
#    x_test_resize = resizeImage(test_paddedImg, i, i)
#    

