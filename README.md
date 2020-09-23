# GTRSB-Traffic-Sign

## 1.1 Padding and Resizing
The Images were observed to have different sizes of height and width. They were padded by
adding Zeros to the shorter side to make it a square shaped. All images were equally resized
to a shape of 30*30 after the padding operation was done. The result of the padding and
resizing as compared to the original image shown in Fig 1-3 is shown in Fig.4 -6.
## 1.2 Frequency of Image Labels (Traffic Signs)
An additional step done as part of the Exploratory data analysis is to examine the frequency
or distribution of our datasets. As shown , it was observed that it is a largely unbalanced
dataset with different frequency of data/Image labels. 

## Dat Augmentation 
There are various techniques to augment data. Augmentation in this sense means generating more training samples from the training samples available. This is done for a number
of reasons. This is usually done when we have unbalanced classes and we do not want our
classifier to be a majority classifier that predicts base on the majority class. We also do it
when we have small dataset. Small dataset can result into overfitting. Also, training our
model without augmenting when situation demands can result into overfitting. In this work,
two data augmentation techniques have been applied. In other to make sure that individual
augmented data are unique, some randomizations have been applied to the parameter off
the function that augments the data. This prevents artificial dataset of being monotonous
or correlated.


## Results
A random forest classifier was used to train an image of 78001*2700 using number of estimators as hyper-parameter. The model was then evaluated on the validation set and the
result obtained is presented in Table 1. It can be observed that the performance of out
model performs better as the number of estimators increased. Though this is at a huge cost
of longer trainning time and computationally expensive if not training on a GPU machine.
While it took less than less than 2 minutes to train when the number of estimators is 10, it
took more than 30 minutes when the number of estimators is 300. The training was done
on a Core i5 8GB ram CPU

![alt text](https://github.com/jimohafeezco/GTRSB-Traffic-Sign/blob/master/result.png)

