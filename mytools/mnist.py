# -*- coding: utf-8 -*- 
import gzip 
import sys 
import urllib2
import os
import cv2
import cPickle
import numpy

# if there is no mnist dataset ,download it
def download_data(url='http://deeplearning.net/data/mnist/mnist.pkl.gz'):
    if not os.path.exists('mnist.pkl.gz'):
        print 'downloading the mnist dataset,wait for a few minit!'
        f = urllib2.urlopen(url)
        data = f.read() 
        with open("mnist.pkl.gz", "wb") as code:     
            code.write(data)
        print 'the mnist dataset has been download,have fun with it!'

# load the mnist dataset
def load_data(path="mnist.pkl.gz"): 
    f = gzip.open(path)
    train_set,valid_set,test_set = cPickle.load(f);
    f.close();
    return train_set,valid_set,test_set
   
# translate the mnist dataset to pictures and make labels for caffe
# and save smoothed test pictures
def save_pictures_and_labels(train_x=None,train_y=None,test_x=None,test_y=None,train_labels = 'train_labels.txt',test_labels = 'test_labels.txt',smooth_labels = 'smooth_labels.txt'):
    if not os.path.exists('train_img'):
        os.mkdir('train_img')
    if not os.path.exists('test_img'):
        os.mkdir('test_img')
    if not os.path.exists('smooth_img'):
        os.mkdir('smooth_img')
    train_root_path = './train_img/'
    test_root_path = './test_img/'
    smooth_root_path = './smooth_img/'
    cnt = 1
    f = open(train_labels,'wb')
    for i in range(len(train_y)):
        tmp_path = train_root_path + "%.5d"%cnt + '.jpg'
        img = train_x[cnt-1]    #图片名从1开始
        img = img.reshape(28,28)
        img *= 255
        cv2.imwrite(tmp_path,img)
        f.write(tmp_path)
        f.write('\t')
        f.write(str(train_y[cnt-1]))
        f.write('\n')
        cnt+=1
    f.close()
    cnt = 1
    f = open(test_labels,'wb')
    for i in range(len(test_y)):
        tmp_path = test_root_path + "%.5d"%cnt + '.jpg'
        img = test_x[cnt-1]    #图片名从1开始
        img = img.reshape(28,28)
        img *= 255
        cv2.imwrite(tmp_path,img)
        f.write(tmp_path)
        f.write('\t')
        f.write(str(test_y[cnt-1]))
        f.write('\n')
        cnt+=1
    f.close()
    cnt = 1
    f = open(smooth_labels,'wb')
    for i in range(len(test_y)):
        tmp_path = smooth_root_path + "%.5d"%cnt + '.jpg'
        img = test_x[cnt-1]    #图片名从1开始
        img = img.reshape(28,28)
        img = cv2.blur(img,(2,2))
        img *= 255
        cv2.imwrite(tmp_path,img)
        f.write(tmp_path)
        f.write('\t')
        f.write(str(test_y[cnt-1]))
        f.write('\n')
        cnt+=1
    f.close()


if __name__ == '__main__':
    download_data()
    train_set,valid_set,test_set = load_data()
    train_x,train_y = train_set
    test_x,test_y = test_set
    save_pictures_and_labels(train_x,train_y,test_x,test_y)
