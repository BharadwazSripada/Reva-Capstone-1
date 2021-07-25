#!/usr/bin/env python
# coding: utf-8

# In[13]:


#This code is for performing Image subtraction when a background image and live image is given


# In[2]:


import matplotlib.pyplot as plt
from IPython.display import Image,display
import pathlib
import os
import glob
import cv2
import numpy as np


# In[3]:



def preprocess_image(image):
    #Resize the Images
    image = cv2.resize(image,(640,480))
    #plt.subplot(212)
    #plt.imshow(image)
    #plt.show()

    '''
    convert imags to grayscale. This reduces matrices from 3 (R, G, B) to just 1 
    cvtColor is used to convert any RGB image into respective color-space conversion - Here we are converting the original
    image to a gray scale image using color_bgr2gray
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #plt.subplot(212)
    #plt.imshow(image)
    #plt.show()
    '''
    blur the images to get rid of sharp edjes/outlines. Gaussian blur take image as first argument, size of the kernel
    which should be odd and sigma for computing the radius of the pixels till which the distance need to be computed
    '''
    image = cv2.GaussianBlur(image, (21, 21), 0)
    #plt.subplot(212)
    #plt.imshow(image)
    #plt.show()
    return(image)


# In[4]:


def contours(img_1,img_2):
    
    img1 = preprocess_image(img_1)
    img2 = preprocess_image(img_2)

# obtain the difference between the two images & display the result
    imgDelta = cv2.absdiff(img1, img2)

# coonvert the difference into binary & display the result
#Here threshold is given as 30 meaning all those pixels which are <= 30 should be replaced with 255
#We use simple binary threshold meaning we will only have 2 values 
    thresh = cv2.threshold(imgDelta, 30, 255, cv2.THRESH_BINARY)[1]

# dilate the thresholded image to fill in holes & display the result
    thresh = cv2.dilate(thresh,None, iterations=2)

# find contours or continuous white blobs in the image
    contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# draw a bounding box/rectangle around the largest contour
    resu = img_2.copy()
    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
    
    return(img2)


# In[11]:


img_2feet = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\Image_6feet.jpg')
img_2feet_object = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\Image_2feet_object.jpg')


# In[12]:


Final_Image = (contours(img_2feet,img_2feet_object))
path = 'C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\'
cv2.imwrite(os.path.join(path , 'Final_Output1.jpg'), Final_Image)


# In[23]:


img_4feet = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\Image_2feet.jpg')
img_2feet_object = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\Image_2feet_object.jpg')


# In[24]:


plt.imshow(contours(img_4feet,img_2feet_object))


# In[25]:


img_6feet = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\Image_2feet.jpg')
img_2feet_object = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\Image_2feet_object.jpg')


# In[26]:


plt.imshow(contours(img_6feet,img_2feet_object))


# In[27]:


img_6feet = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\Image_2feet.jpg')
img_2feet_object = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\Image_2feet_addl_objects.jpg')


# In[32]:


plt.imshow(contours(img_6feet,img_2feet_object))


# In[5]:


bck_image = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Research project\\Image subtraction\\REVA pics\\Background1.jpg')
obj_image = cv2.imread('C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Research project\\Image subtraction\\REVA pics\\Objects.jpg')


# In[6]:


Final_Image = (contours(bck_image,obj_image))


# In[7]:


path = 'C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Capstone Project\\Images for back ground subtraction\\'
cv2.imwrite(os.path.join(path , 'Final_Output.jpg'), Final_Image)


# In[ ]:




