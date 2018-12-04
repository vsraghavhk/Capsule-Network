import numpy as np
import cv2
import os

def createArray(folder, count, flag): 	 # Will be called for each of train, test and validate folders
		
    classCounter = 0
    imageCounter = 0
    
    xArray = np.zeros((count, 32, 32, 3)) # size can be 32 or 128
    if flag==0:
        yArray = np.zeros((count))
    if flag==1:
        ytest = []
    for fn1 in os.listdir(folder):
        
        for fn2 in os.listdir(os.path.join(folder, fn1)):
            
            path = os.path.join(fn1, fn2)
            img = cv2.imread(os.path.join(folder, path))	# Read images one by one
            
            # if imageCounter%500 == 0:
            #     print("img before canny : ", img.shape)
            # img = cv2.Canny(img, 125, 375)
            # if imageCounter%500 == 0:
            #     print("img after canny : ", img.shape)
            
            # img = img.reshape((img.shape[0], img.shape[1], 1))
            
            xArray[imageCounter] = img
                        
            if flag==0:
                yArray[imageCounter] = classCounter #Label of image. 0 for all images in class 0, 1 for all in class 1...
            
            if flag==1:
                ytest.append(classCounter)

            imageCounter+=1
            '''
            if imageCounter%500 == 0:
                print("img after reshape: ", img.shape)
                print(img[2])

                print("xArray : ", xArray[imageCounter].shape)
                print(xArray[imageCounter][2])
            '''
        classCounter+=1

    # xArray = np.array(x)
    # print("x : ", type(xArray))
    # print("y : ", type(yArray))
    # yArray = np.array(y)
    if flag==1:
        ynum = np.array(ytest)
    
    if flag==0:
        return xArray, yArray
    if flag==1:
        return xArray, ynum