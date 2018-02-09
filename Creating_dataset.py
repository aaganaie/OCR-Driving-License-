import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import csv

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


bin_n = 16
Features=[]
Target=[]

path=r'Data/Train_1/'
for root,dir_name,filename in os.walk(path):
    for name in filename:
        path = os.path.join(root,name)
        Image=cv2.imread(path)
        y=12
        x=20
        h=105
        w=85

        Image= Image[y:y+h, x:x+w]

        #plt.imshow(Image)
        #plt.show()
        G_Image=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
        ret,th = cv2.threshold(G_Image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        th=cv2.resize(th,(8,8),interpolation = cv2.INTER_AREA)
        L=[]
        L=hog(th)
        L=np.append(L,root[-1])
        Features.append(L)
        #Target.append(root[-1])

path=r'Data/Train_2/'
for root,dir_name,filename in os.walk(path):
    for name in filename:
        path = os.path.join(root,name)
        Image=cv2.imread(path)
        y=12
        x=20
        h=105
        w=85

        Image= Image[y:y+h, x:x+w]

        #plt.imshow(Image)
        #plt.show()
        G_Image=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
        ret,th1 = cv2.threshold(G_Image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        th1=cv2.resize(th1,(8,8),interpolation = cv2.INTER_AREA)
        L1=[]
        L1=hog(th1)
        L1=np.append(L1,root[-1])
        Features.append(L1)
        #Target.append(root[-1])

with open ("Image_Datac8.csv",'w',newline='')as file:
    writer=csv.writer(file,delimiter=',')
    for data in Features :
        writer.writerow(data)
        




        
