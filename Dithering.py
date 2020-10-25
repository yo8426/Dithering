import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

#input
im1 = cv2.imread('newborn.tif')
plt.subplot(2,2,1)
plt.imshow(im1)
plt.subplot(2,2,3)
plt.imshow(im1)

#uint82double
im1=im1.astype(np.double)
#rgb2gray
im1=(im1[:,:,0]+im1[:,:,1]+im1[:,:,2])/3
#avoid effect others
im2=copy.deepcopy(im1)
im3=copy.deepcopy(im1)
###############################################################################
#D1
D1=np.array([[0,128],
            [192,64]])
msk_D1=np.tile(D1,(int(im1.shape[0]/2),int(im1.shape[1]/2)))

for i in range(im2.shape[0]):
    for j in range(im2.shape[1]):
        if im2[i,j] > msk_D1[i,j]:
            im2[i,j]=1
        else:
            im2[i,j]=0

im2 = (255*(im2 - np.min(im2))/(np.max(im2)-np.min(im2)).astype(int))
im2=im2.astype(np.uint8)
plt.subplot(2,2,2)
plt.imshow(im2, cmap = 'gray')
cv2.imwrite('newborn2.tif',im2)
###############################################################################
#D2
D2=np.array([[0,128,32,160],
             [192,64,224,96],
             [48,176,16,144],
             [240,112,208,80]])
msk_D2=np.tile(D2,(int(im1.shape[0]/4),int(im1.shape[1]/4)))

for i in range(im1.shape[0]):
    for j in range(im1.shape[1]):
        if im1[i,j]>msk_D2[i,j]:
            im3[i,j]=1
        else:
            im3[i,j]=0

plt.subplot(2,2,4)
plt.imshow(im3, cmap='gray')
im3=im3.astype(np.uint8)
im3 = (255*(im3 - np.min(im3))/np.ptp(im3)).astype(int)
cv2.imwrite('newborn3.tif',im3)
