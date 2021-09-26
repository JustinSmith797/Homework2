import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#
"""
Homework 2
Justin Smith
RBE 549-SF
"""


def harris_corner(img):
    print("HarrisCornerMain")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    u,v = img_gray.shape
    sigma = 1.4 #2   
##    img_gray = cv.GaussianBlur(img_gray,(3,3),sigma)
    #Get Sobel x an y
    SobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    SobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    

    gx = convolve(img_gray,SobelX)
    gy = convolve(img_gray,SobelY)
    
##    cv.imshow("SobeX",gx)
##    cv.imshow("SobelY",gx)

    
    Gxx = cv.GaussianBlur(np.square(gx),(3,3),sigma) 
    Gxy = cv.GaussianBlur(np.multiply(gx,gy),(3,3),sigma)  
    Gyy = cv.GaussianBlur(np.square(gy),(3,3),sigma)

   
##    cv.imshow("SobelSquareX",Gxx)
##    cv.imshow("SobelSquareY",Gyy)
    
    
    k = 0.04
    R = np.zeros((u, v), np.float32)
#Determine R Matrix
    for row in range(u):
        for col in range(v):
            M = np.array([[Gxx[row][col], Gxy[row][col]], [Gxy[row][col], Gyy[row][col]]])
            R[row][col] = np.linalg.det(M) - (k * np.square(np.trace(M)))
    
    radius = 5
    color = (0, 255, 0)  # Green
    thickness = 1
    
    output_img = np.copy(img)
    cornerList = []
#LOCAL NON-MAXIMUM SUPPRESSION    
    for row in range(u):
        for col in range(v):
            if R[row][col] > 400000000:
                max = R[row][col]
                skip=False
                for n_row in range(5):
                    for n_col in range(5):
                        if row + n_row - 2 < u and col + n_col - 2 < v:
                            if R[row + n_row - 2][col + n_col - 2] > max:
                                skip = True
                                break
                
                if not skip:
                    cornerList.append([row, col, R[row][col]])
                    cv.circle(output_img,(col,row),radius, color ,thickness)

    return  cornerList,output_img,R 

def convolve(img,kernel):
    print("Processing Convolution...")
    w,h = img.shape
    i,j = kernel.shape
    G  = np.zeros(img.shape, np.float32)

    
    for row in range(i,w):
        for col in range(j,h):
            for m in range(i):
                for n in range(j):
                    G[row][col] += kernel[m][n]*img[row-m-1][col-n-1]
                    
    return G

##img1 = cv.imread("img1.jpg")
##
##
##model_list,output_img1,R = harris_corner(img1)
##
##cv.imshow("Corners1", output_img1)

cv.waitKey(0)
cv.destroyAllWindows()


    
    



## References: https://github.com/ShivamChourey/Harris-Corner-Detection/blob/master/Corner_Detection.py
##https://www.youtube.com/watch?v=BPBTmXKtFRQ
#http://cs.brown.edu/courses/csci1430/2013/results/proj2/valayshah/
