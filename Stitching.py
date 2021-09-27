import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import HCD


def SSD(R1,R2,modelVect,dataVect):
# Computes the sum of squares difference between 3x3 patches centered on the marked corner in both the model and data image.     
    i,j = modelVect[:2]
##    print(i)
##    print(j)
    i = int(i)
    j = int(j)

    m,n = dataVect[:2]     
##    print(m)
##    print(n)
    
    m = int(m)
    n = int(n) 
    modelKernel = np.array([[R1[i-1][j+1],R1[i][j+1],R1[i+1][j+1]],[R1[i-1][j],R1[i][j],R1[i+1][j+1]],[R1[i-1][j+1],R1[i-1][j],R1[i-1][j-1]]])
    #print(modelKernel)
    dataKernel = np.array([[R2[m-1][n+1],R2[m][n+1],R2[m+1][n+1]],[R2[m-1][n],R1[m][n],R2[m+1][n+1]],[R2[m-1][n+1],R2[m-1][n],R2[m-1][n-1]]])
    #print(dataKernel)
    SSD_value = np.sqrt(np.sum((modelKernel-dataKernel)**2))
 #   print(SSD_value)

    return SSD_value

def feature_matching(ref_img, img):

   
    
    model_list,output_img2,R1 = HCD.harris_corner(ref_img)
    data_list,output_img2,R2 = HCD.harris_corner(img)

    
    threshold = 10**18
    
    model_list = np.array(model_list)
    data_list =  np.array(data_list)
    
    h_model = len(model_list)
    print(h_model)
    h_data = len(data_list)
    print(h_data)
    
    kpts_des = np.zeros((h_model,7), np.float32)
    for i in range(h_model):
        matchList =[] 
        for j in range(h_data):
            modelVect= model_list[i]
##            print(modelVect)
            dataVect= data_list[j]
##            print(dataVect)
            SSD_value = SSD(R1,R2,modelVect,dataVect)
            if SSD_value < threshold:
##                print(model_list[i])
##                print(data_list[j])
                a = np.hstack((model_list[i], data_list[j], SSD_value))
                b = a.tolist()
                matchList.append(b)

#Return the smallest value in SSD list for that particular point. Is marked as a potential match between two images.                 
            kpts = min(matchList, key=lambda x: x[6])
            kpts_des[i] = kpts              


#RANSAC ALGORITHM
    
    n = 100
    k = 10
    t = 10
    d = 4
    data = kpts_des
    iterations = 0

    bestErr = 10

    while iterations < k:
        Ts = random.sample(list(kpts_des),d)
        p1 = [Ts[0][0],Ts[0][1]]
        p2 = [Ts[1][0],Ts[1][1]]
        p3 = [Ts[2][0],Ts[2][1]]
        p4 = [Ts[3][0],Ts[3][1]]
        pp1 =[Ts[0][3],Ts[0][4]]
        pp2 =[Ts[1][3],Ts[1][4]]
        pp3 =[Ts[2][3],Ts[2][4]]
        pp4 =[Ts[3][3],Ts[3][4]]
        pts_src =np.array([pp1,pp2,pp3,pp4])
        pts_dst =np.array([p1,p2,p3,p4])
        h, status = cv.findHomography(pts_src,pts_dst)
        print(h)
"""
Unfortunately I did not have enough time to complete or optimize the assignment.  Troubleshooting each step took a very long
as the code runs  slow. 

My next steps were to use the sample homography to compute point transformations of the points in the model image.
From here I would calculate the error between those transformed points to the points matched in the other image using sum
of squares differences. This process would be repeated until 4 matches are achieved.

Using these matches, I would re-compute the homography and stitch the images together. 
"""


        
        iterations += 1      
    print("done")
    return Ts


    
img1 = cv.imread("img1.jpg")
img2 = cv.imread("img2.jpg")


kpts_des = feature_matching(img1,img2)




cv.waitKey(0)
cv.destroyAllWindows()
