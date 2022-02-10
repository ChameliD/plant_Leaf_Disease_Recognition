import cv2
import numpy as np
from matplotlib import pyplot as plt

originalImage=cv2.imread("tomato1.jpg")
imageCheck=cv2.imread("tomato1.jpg")

kernel = np.ones((5,5),np.uint8)
removeNoiseC = cv2.morphologyEx(imageCheck, cv2.MORPH_OPEN, kernel)
removeNoiseOriginal = cv2.morphologyEx(originalImage, cv2.MORPH_OPEN, kernel)

cv2.imshow("Original leaf without infected",removeNoiseC)
cv2.imshow("Leaf you want to check",removeNoiseOriginal)


def detectColorChanges(img):

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_range=np.array([25,52,72])
    upper_range=np.array([102,255,255])
    mask=cv2.inRange(hsv,lower_range,upper_range)

    res = cv2.bitwise_and(img,img,mask= mask)

    cv2.imshow('Result',res)


    newIm=img-res
    cv2.imshow("After remove green",newIm)


    
    if all(newIm[0,0] == [0,0,0]):
        print("leaf has not color changes.")
        return False
    else:
        print("leaf has not color changes.")
        return True

    
def detectMutance(img1,img2):
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gg1",gray1)
    cv2.imshow("gg2",gray2)

    ret,thresh1 = cv2.threshold(gray1,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh2 = cv2.threshold(gray2,127,255,cv2.THRESH_BINARY_INV)


    dim=gray1.shape

    new_img1 = cv2.resize(thresh1,(dim[1],dim[0]))
    new_img2 = cv2.resize(thresh2,(dim[1],dim[0]))

    sub=new_img2-new_img1
    cv2.imshow("mm",sub)

    if all(sub[0,0] == [0,0,0]):
        print("Leaf has mutanse")
        return False
    
    else:
        print("Leaf has mutance")
        return False

if detectColorChanges(removeNoiseC)or detectMutance(removeNoiseOriginal,removeNoiseC):
    print("Leaf has mutance")
    print("Leaf has infected by diseas")

else:
    print("Leaf has infected by diseas")

    

cv2.waitKey(0)
cv2.destroyAllWindows()
