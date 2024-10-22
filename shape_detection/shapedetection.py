import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
font1 = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread("C:\\Users\\Merve\\Desktop\\openCv\\shape_detection\\polygons.png")
img1=cv2.imread("C:\\Users\\Merve\\Desktop\\openCv\\shape_detection\\shape.png")
img2 = cv2.imread("C:\\Users\\Merve\\Desktop\\openCv\\shape_detection\\shape1.png")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2  = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

_,threshold = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
_,threshold1 = cv2.threshold(gray1,240,255,cv2.THRESH_BINARY)
_,threshold2 = cv2.threshold(gray2,240,255,cv2.THRESH_BINARY)

a=0
b=0
c=0
d=0 
e=0

#erosion ve dilation

contours,_ = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours1, _ = cv2.findContours(threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(threshold2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

kernel = np.ones((5, 5), np.uint8)
#img_erosion = cv2.erode(img2, kernel, iterations=1) 
img_dilation = cv2.dilate(img2, kernel, iterations=2) 

def process_contours(image, contours):  
    global a,b,c,d,e
    for cnt in contours:
        epsilon = 0.01*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt,epsilon, True)
       
        cv2.drawContours(image,[approx],0,(0,0,0),5)

        x = approx.ravel()[0]
        y = approx.ravel()[1]

        print(approx)
        print(len(approx))

        if len(approx)== 3:
            cv2.putText(image,"Triangle",(x,y),font1,1,(0,0,0))
            a+=1
        

    
        elif len(approx)== 4:
            cv2.putText(image,"Rectangle",(x,y),font1,1,(0,0,0))
            b+=1
    
        elif len(approx)== 5:
            cv2.putText(image,"Pentagon",(x,y),font,1,(0,0,0))
            c+=1
    
        elif len(approx)== 6:
            cv2.putText(image,"Hexagon",(x,y),font,1,(0,0,0))
            d+=1
    
        else: 
            cv2.putText(image,"Ellipse",(x,y),font,1,(0,0,0))
            e+=1

process_contours(img, contours)
process_contours(img1, contours1)
process_contours(img_dilation, contours2)

print(f"triangle={a}, rectangle={b}, pentagon={c}, hexagon={d}, ellipse={e}")

cv2.imshow("IMG", img)
cv2.imshow("IMG1", img1)
cv2.imshow("IMG2",img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
