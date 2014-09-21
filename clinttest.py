import cv2 
import pano_stitcher 
import numpy

left = cv2.imread("my_panos//pano1.jpg",-1) 
right = cv2.imread("my_panos//pano2.jpg") 
middle = cv2.imread("my_panos//pano3.jpg",-1) 


# left = cv2.imread("test_data//books_1.png", -1) 
# middle = cv2.imread("test_data//books_2.png") 
# right = cv2.imread("test_data//books_3.png", -1)


leftHomo = pano_stitcher.homography(middle, left) 
warpedLeft, leftOrigin = pano_stitcher.warp_image(left, leftHomo) 
# cv2.imshow("warped left", warpedLeft) 
# cv2.waitKey() 

rightHomo = pano_stitcher.homography(middle, right) 
warpedRight, rightOrigin = pano_stitcher.warp_image(right, rightHomo) 
# cv2.imshow("warped right", warpedRight) 
# cv2.waitKey() 

print leftOrigin
print rightOrigin 
middle = cv2.cvtColor(middle, cv2.COLOR_BGR2BGRA) 

images = (warpedLeft, warpedRight, middle) 
origins = (leftOrigin,rightOrigin,(0, 0))
pano = pano_stitcher.create_mosaic(images, origins) 

cv2.imshow('pano', pano)
cv2.waitKey(0)