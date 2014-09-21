import cv2 
import pano_stitcher 
import numpy 
# pano1 = cv2.imread("my_panos//pano1.jpg") 
# Left 
# pano2 = cv2.imread("my_panos//pano2.jpg") 
# Middle 
# pano3 = cv2.imread("my_panos//pano3.jpg") 

# Right 
pano1 = cv2.imread("test_data//books_1.png", -1)
# Left 
pano2 = cv2.imread("test_data//books_2.png") 
 
pano3 = cv2.imread("test_data//books_3.png", -1)

pano12Homography = pano_stitcher.homography(pano2, pano1) 
warpedPano12, tup12 = pano_stitcher.warp_image(pano1, pano12Homography) 
print tup12 
cv2.imshow("warped 1 2", warpedPano12) 
cv2.waitKey() 

# pano23Homography = pano_stitcher.homography(pano2, pano3) 
# warpedPano23, tup23 = pano_stitcher.warp_image(pano2, pano23Homography) 


# print tup23 
# print tup12 
images = (warpedPano12, pano2) 
# pano2Origins = (0, 0) 
origins = (tup12, (0, 0)) 
pano = pano_stitcher.create_mosaic(images, origins) 
cv2.imwrite("myPanoTest.png", pano) # print tup3