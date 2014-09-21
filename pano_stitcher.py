"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def homography(image_a, image_b, bff_match=False):
    """Returns the homography mapping image_b into alignment with image_a.

    Arguments:
      image_a: A grayscale input image.
      image_b: A second input image that overlaps with image_a.

    Returns: the 3x3 perspective transformation matrix (aka homography)
             mapping points in image_b to corresponding points in image_a.
    """

    MIN_MATCH_COUNT = 10

    sift = cv2.SIFT(edgeThreshold=10, sigma = 1.25, contrastThreshold=0.08)

    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # Brute force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

    good = []
    for m, n in matches:
        if m.distance < .9 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good])\
            .reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good])\
            .reshape(-1, 1, 2)

    # cv2.waitKey()
    # cv2.destroyAllWindows()

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
    return M


def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """

    # print homography
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) 
    h, w, z = image.shape 

    upperLeft = np.array([[0], [0], [1]]) 
    upperRight = np.array([[w], [0], [1]])
    bottomLeft = np.array([[0], [h], [1]])
    bottomRight = np.array([[w], [h], [1]])

    tul = np.dot(homography, upperLeft)
    tur = np.dot(homography, upperRight)
    tbl = np.dot(homography, bottomLeft)
    tbr = np.dot(homography, bottomRight)

    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])

    p_prime = np.dot(homography, p)
    # print "p prime"
    # print p_prime

    yrow = p_prime[1]*(1/p_prime[2])
    xrow = p_prime[0]*(1/p_prime[2])
    ymin = min(yrow)
    xmin = min(xrow)
    ymax = max(yrow)
    xmax = max(xrow)

    # print (xmin, xmax)

    new_mat = np.array([[1,0,-1*xmin],[0,1,-1*ymin],[0,0,1]])

    homography = np.dot(new_mat, homography)
    # print homography
    asList = list(tul) 
    tup = (asList[0][0], asList[1][0]) 

    height = int(round(ymax-ymin))
    width = int(round(xmax-xmin))
    
   
    stitch = cv2.warpPerspective(src = image, M = homography, dsize = (width, height))
    # print stitch.shape
    return stitch, tup

def create_mosaic(images, origins):
    mapped = map(lambda x, y: (x, y), origins, images)
    dist_sorted = sorted(mapped, key=lambda x: math.sqrt(x[0][0]**2 + x[0][1]**2), reverse= True)
    x_sorted = sorted(mapped, key=lambda x: x[0][0])
    y_sorted = sorted(mapped, key=lambda x: x[0][1])

    width = x_sorted[-1][1].shape[1]
    for i in range(0, len(x_sorted)):
        width += np.abs(x_sorted[i][0][0])

    height = y_sorted[0][1].shape[0] + y_sorted[0][0][1]
    print "start height " + str(height)
    for image in y_sorted:
        height = max(height, -1*y_sorted[0][0][1]+image[1].shape[0]+image[0][1])

    print "width, height " + str((width, height))

    stitch = np.zeros((height, width, 4), np.uint8)

    if x_sorted[0][0][0] > 0:
        cent_x = 0 # leftmost image is central image
    else :
        cent_x = abs(x_sorted[0][0][0])

    if y_sorted[0][0][1] > 0:
        cent_y = 0 # topmost image is central image
    else :
        cent_y = abs(y_sorted[0][0][1])

    central_image = (cent_x, cent_y)
    print central_image

    for image in dist_sorted:
        start_y = image[0][1] + central_image[1]
        start_x = image[0][0] + central_image[0]
        end_y = start_y + image[1].shape[0]
        end_x = start_x + image[1].shape[1]
        print image[1].shape
        print "start x start y" + str((start_x, start_y))
        print "end_x end_y" + str((end_x, end_y))
        

        stitch[start_y:end_y, start_x:end_x, :4] = image[1]

    return stitch