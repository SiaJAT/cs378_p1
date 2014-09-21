"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def homography(image_a, image_b, bff_match=False):
    """Returns the homography mapping image_b into alignment with image_a.

    Arguments:
      image_a: A grayscale input image.
      image_b: A second input image that overlaps with image_a.

    Returns: the 3x3 perspective transformation matrix (aka homography)
             mapping points in image_b to corresponding points in image_a.
    """

    MIN_MATCH_COUNT = 10

    sift = cv2.SIFT()

    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # Brute force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

    good = []
    for m, n in matches:
        if m.distance < .9 * n.distance:
            good.append(m)

    print len(good)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good])\
            .reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good])\
            .reshape(-1, 1, 2)

    # cv2.waitKey()
    # cv2.destroyAllWindows()

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3)
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

    print homography
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) 
    h, w, z = image.shape 

    # homo_inv = np.linalg.inv(homography)


    upperLeft = np.array([[0], [0], [1]]) 
    upperRight = np.array([[w], [0], [1]])
    bottomLeft = np.array([[0], [h], [1]])
    bottomRight = np.array([[w], [h], [1]])

    tul = np.dot(homography, upperLeft)
    tur = np.dot(homography, upperRight)
    tbl = np.dot(homography, bottomLeft)
    tbr = np.dot(homography, bottomRight)

    pts1 = np.float32([[0,0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[tul[0][0], tul[1][0]],[tur[0][0], tur[1][0]],[tbl[0][0], tbl[1][0]],[tbr[0][0], tbr[1][0]]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    # print transformUpperLeft, transformUpperRight, transformBottomRight, transformBottomLeft

    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])

    p_prime = np.dot(homography, p)

    # print p_prime

    yrow = p_prime[1]*(1/p_prime[2])
    xrow = p_prime[0]*(1/p_prime[2])
    ymin = min(yrow)
    xmin = min(xrow)
    ymax = max(yrow)
    xmax = max(xrow)

    if ymin < 0:
        homography[1,2] = homography[1,2]-ymin
    if xmin < 0:
        homography[0,2] = homography[0,2]-xmin

    print homography
    asList = list(tul) 
    tup = (asList[0][0], asList[1][0]) 
    
    # x_scale = int(round(homo_inv[0, 0])) 
    # y_scale = int(round(homo_inv[1, 1])) 
    # x_translate = int(round(homo_inv[0,2]))
    # y_translate = int(round(homo_inv[1,2]))

    # height = (y_scale * h) 
    # width = (x_scale * w)
    # print height
    # print width
   
    stitch = cv2.warpPerspective(src = image, M = homography, dsize = (int(round(xmax-xmin)), int(round(ymax-ymin))))
    print stitch.shape
    return stitch, tup



def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.

    Arguments:
      images: a list of 4-channel images to combine in the mosaic.
      origins: a list of the locations upper-left corner of each image in
               a common frame, e.g. the frame of a central image.

    Returns: a new 4-channel mosaic combining all of the input images. pixels
             in the mosaic not covered by any input image should have their
             alpha channel set to zero.
    """
    print origins
    mapped = map(lambda x, y: (x, y), origins, images)

    mapped_sorted = sorted(mapped, key=lambda x: x[0])

    width = mapped_sorted[-1][1].shape[1]
    for x in range(0, len(mapped_sorted)):
        width += np.abs(mapped_sorted[x][0][0])

    origins_diffs = []
    i = 0
    j = len(mapped_sorted) - 1
    mid = len(mapped_sorted) / 2
    while i <= mid and j > mid:
        origins_diffs.append(
            np.abs(mapped_sorted[i][0][1]) -
            np.abs(mapped_sorted[j][0][1]))
        i = i + 1
        j = j - 1

    max_origin_diff = max(origins_diffs)

    height = max([img.shape[0] for img in images]) + max_origin_diff
    print str(height) + " the height"

    stitch = np.zeros((height, width, 4), np.uint8)

    left = mapped_sorted[0]

    # Get Leftmost height and width
    leftX2 = left[1].shape[0]
    leftY2 = left[1].shape[1]

    # stitch the leftmost image
    stitch[:leftX2, :leftY2, :4] = left[1]

    # get the rightmost image
    right = mapped_sorted[-1]

    # rightmost image
    rightX2 = np.abs(left[0][1]) - np.abs(right[0][1])
    rightY2 = np.abs(left[0][0]) + np.abs(right[0][0])

    # append the rightmost image to the stitch
    stitch[rightX2:, rightY2:, :4] = right[1]

    # append middle images one by one
    for x in range(1, len(mapped_sorted) - 1):
        meetX1 = np.abs(mapped_sorted[x - 1][0][1])
        meetY1 = np.abs(mapped_sorted[x - 1][0][0])

        spanX2 = meetX1 + mapped_sorted[x][1].shape[0]
        spanY2 = meetY1 + mapped_sorted[x][1].shape[1]

        stitch[meetX1:spanX2, meetY1:spanY2, :4] = mapped_sorted[x][1]
    # cv2.imshow('img', stitch)
    # cv2.waitKey(0)

    return stitch
    # pass
