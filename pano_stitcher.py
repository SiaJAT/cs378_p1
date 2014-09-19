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

    # sift = cv2.SIFT(edgeThreshold=10, sigma = 1.25, contrastThreshold=0.08)
    sift = cv2.ORB(nlevels=4, edgeThreshold=5, firstLevel=0)

    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # Brute force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

    # Flann matching
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)   # or pass empty dictionary

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(np.asarray(des_a, np.float32),
    #                          np.asarray(des_b, np.float32), 2)

    good = []
    for m, n in matches:
        if m.distance < .90 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good])\
            .reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good])\
            .reshape(-1, 1, 2)

    # cv2.waitKey()
    # cv2.destroyAllWindows()

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 1)
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
    upperLeft = np.array([[0], [0], [1]])
    transformUpperLeft = np.dot(homography, upperLeft)
    asList = list(transformUpperLeft)
    tup = (asList[0][0], asList[1][0])
    y, x, z = image.shape
    x_scale = int(round(homography[0, 0]))
    y_scale = int(round(homography[1, 1]))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image = cv2.warpPerspective(image, homography, (x_scale*x, y_scale*y))

    return image, tup


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
        meetX1 = np.abs(mapped_sorted[x-1][0][1])
        meetY1 = np.abs(mapped_sorted[x-1][0][0])

        spanX2 = meetX1 + mapped_sorted[x][1].shape[0]
        spanY2 = meetY1 + mapped_sorted[x][1].shape[1]

        stitch[meetX1:spanX2, meetY1:spanY2, :4] = mapped_sorted[x][1]
    # cv2.imshow('img', stitch)
    # cv2.waitKey(0)

    return stitch
    # pass
