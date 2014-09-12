"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from find_obj import filter_matches,explore_match


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
    #sift = cv2.orb()
    #sift = cv2.BRISK()

    kp_a, des_a = sift.detectAndCompute(image_a,None)
    kp_b, des_b = sift.detectAndCompute(image_b,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a,trainDescriptors = des_b,k=2)
    p1, p2, kp_pairs = filter_matches(kp_a, kp_b, matches)
    explore_match('find_obj', image_a,image_b,kp_pairs)#cv2 shows image
    
    good = []
    for m,n in matches:
        if m.distance < .75*n.distance:
            good.append(m)
    
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_a[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_b[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    

    cv2.waitKey()
    cv2.destroyAllWindows()

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
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
    pass


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
    pass

