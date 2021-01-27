import cv2
import cProfile
import numpy as np
import time

img1 = cv2.imread("img0.jpg", 0)  # queryimage # left image
img2 = cv2.imread("img1.jpg", 0)  # trainimage # right image


orb = cv2.ORB_create(nfeatures=32)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# create BFMatcher object

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)


p = cProfile.Profile()
p.enable()
start = time.time()
# find the keypoints and descriptors with SIFT

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# Match descriptors.
matches = flann.knnMatch(des1.astype("float32"), des2.astype("float32"), k=2)
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for m, n in matches:
    pts2.append(kp2[m.trainIdx].pt)
    pts1.append(kp1[m.queryIdx].pt)


pts1 = np.int32(pts1[:16])
pts2 = np.int32(pts2[:16])
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

end = time.time()
print(f"Took {end - start}s")
p.disable()
p.print_stats()
