# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from skimage import exposure
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
image = imutils.resize(image, height = 700)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
 

tl = refPt[0]
tr = [refPt[1][0],refPt[0][1]]
bl = [refPt[0][0],refPt[1][1]]
br = refPt[1]


(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)

# compute the midpoint between the top-left and top-right points,
# followed by the midpoint between the top-righ and bottom-right
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)


dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

pixelsPerMetric = dB / 8.27

# compute the size of the object
dimA = dA / pixelsPerMetric
dimB = dB / pixelsPerMetric

# draw the object sizes on the image
cv2.putText(image, "{:.2f}in".format(dimB),
	(int(tltrX - 35), int(tltrY - 30)), cv2.FONT_HERSHEY_SIMPLEX,
	.5, (100, 0, 0), 1)
cv2.putText(image, "{:.2f}in".format(dimA),
	(int(tlblX-70 ), int(tlblY)), cv2.FONT_HERSHEY_SIMPLEX,
	.5, (100, 0, 0), 1)

"""
cv2.setMouseCallback("image", click_and_crop) 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

tl = refPt[0]
tr = [refPt[1][0],refPt[0][1]]
bl = [refPt[0][0],refPt[1][1]]
br = refPt[1]

# compute the midpoint between the top-left and top-right points,
# followed by the midpoint between the top-righ and bottom-right
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)

dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

# compute the size of the object
dimA = dA / pixelsPerMetric
dimB = dB / pixelsPerMetric

# draw the object sizes on the image
cv2.putText(image, "{:.2f}in".format(dimB),
	(int(tltrX), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX,
	.5, (0, 0, 100), 1)
cv2.putText(image, "{:.2f}in".format(1),
	(int(tlblX-70 ), int(tlblY)), cv2.FONT_HERSHEY_SIMPLEX,
	.5, (0, 0, 100), 1)


cv2.imshow("image", image)

cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
"""
img_croped = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

#Color Boundary for Detecting Black Socks
bound_skin = [[0,0,0], [135,135,135]]

# create NumPy arrays from the boundaries
lower = np.array(bound_skin[0], dtype = "uint8")
upper = np.array(bound_skin[1], dtype = "uint8")


# find the colors within the specified boundaries and apply
# the mask
mask_feet = cv2.inRange(img_croped, lower, upper)
output_feet = cv2.bitwise_and(img_croped, img_croped, mask = mask_feet)

#Creating Threshold for Detecting Contours
ret,thresh_feet = cv2.threshold(mask_feet, 20, 255, 0)


(feet_cntr, _) = cv2.findContours(thresh_feet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(feet_cntr) != 0:
    #find the biggest area
    d = max(feet_cntr, key = cv2.contourArea)

    rect_feet = cv2.minAreaRect(d)
    feet_box = cv2.cv.BoxPoints(rect_feet) if imutils.is_cv2() else cv2.boxPoints(rect_feet)
    feet_box = np.array(feet_box, dtype="int")
    feet_box = perspective.order_points(feet_box)
    #cv2.drawContours(img_croped, [feet_box.astype("int")], -1, (0, 0, 255), 20)
 

(tl, tr, br, bl) = feet_box
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)

# compute the midpoint between the top-left and top-right points,
# followed by the midpoint between the top-righ and bottom-right
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)


#dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
width = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

#dimA = dA / pixelsPerMetric
feet_width = width / pixelsPerMetric

# draw the object sizes on the image
cv2.drawContours(img_croped, [feet_box.astype("int")], 0, (0, 0, 255), 2)
 
cv2.putText(image, "W:{:.2f}in".format(feet_width),
	(int(95), int(200 )
		), cv2.FONT_HERSHEY_SIMPLEX,
	.5, (0, 0, 255), 2)

cv2.imshow("image", image)
cv2.imshow("Cont",output_feet)
cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
