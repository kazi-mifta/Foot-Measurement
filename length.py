import numpy as np
import argparse
import cv2
import imutils
from imutils import perspective
from imutils import contours
from skimage import exposure
from scipy.spatial import distance as dist
 
def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")


args = vars(ap.parse_args())
 
# load the image
image = cv2.imread(args["image"])




# define the list of boundaries
bound_wht = [[140,140,140], [255, 255, 255]]

# create NumPy arrays from the boundaries
lower = np.array(bound_wht[0], dtype = "uint8")
upper = np.array(bound_wht[1], dtype = "uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(image, lower, upper)

output = cv2.bitwise_and(image, image, mask = mask)


ret,thresh = cv2.threshold(mask, 40, 255, 0)
(contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    #find the biggest area
    c = max(contours, key = cv2.contourArea)

    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 20)
 



# unpack the ordered bounding box, then compute the midpoint
# between the top-left and top-right coordinates, followed by
# the midpoint between bottom-left and bottom-right coordinates
(tl, tr, br, bl) = box
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)

# compute the midpoint between the top-left and top-right points,
# followed by the midpoint between the top-righ and bottom-right
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)


dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))



pixelsPerMetric = dB / 11.69


# compute the size of the object
dimA = dA / pixelsPerMetric
dimB = dB / pixelsPerMetric

# draw the object sizes on the image
cv2.putText(image, "{:.2f}in".format(dimB),
	(int(tltrX - 35), int(tltrY - 30)), cv2.FONT_HERSHEY_SIMPLEX,
	5, (100, 255, 100), 12)







#Image containg Just A4 Page
img_croped = image[int(tl[1])+5:int(bl[1])-40,int(tl[0])+10:int(tr[0])-10]

#Color Boundary for Detecting Black Socks
bound_blk = [[0,0,0], [108,108,108]]

# create NumPy arrays from the boundaries
lower = np.array(bound_blk[0], dtype = "uint8")
upper = np.array(bound_blk[1], dtype = "uint8")


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
    cv2.drawContours(img_croped, [feet_box.astype("int")], -1, (0, 0, 255), 20)
 

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
cv2.drawContours(img_croped, [feet_box.astype("int")], -1, (0, 255, 0), 2)
 

cv2.putText(image, "H:{:.2f}in".format(feet_width),
	(int(95), int(200 )
		), cv2.FONT_HERSHEY_SIMPLEX,
	5, (0, 0, 255), 12)

#show the images
cv2.imshow("images",imutils.resize(image, height = 300))
cv2.imshow("gray",imutils.resize(output_feet, height = 300))
cv2.waitKey(0)
