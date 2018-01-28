# import necessary packages
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
from skimage import exposure
import numpy as np
import argparse
import cv2


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.cv.BoxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q1", "--query1", required = True,
	help = "Path to the query image for Feet Width")
args = vars(ap.parse_args())

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["query1"])
ratio = image.shape[0] / 300.0
orig = image.copy()
image = imutils.resize(image, height = 300)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
gray = cv2.medianBlur(gray,5)
edged = cv2.Canny(gray, 30, 200)
 

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
A4 = None


for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then
	# we can assume that we have found our A4 Page
	if len(approx) >= 4:
		A4 = approx
		break

rect = cv2.minAreaRect(A4)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.array(box, dtype="int")

# order the points in the contour such that they appear
# in top-left, top-right, bottom-right, and bottom-left
# order, then draw the outline of the rotated bounding
# box
box = perspective.order_points(box)
#cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

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



pixelsPerMetric = dB / 8.27


# compute the size of the object
dimA = dA / pixelsPerMetric
dimB = dB / pixelsPerMetric

# draw the object sizes on the image
cv2.putText(image, "{:.2f}in".format(dimB),
	(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.45, (255, 255, 255), 2)
cv2.putText(image, "{:.2f}in".format(dimA),
	(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
	0.45, (255, 255, 255), 2)




img_croped = image[int(tl[1])+5:int(bl[1])-40,int(tl[0])+10:int(tr[0])-10]


edge_feet = cv2.Canny(img_croped, 50, 100)
edge_feet = cv2.dilate(edge_feet, None, iterations=1)
edge_feet = cv2.erode(edge_feet, None, iterations=1)
 


(feet_cntr, _) = cv2.findContours(edge_feet.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
feet = None

for d in feet_cntr:
	#This Section is 
	peri = cv2.arcLength(d, True)
	approx = cv2.approxPolyDP(d, 0.01 * peri, True)

	if cv2.contourArea(d) > 30:
		feet = d
		break

feet_rect = cv2.minAreaRect(feet)
feet_box = cv2.cv.BoxPoints(feet_rect) if imutils.is_cv2() else cv2.boxPoints(feet_rect)
feet_box = np.array(feet_box, dtype="int")
feet_box = perspective.order_points(feet_box)
#cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

# unpack the ordered bounding box, then compute the midpoint
# between the top-left and top-right coordinates, followed by
# the midpoint between bottom-left and bottom-right coordinates
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
 

cv2.putText(img_croped, "{:.2f}in".format(feet_width),
	(int(5), int(20 )), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (0, 0, 255), 2)

cv2.imshow("Original",gray)
cv2.imshow("Edges",edge_feet)
cv2.imshow("Feet",img_croped)
cv2.waitKey(0)

















'''

# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then
	# we can assume that we have found our A4 Page
	if len(approx) == 4:
		A4 = approx
		break


#Extracting A4
box = cv2.minAreaRect(A4)
box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)



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


# compute the Euclidean distance between the midpoints
dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

# pixelPerMetric is 8.27 inch as A4 standard width
pixelsPerMetric = dB / 8.27


# compute the size of the A4
dimA = dA / pixelsPerMetric
dimB = dB / pixelsPerMetric

# draw the object sizes on the image
cv2.putText(image, "{:.3f}in".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
cv2.putText(image, "{:.3f}in".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)



#Now we got the A4 we will calculate every possible contour inside it
#First Crop from extracted box information, we will use same methodology
#as used in A4 detection, naming is almost similar for understanding purpose
cropped = image[int(box[0][1]):int(box[2][1]),int(box[0][0]):int(box[2][0])]
grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
grayscale = cv2.bilateralFilter(grayscale, 11, 17, 17)
edges = cv2.Canny(grayscale, 30, 200)
edges = cv2.dilate(edges, None, iterations=1)
edges = cv2.erode(edges, None, iterations=1)

contr = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
contr = contr[0] if imutils.is_cv2() else contr[1]

(contr, _) = contours.sort_contours(contr)

footPrint = None

#Detecting FootPrint
for A in contr:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(A) < 1000:
        continue
	footPrint = A


boxed = cv2.minAreaRect(A)
boxed = cv2.cv.BoxPoints(boxed) if imutils.is_cv2() else cv2.boxPoints(boxed)
boxed = np.array(boxed, dtype="int")
boxed = perspective.order_points(boxed)
#Outline of FootPrint
cv2.drawContours(cropped, [boxed.astype("int")], -1, (255, 0, 0), 2)
#Circles as Node Points on Bounding Box
for (x, y) in boxed:
    cv2.circle(cropped, (int(x), int(y)), 5, (0, 0, 255), -1)

#a,b,c,d goes as topleft,topright,bottomright,bottomleft Co-ordinate Point 
(a,b,c,d) = boxed    
(topmidX,topmidY) = midpoint(a,b)
(btmmidX,btmmidY) = midpoint(c,d)

(lftmidX,lftmidY) = midpoint(a,d)
(rgtmidX,rgtmidY) = midpoint(b,c)


disA = dist.euclidean((topmidX, topmidY), (btmmidX,btmmidY))/pixelsPerMetric
disB = dist.euclidean((lftmidX,lftmidY), (rgtmidX,rgtmidY))/pixelsPerMetric
cv2.putText(image, "H: {:.3f}in".format(disA),
        (int(5), int(50)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 0), 2)
cv2.putText(image, "W: {:.3f}in".format(disB),
        (int(5), int(80)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 0), 2)

cv2.imwrite("output.jpg", image)
#python test.py --query1 test.jpg
'''