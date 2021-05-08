import numpy as np
import argparse
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")

ap.add_argument("-o" , "--output",required=True,help = "path to output directory")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

labelpath = os.path.sep.join([args["yolo"],"coco.names"])
labels= open(labelpath).read().strip().split("\n")

np.random.seed(42)
colors=np.random.randint(0,255,size = (len(labels),3),dtype ="uint8")


weightpath=os.path.sep.join([args["yolo"], "yolov3.weights"])
configpath= os.path.sep.join([args["yolo"],"yolov3.cfg"])

print("[INFO] loading YOLO from disk ...")
net = cv2.dnn.readNetFromDarknet(configpath, weightpath)

image = cv2.imread(args["image"])
image = cv2.resize(image, (580, 340),
               interpolation = cv2.INTER_NEAREST)
(H,W) = image.shape[:2]

ln = net.getLayerNames()
ln =[ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416) , swapRB = True , crop = False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took {:.6f} seconds".format(end-start))


boxes = []
confidences = []
classIDs = []


for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])


if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in colors[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
# show the output image

cv2.imwrite(args["output"],image)
cv2.imshow("Image", image)

cv2.waitKey(0)