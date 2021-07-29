# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
'''
ap.add_argument("-p", "--prototxt", required=True,
                help="C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Research "
                     "project\\")
ap.add_argument("-m", "--model", required=True,
                help="C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Research "
                     "project\\")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
'''
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

path1 = "C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Research project\\MobileNetSSD_deploy.prototxt"
path2 = "C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Research project\\MobileNetSSD_deploy.caffemodel"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(path1, path2)
print("[INFO] Model Loaded...")
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter


#Function for object detection when an image is uploaded

def object_detection(image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # draw the prediction on the frame
                if CLASSES[idx] not in CLASSES:
                    print("Object is not Present")
                else:
                    print("Object is Present")
                    if CLASSES[idx] in ['person']:
                        category = 'Human'
                    else:
                        category = 'Non-Human'   
                    label = "{}: {:.2f}%".format(category,
                                                confidence * 100)
                    cv2.rectangle(image, (startX, startY), (endX, endY),
                                COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        return(image)



print('\n\n Choose Input Mode: Image,Live_video')
x = input("Enter Input choice: ")
if x == "Live_video":
    print('Input chosen in Live_video')
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                    0.007843, (300, 300), 127.5)
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # draw the prediction on the frame
                if CLASSES[idx] not in CLASSES:
                    print("Object is not Present")
                else:
                    print("Object is Present")
                    if CLASSES[idx] in ['person']:
                        category = 'Human'
                    else:
                        category = 'Non-Human'   
                    label = "{}: {:.2f}%".format(category,
                                                confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    cv2.imshow("Frame", frame)

        #cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # update the FPS counter
    fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
else:
    imageFileName = input("Enter the image name with absolute path ")
    Img = cv2.imread(imageFileName)
    plt.imshow(object_detection(Img))
    plt.show()


#C:\\Users\\212626492\\Box\\H drive\\2020_desktop_dec5\\REVA\\Research project\\testimages\\sample_person.jpg
