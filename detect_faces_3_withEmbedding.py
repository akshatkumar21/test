import numpy as np
import cv2

# Parameters
imageLocation = r"test.jpg"
prototxtLocation = "deploy.prototxt.txt"
caffeModel = "model.caffemodel"
thresholdConfidence = 0.9


net = cv2.dnn.readNetFromCaffe(prototxtLocation, caffeModel)

image = cv2.imread(imageLocation)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    
	confidence = detections[0, 0, i, 2]
	
	if confidence > thresholdConfidence:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            roi = image[startY:endY, startX:endX]


#Resize to 96*96 as model will take input image as 96*96
img = cv2.resize(roi,(96,96))
#cv2.imwrite(imageLocation+"_detected.jpg",img)


from model import create_model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

#Embedding
img = (img / 255.).astype(np.float32)
embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
embedded = list(embedded)
print(embedded)



