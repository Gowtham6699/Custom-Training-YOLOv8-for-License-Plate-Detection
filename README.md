# License Plate Detection using YOLOv5

## Description
This Python script detects license plates in images using a custom-trained YOLOv5 model. It displays the detected license plate number above the detected car, along with a bounding box around the license plate.

## Installation
1. Install the required packages:
pip install opencv-python
pip install cvzone
pip install git+https://github.com/ultralytics/yolov5.git


## Usage
1. Run the script with the following command:
python license_plate_detection.py

2. Provide the path to the image file as an argument:
python license_plate_detection.py --image_path '/path/to/image.jpg'


## Code
```python
from google.colab.patches import cv2_imshow
import cv2
import cvzone
import math
from ultralytics import YOLO

# Path to the image file
image_path = '/content/Tata-Zica-Front-Bumper-1-600x400.jpg'

# Load the image
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (1080, 720))  # Resize the image if necessary

# Load the YOLOv5 model
model = YOLO('/content/best_model.pt')
classnames = ['license-plate', 'vehicle']

# Perform object detection
results = model(frame)

# Process detection results
for info in results:
    parameters = info.boxes
    for box in parameters:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = box.conf[0]
        class_detect = box.cls[0]
        class_detect = int(class_detect)
        class_detect = classnames[class_detect]
        conf = math.ceil(confidence * 100)
        if conf > 50 and class_detect == 'license-plate':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)

# Display the processed image
cv2_imshow(frame)




