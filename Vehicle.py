                                # Real-Time Vehicle Detection and Counting Using MobileNet-SSD and Background Subtraction

# Importing Essential Libraries for Computer Vision
import cv2
import numpy as np
# This step imports the OpenCV library (cv2), a powerful tool for image and video processing, and NumPy (np), a fundamental package for numerical operations and array manipulation in Python. These libraries are foundational for building computer vision applications such as vehicle detection and tracking.



# Initializing the Deep Neural Network and Defining Class Labels
net = cv2.dnn.readNetFromCaffe(
    'C:\\Users\\basil\\Downloads\\deploy.prototxt',
    'C:\\Users\\basil\\Downloads\\MobileNet-SSD-master\\MobileNet-SSD-master\\mobilenet_iter_73000.caffemodel'
)

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# This step loads a pre-trained MobileNet-SSD model using OpenCV's DNN module. The model architecture is specified in the .prototxt file, while the learned weights are loaded from the .caffemodel file. Additionally, the list of object classes that the model can detect is defined, ranging from vehicles like cars and buses to animals and everyday objects.



# Setting Up Video Capture and Detection Parameters
count_line_position = 550
cap = cv2.VideoCapture('Test Video.mp4')

min_width = 80
min_height = 80
# This section sets the position of the counting line in the video frame, which will be used to count objects crossing that line. It then initializes the video capture from a file named 'Test Video.mp4'. The minimum width and height parameters define the size threshold for detected objects to be considered valid, helping to filter out small or irrelevant detections.



# Center Point Calculation and Initialization of Detection Variables
def center_handle(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

detect = []
offset = 6
count = 0
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
# This function calculates the center coordinates of a detected bounding box, which is essential for tracking objects as they move across frames. The code also initializes an empty list detect to store detected object centers, sets an offset value for counting accuracy near the counting line, and initializes a vehicle count variable. Additionally, it sets up a background subtractor algorithm (MOG) to help separate moving objects from the static background in the video.



# Real-time Vehicle Detection, Tracking, and Counting in Video Stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    image_sub = algo.apply(blur)
    dilat = cv2.dilate(image_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.line(frame, (150, count_line_position), (1200, count_line_position), (0, 0, 255), 3)

    for c in contours:
        #if cv2.contourArea(c) < 500:
            #continue
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_width and h >= min_height:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame, center, 4, (255, 0, 0), -1)

    for (x, y) in detect.copy():
        if (count_line_position - offset) < y < (count_line_position + offset):
            count += 1
            detect.remove((x, y)) 

    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300))) 
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2] 
        
        if confidence >0.2:
            idx = int(detections[0,0,i,1])
            label = classes[idx]
            
            if label in ['car']:
                
                box = detections[0,0,i,2,3:7]*np.array([w,h,w,h])
                (start_x, start_y, end_x, end_y) = box.astype('int') 
                
                cv2.rectangle(frame(start_x,start_y),(end_x,end_y), (0,255,0),2)
                cv2.putText(frame,label,(start_x,start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2) 
    
    cv2.putText(frame, f'Vehicles: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    print(f"Total Vehicles Counted: {count}")

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# This loop processes each frame from the video feed to detect and count vehicles. It first converts the frame to grayscale, applies Gaussian blur, and uses a background subtractor to isolate moving objects. Morphological operations clean the mask for better contour detection, and bounding boxes are drawn around sufficiently large objects. The center of each detected object is tracked to count vehicles crossing a predefined line. Simultaneously, a pre-trained MobileNet-SSD deep learning model performs object detection to specifically identify cars and mark them with bounding boxes and labels. The running count of vehicles is displayed on each frame, and the processed video feed is shown in a window. The loop continues until the video ends or the user presses 'q'.



# Release Resources and Close All OpenCV Windows
cap.release()
cv2.destroyAllWindows()
# After processing the video frames and detecting vehicles, this snippet ensures the proper release of the video capture resource and closes all OpenCV windows opened during the execution. This prevents resource locking and allows graceful program termination.

