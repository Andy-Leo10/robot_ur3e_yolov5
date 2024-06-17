import torch

# Check if CUDA (GPU support) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Model linux
model = torch.hub.load("/home/sinso/Downloads/GITHUB/yolov5", 
                       "custom", 
                       path="/home/sinso/Downloads/GITHUB/yolov5/runs/train/exp/weights/best.pt", 
                       source="local").to(device)

# Images
img = "/home/sinso/Downloads/GITHUB/robot_ur3e_yolov5/for_testing/photo_1718317391.jpg"
# img = "/home/sinso/Downloads/GITHUB/robot_ur3e_yolov5/for_testing/coffee.png"
# Inference
results = model(img)

# Results
print('-------------------------------------------------------Results')
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# Display the image with detections
results.show()
print('-------------------------------------------------------coordinates')
# Access the bounding box coordinates and other information
detections = results.xyxy[0]  # Detections for the first image
print("Detections (x1, y1, x2, y2, confidence, class):")
print(detections)
print('-------------------------------------------------------separating')
# Optionally, you can iterate over the detections and print or use them as needed
for detection in detections:
    x1, y1, x2, y2, confidence, cls = detection
    print(f"Box coordinates: {x1}, {y1}, {x2}, {y2}. Confidence: {confidence}. Class: {cls}")





# draw the results on the original image using opencv
import cv2
# Load the original image
original_img = cv2.imread(img)

confidence_threshold = 0.75  # Set your desired confidence threshold here
# Filter detections based on the confidence threshold
filtered_detections = [detection for detection in detections if detection[4] > confidence_threshold]

# Correctly unpack the detection tuple
for detection in filtered_detections:
    x1, y1, x2, y2, confidence, cls_id = detection[:6].cpu().numpy()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    confidence = float(confidence)
    cls = int(cls_id)  # Convert class id to int if it's not already
    cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(original_img, f'Class: {cls}, Conf: {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 0, 255), 2)

cv2.putText(original_img, f'Image Size: {original_img.shape[1]}x{original_img.shape[0]}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# Display the image with bounding boxes
cv2.imshow('Detected Objects', original_img)
cv2.waitKey(0)  # Wait for a key press to close the displayed image
cv2.destroyAllWindows()  # Close all OpenCV windows
