#!/usr/bin/env python
# coding: utf-8

# In[74]:


from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load YOLO11s model
model = YOLO("yolo11s.pt")  # Path to your YOLO11s model

# Define video path
video_path = "C:\\Users\\user\\Downloads\\car.mp4"
cap = cv2.VideoCapture(video_path)

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "output_video_fixed.avi", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4)))
)

# Define vertical line position
line_position = int(cap.get(3)) // 2  # Middle of the frame
car_count = 0

# Track cars to avoid duplicate counts
tracked_objects = defaultdict(lambda: {"counted": False, "last_position": None})

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with YOLO
    results = model(frame, conf=0.3)

    # Extract detections
    detections = results[0].boxes
    cars_in_frame = 0
    current_frame_objects = {}

    for detection in detections:
        # Get bounding box and class
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        cls = int(detection.cls[0])  # Class ID
        conf = detection.conf[0]

        if cls == 2:  # Car class ID (adjust based on your model)
            cars_in_frame += 1

            # Calculate center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw bounding box and center point
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Create a unique ID for the car
            car_id = f"{x1}-{y1}-{x2}-{y2}"

            # Add to current frame objects
            current_frame_objects[car_id] = center_x

            # If car is new, initialize it in tracked_objects
            if car_id not in tracked_objects:
                tracked_objects[car_id] = {"counted": False, "last_position": center_x}

            # Check if the car crosses the line
            if (
                tracked_objects[car_id]["last_position"] < line_position
                and center_x >= line_position
                and not tracked_objects[car_id]["counted"]
            ):
                car_count += 1
                tracked_objects[car_id]["counted"] = True
                print(f"Car {car_id} crossed the line! Total Car Count: {car_count}")

            # Update last position of the car
            tracked_objects[car_id]["last_position"] = center_x

    # Remove cars not in the frame
    tracked_objects = {
        k: v for k, v in tracked_objects.items() if k in current_frame_objects
    }

    # Draw the vertical red line
    cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (0, 0, 255), 2)

    # Display metrics
    cv2.putText(
        frame,
        f"Total Car Count: {car_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Cars in Frame: {cars_in_frame}",
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    # Write frame to the output video
    out.write(frame)

    # Show frame
    cv2.imshow("Car Counter Fixed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




