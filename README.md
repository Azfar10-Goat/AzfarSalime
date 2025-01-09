1. Approach to Solving the Task:
The task uses a YOLOv11 model to detect objects, specifically focusing on tracking and counting cars as they cross a defined vertical line in a video. Each car is uniquely identified using its bounding box coordinates, and duplicate counts are avoided by maintaining a tracking dictionary to monitor their positions and counting state.
2. Technologies and Tools Used:
YOLOv11: For real-time object detection and classification.
OpenCV: For video processing, including reading video frames, drawing bounding boxes, lines, and writing output videos.
Python: To implement the algorithm for car detection, tracking, and counting.
3. Steps to Reproduce the Results:
Install the required libraries: ultralytics (for YOLO) and opencv-python.
Replace "yolo11s.pt" with the path to your trained YOLOv11 model file.
Set the video_path variable to the path of the input video.
Run the script to detect cars, track their movements, and count them as they cross the vertical line in the video.
The output is saved as output_video_fixed.avi with visual indicators such as bounding boxes, a vertical line, and car count displayed on each frame.

