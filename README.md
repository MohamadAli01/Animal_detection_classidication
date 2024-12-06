#Animal Detection and Classification GUI Application  
  
Overview  
This application allows users to detect and classify animals in images and videos using a graphical user interface (GUI). It leverages YOLOv11 for object detection and Roboflow for object classification. The app supports both images (.jpg, .png) and videos (.mp4), providing visual feedback and adjustable detection thresholds.  
  
Features  
Image and Video Support: Users can upload images or videos for processing.  
Detection Threshold Control: Adjust the YOLOv11 detection confidence threshold.  
Classification Threshold Control: Set the confidence threshold for classification via Roboflow.  
Visual Feedback: Displays processed images with bounding boxes and labels.  
Progress Bar: Shows the processing status for videos.  
  
Library Dependencies  
The application requires the following Python libraries:  
os: For managing file paths and directories.  
cv2 (OpenCV): For image and video processing.  
tkinter: For creating the GUI.  
Pillow: For image manipulation and display.  
matplotlib: For creating and saving processed images with annotations.  
ultralytics: For YOLOv11 object detection.  
datetime: For generating unique filenames.  
roboflow: For animal classification via Roboflow API.  
threading: For running tasks without freezing the GUI.    
  
Installation: 
Python: Ensure you have Python 3.8 or later installed. You can download it from python.org.  
Install Required Libraries: Run the following command to install the necessary libraries:  
pip install opencv-python-headless pillow matplotlib ultralytics roboflow  
Verify Installation: Test if the libraries are correctly installed by running:  
python -c "import cv2, tkinter, PIL, matplotlib, ultralytics, roboflow"  
No errors should be displayed.  
Set Up the YOLOv11 Model: Download the yolo11m.pt model file and ensure it is in the same directory as the script.  
API Key for Roboflow:  7P6wSkFD6Zb39ZYTL84S , workspace("animal-class") , project("animal-class-cnxhg"), version 5  
  
Usage Instructions  
Run the Application: Execute the script:  
python app.py  
Upload an Image or Video:  
Click the Upload Image or Video button.  
Select an image file (.jpg or .png) or a video file (.mp4).  
  
Adjust Thresholds:  
Use the Detection Threshold slider to adjust YOLO's detection confidence threshold.  
Use the Classification Threshold slider to adjust Roboflow's classification confidence threshold.  
  
View Results:  
For images: The processed image will be displayed in the canvas area with bounding boxes and classification labels.  
For videos: The application processes the video frame-by-frame, showing progress via a progress bar.  
  
Output Files:  
Cropped animal detections are saved in the Output/Detected_Animals directory.  
Processed images with annotations are saved in the Output/Classified_Animals directory.  
  
Directory Structure  
  
app.py                  # Main Python script  
yolo11m.pt              # YOLOv11 model file  
Output  
Detected_Animals    # Directory for cropped detections  
Classified_Animals  # Directory for annotated images  
  
Known Issues  
Progress Bar: The progress bar updates during video processing but simulates progress for images since image processing is quick.  
  
Error Handling: Ensure the Roboflow API key is valid and there is internet access for classification.  
  
Future Enhancements  
Support for additional file formats.  
Enhanced error handling for unsupported or corrupted files.  
Asynchronous processing for faster performance.  
 
License 
This project is open-source and licensed under the MIT License. 
 
Author 
[Mohamad Ali] 
Feel free to reach out for suggestions or issues. 
