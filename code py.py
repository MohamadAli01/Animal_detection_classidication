import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
from datetime import datetime
from roboflow import Roboflow
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import threading

# Initialize YOLOv11 and Roboflow models
model = YOLO("yolo11m.pt")  # Use YOLOv11 model
rf = Roboflow(api_key="7P6wSkFD6Zb39ZYTL84S")  # Private API Key 
project = rf.workspace("animal-class").project("animal-class-cnxhg")  #  Workspace and Project Name
rf_model = project.version(5).model  # Use Version 5 of the Model diffrent version 1-5 excit but 5 recomended 

# Set up output directories
output_dir_detected = "Output/Detected_Animals"
output_dir_classified = "Output/Classified_Animals"
output_dir_processed_video = "Output/Processed_Videos"

os.makedirs(output_dir_detected, exist_ok=True) #no error if directories already exist.
os.makedirs(output_dir_classified, exist_ok=True)#no error if directories already exist.
os.makedirs(output_dir_processed_video, exist_ok=True)#no error if directories already exist.


# Function to save cropped images for each detection
def save_cropped_image(image_rgb, detection): #take the image 
    x1, y1, x2, y2, conf, cls_id = map(int, detection) # boxes coordinates
    cropped_image = image_rgb[y1:y2, x1:x2] # crop the image only inside the box 
    cropped_filename = f'object_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.jpg'
    cropped_path = os.path.join(output_dir_detected, cropped_filename)
    cv2.imwrite(cropped_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    return cropped_path


# GUI Setup
class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detection and Classification ") # title
        self.root.geometry("900x800")# window size
        self.root.configure(bg="#F5F5F5")  #  gray background

        # Header
        header = tk.Label(
            root,
            text="Animal Detection and Classification",
            font=("Arial", 20, "bold"),
            bg="#2E3B4E",
            fg="white",
            pady=10
        )
        header.pack(fill=tk.X)

        # Upload Section
        upload_frame = tk.Frame(root, bg="#F5F5F5")
        upload_frame.pack(pady=20)
        upload_button = ttk.Button(upload_frame, text="Upload Image or Video", command=self.upload_file)
        upload_button.pack()

        # Threshold Sliders
        threshold_frame = tk.Frame(root, bg="#F5F5F5")
        threshold_frame.pack(pady=20, fill=tk.X)

        # Detection Threshold
        detection_label = tk.Label(
            threshold_frame, text="Detection Threshold:", font=("Arial", 12), bg="#F5F5F5"
        )
        detection_label.grid(row=0, column=0, padx=10, sticky="w")

        self.detection_threshold = tk.DoubleVar(value=0.1)
        detection_slider = ttk.Scale(
            threshold_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.detection_threshold,
            command=self.update_threshold_label,
        )
        detection_slider.grid(row=0, column=1, padx=10, sticky="we")
        self.detection_threshold_label = tk.Label(
            threshold_frame,
            text=f"{self.detection_threshold.get():.2f}",
            font=("Arial", 12),
            bg="#F5F5F5"
        )
        self.detection_threshold_label.grid(row=0, column=2, padx=10)

        # Classification Threshold
        classification_label = tk.Label(
            threshold_frame, text="Classification Threshold:", font=("Arial", 12), bg="#F5F5F5"
        )
        classification_label.grid(row=1, column=0, padx=10, sticky="w")

        self.classification_threshold = tk.DoubleVar(value=0.5)
        classification_slider = ttk.Scale(
            threshold_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.classification_threshold,
            command=self.update_threshold_label,
        )
        classification_slider.grid(row=1, column=1, padx=10, sticky="we")
        self.classification_threshold_label = tk.Label(
            threshold_frame,
            text=f"{self.classification_threshold.get():.2f}",
            font=("Arial", 12),
            bg="#F5F5F5"
        )
        self.classification_threshold_label.grid(row=1, column=2, padx=10)

        threshold_frame.columnconfigure(1, weight=1)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=900, mode="determinate")
        self.progress_bar.pack(pady=20)

        # Canvas for Image Display
        self.canvas_frame = tk.Frame(self.root, bg="#D3D3D3", relief=tk.GROOVE, bd=2)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#FFFFFF")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # To store the last uploaded image path
        self.last_image_path = None

    def upload_file(self): # let the user browse an image or video to upload must be jpg/png/mp4 file only 
        file_path = filedialog.askopenfilename(filetypes=[("Image/Video files", "*.jpg *.png *.mp4")])
        if file_path:
            if file_path.endswith(('.jpg', '.png')):
                self.last_image_path = file_path
                self.process_image(file_path)
            elif file_path.endswith('.mp4'):
                threading.Thread(target=self.process_video, args=(file_path,)).start()
            else:
                messagebox.showerror("Invalid File", "Please upload a .jpg, .png, or .mp4 file.")

    def process_image(self, image_path): # pass the image to yolo then crop the detection and sent it to roboflow and display results on gui 
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Processing Image with YOLO...")
        results = model(image_rgb)
        processed_img_path = self.detect_and_classify(image_rgb, results)
        self.display_image(processed_img_path)

    def process_video(self, video_path): # same as images 
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        # Define the output video path
        output_video_path = os.path.join(output_dir_processed_video, f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4")
        out = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height)
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            if results and hasattr(results[0], 'boxes') and results[0].boxes.data.numel() > 0:
                detections = results[0].boxes.data
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det[:6].tolist()
                    conf = float(conf)

                    if conf >= self.detection_threshold.get():
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cropped_image = frame_rgb[y1:y2, x1:x2]

                        # Save cropped image temporarily
                        cropped_filename = f'temp_cropped_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.jpg'
                        cropped_path = os.path.join(output_dir_detected, cropped_filename)
                        cv2.imwrite(cropped_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                        try:
                            prediction = rf_model.predict(cropped_path).json()
                            print(f"Roboflow Prediction Response: {prediction}")

                            if "predictions" in prediction and len(prediction["predictions"]) > 0:
                                top_prediction = prediction["predictions"][0]
                                obj_class = top_prediction.get("class", top_prediction.get("top", "unknown"))
                                confidence = top_prediction.get("confidence", 0)

                                if confidence >= self.classification_threshold.get():
                                    label = f"{obj_class} ({confidence * 100:.2f}%)"
                                    color = (0, 0, 255) if obj_class.lower() == "coyote" else (0, 255, 0)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception as e:
                            print(f"Error classifying cropped image: {e}")

            out.write(frame)
            processed_frames += 1
            self.progress_bar["value"] = (processed_frames / frame_count) * 100
            self.root.update()

        cap.release()
        out.release()
        self.progress_bar["value"] = 0
        messagebox.showinfo("Processing Complete", f"Processed video saved to {output_video_path}")

    def detect_and_classify(self, image_rgb, results):
        detection_threshold = self.detection_threshold.get()
        classification_threshold = self.classification_threshold.get()

        plt.figure(figsize=(12, 8), dpi=150)
        plt.imshow(image_rgb)
        ax = plt.gca()

        if results and hasattr(results[0], 'boxes') and results[0].boxes.data.numel() > 0:
            detections = results[0].boxes.data
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det[:6].tolist()
                if conf >= detection_threshold:
                    try:
                        cropped_path = save_cropped_image(image_rgb, [x1, y1, x2, y2, conf, cls_id])
                        prediction = rf_model.predict(cropped_path).json()
                        print(f"Roboflow Prediction Response: {prediction}")

                        if "predictions" in prediction and len(prediction["predictions"]) > 0:
                            top_prediction = prediction["predictions"][0]
                            obj_class = top_prediction.get("class", top_prediction.get("top", "unknown"))
                            confidence = top_prediction.get("confidence", 0)

                            if confidence >= classification_threshold:
                                color = "red" if obj_class.lower() == "coyote" else "green"
                                rect = plt.Rectangle(
                                    (x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor=color, facecolor="none"
                                )
                                ax.add_patch(rect)
                                ax.text(
                                    x1, y1, f"{obj_class}: {confidence * 100:.2f}%",
                                    color="Black", fontsize=14, backgroundcolor=color
                                )
                    except Exception as e:
                        print(f"Error during classification: {e}")

        plt.axis("off")
        output_path = os.path.join(output_dir_classified, f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        return output_path

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((900, 500), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def update_threshold_label(self, event=None):
        self.detection_threshold_label.config(text=f"{self.detection_threshold.get():.2f}")
        self.classification_threshold_label.config(text=f"{self.classification_threshold.get():.2f}")


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
