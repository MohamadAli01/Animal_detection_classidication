pipeline {
    agent any

    environment {
        // Roboflow API key and project details from the provided Python code
        ROBOFLOW_API_KEY = "7P6wSkFD6Zb39ZYTL84S"  
        WORKSPACE_NAME = "animal-class"
        PROJECT_NAME = "animal-class-cnxhg"
        MODEL_VERSION = "5"
        YOLO_MODEL = "yolo11m.pt"
    }

    stages {
        stage('Setup Workspace') {
            steps {
                echo 'Cleaning up and setting up workspace...'
                // Clean and prepare output directories
                sh '''
                rm -rf Output
                mkdir -p Output/Detected_Animals
                mkdir -p Output/Classified_Animals
                mkdir -p Output/Processed_Videos
                '''
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo 'Setting up Python virtual environment and installing dependencies...'
                // Set up Python virtual environment and install libraries
                sh '''
                python3 -m venv venv
                source venv/bin/activate
                pip install --upgrade pip
                pip install ultralytics roboflow matplotlib Pillow opencv-python-headless tkinter
                '''
            }
        }

        stage('Run Object Detection Script') {
            steps {
                echo 'Running the Python object detection and classification script...'
                // Save the Python code into a script file
                writeFile file: 'main.py', text: '''
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
model = YOLO("${YOLO_MODEL}")  # Use YOLOv11 model
rf = Roboflow(api_key="${ROBOFLOW_API_KEY}")  # Private API Key
project = rf.workspace("${WORKSPACE_NAME}").project("${PROJECT_NAME}")  # Workspace and Project Name
rf_model = project.version(${MODEL_VERSION}).model  # Use specified version of the model

# Set up output directories
output_dir_detected = "Output/Detected_Animals"
output_dir_classified = "Output/Classified_Animals"
output_dir_processed_video = "Output/Processed_Videos"

os.makedirs(output_dir_detected, exist_ok=True)
os.makedirs(output_dir_classified, exist_ok=True)
os.makedirs(output_dir_processed_video, exist_ok=True)

# Main function here would go (truncated for simplicity)

if __name__ == "__main__":
    print("Object detection script is running!")
                '''
                // Execute the script
                sh '''
                source venv/bin/activate
                python main.py
                '''
            }
        }

        stage('Archive Results') {
            steps {
                echo 'Archiving output results...'
                // Archive all output files
                archiveArtifacts artifacts: 'Output/**/*', allowEmptyArchive: true
            }
        }
    }

    post {
        always {
            node {
                echo 'Cleaning up workspace...'
                sh 'rm -rf venv Output'
            }
        }
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline execution failed. Check logs for details.'
        }
    }
}
