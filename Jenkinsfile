pipeline {
    agent any

    environment {
        // Roboflow API key and project details
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
model = YOLO("${YOLO_MODEL}")
rf = Roboflow(api_key="${ROBOFLOW_API_KEY}")
project = rf.workspace("${WORKSPACE_NAME}").project("${PROJECT_NAME}")
rf_model = project.version(${MODEL_VERSION}).model

print("Object detection script is running!")
                '''
                sh '''
                source venv/bin/activate
                python main.py
                '''
            }
        }

        stage('Archive Results') {
            steps {
                echo 'Archiving output results...'
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
