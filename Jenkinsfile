pipeline {
    agent any

    environment {
        // Securely pass your Roboflow API key as an environment variable
        ROBOFLOW_API_KEY = credentials('ROBOFLOW_API_KEY') // Replace with Jenkins credential ID
    }

    stages {
        stage('Setup Workspace') {
            steps {
                echo 'Cleaning up and setting up workspace...'
                // Clean workspace and prepare output directories
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
                // Set up a Python virtual environment and install required libraries
                sh '''
                python3 -m venv venv
                source venv/bin/activate
                pip install --upgrade pip
                pip install ultralytics roboflow matplotlib Pillow opencv-python-headless
                '''
            }
        }

        stage('Run Script') {
            steps {
                echo 'Running the Python script...'
                // Execute the Python script with the environment variables
                sh '''
                source venv/bin/activate
                python main.py
                '''
            }
        }

        stage('Archive Results') {
            steps {
                echo 'Archiving results...'
                // Archive the output artifacts for analysis
                archiveArtifacts artifacts: 'Output/**/*', allowEmptyArchive: true
            }
        }
    }

    post {
        always {
            echo 'Cleaning up workspace...'
            // Cleanup virtual environment after the pipeline completes
            sh 'rm -rf venv'
        }
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline execution failed. Check logs for details.'
        }
    }
}
