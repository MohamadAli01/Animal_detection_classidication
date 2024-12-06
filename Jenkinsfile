pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building the application...'
                // Add your build commands here (e.g., `mvn clean install` for Maven or `npm install` for Node.js)
            }
        }

        stage('Test') {
            steps {
                echo 'Running tests...'
                // Add your testing commands here
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying the application...'
                // Add your deployment commands here
            }
        }
    }
}
