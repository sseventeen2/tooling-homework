Streamlit Project with Docker

This project is a Streamlit web application that has been containerized using Docker. Follow the instructions below to set up and run the project using Docker on your local machine.
Prerequisites

Make sure you have Docker installed on your system

To run the project
1. First, clone this repository to your local machine.

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Build the Docker Image

To build the Docker image for the Streamlit app, run the following command in the project directory:

docker build -t streamlit .

This command will create a Docker image named streamlit using the Dockerfile provided in the project.

3. Run the Docker Container

After the image is built, you can run the container using the following command:

docker run -p 8502:8502 streamlit

This command will map port 8502 on your local machine to port 8502 inside the container. Make sure to use port 8502 (not the default Streamlit port 8501).
Step 4: Access the Streamlit App

Once the container is running, you can access the app by opening a web browser and navigating to either of the following addresses:

    http://localhost:8502
    http://0.0.0.0:8502

If one of these addresses does not work, try the other one.