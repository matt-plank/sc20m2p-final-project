# Uses the TensorFlow Docker image as a base and installs dependencies from requirements.txt
# Creates an environment for training models

FROM tensorflow/tensorflow:latest-gpu

# Install dependencies first (they are least likely to change)
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /transferEngine
