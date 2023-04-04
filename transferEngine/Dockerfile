# TensorFlow can be really hard to set up to work with a GPU on some devices
# This Dockerfile makes that process a lot easier

FROM tensorflow/tensorflow:latest-gpu

# Install dependencies first (they are least likely to change)
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /transferEngine
