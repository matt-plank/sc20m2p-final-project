FROM tensorflow/tensorflow:latest-gpu

# Install dependencies first (they are least likely to change)
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the files (descending order of change frequency)
COPY trainingImages trainingImages
COPY transferEngine transferEngine
COPY main.py main.py

CMD ["python", "main.py"]
