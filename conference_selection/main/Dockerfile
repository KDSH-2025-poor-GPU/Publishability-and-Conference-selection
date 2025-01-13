# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies required for PyMuPDF, PIL, pytesseract, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY main.py . 
COPY credentials.json . 


# Set any necessary environment variables
ENV USE_OCR=False

# Specify the command to run on container start
CMD ["python", "main_script.py"]
