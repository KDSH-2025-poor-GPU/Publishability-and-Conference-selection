FROM pathwaycom/pathway:latest

# Install system dependencies required for PyMuPDF, PIL, pytesseract, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /usr/local/bin/wait-for-it.sh
RUN chmod +x /usr/local/bin/wait-for-it.sh

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install tensorflow
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install datasets
RUN pip install tf-keras
RUN pip install pathway[all]
RUN pip install "pathway[xpacks]"

# Copy the rest of your application code into the container
COPY main.py .
COPY notebook_loader.py .
COPY credentials.json . 
COPY decisionclassifier.joblib .

# Specify the command to run on container start
CMD ["wait-for-it.sh", "api:8000", "--", "python", "main.py"]
