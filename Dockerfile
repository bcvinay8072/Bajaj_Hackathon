# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required for OCR
# This installs Tesseract and Poppler (for pdf2image)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code into the container
COPY . /code/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the app. CMD is used to run the main command of the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]