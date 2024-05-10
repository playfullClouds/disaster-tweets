# Use Python 3.11 as the base image
FROM python:3.11-slim-buster

# set environment variables to prevent Python from buffering output
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /app

# Copy requirements.txt file to the working directory
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the entire project into the Docket image
COPY . /app

# Expose the port the app runs on
EXPOSE 5000

CMD ["python", "app.py"]