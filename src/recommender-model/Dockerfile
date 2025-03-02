# Use the appropriate Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install cron
RUN apt-get update && apt-get install -y cron

# Copy the necessary files into the container at /app
COPY dockerInference.py dockerTrain.py dockerRetrain.py utils.py start.sh query.py recommenderApp.py /app/

# Copy the .env file into the container
COPY .env /app/

# Install required Python packages with specific versions
RUN pip install tensorflow==2.15.0 scikit-learn==1.2.2 pandas==2.2.2 numpy==1.23.5 Flask==3.0.3

# Create an archive directory
RUN mkdir -p /app/archive

# Ensure the start script is executable
RUN chmod +x /app/start.sh

# Command to run the start script
CMD ["/app/start.sh"]
