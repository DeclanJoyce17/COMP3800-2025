#!/bin/bash

#cron job is running every minute
echo "Cron job has started and is running at 1AM PST everyday!"

# Function to handle initial setup (train and inference)
initial_setup() {
    echo "Executing training and initial inference"
    python /app/query.py
    python /app/dockerTrain.py
    python /app/dockerInference.py
}

# Check if it's the first run
initial_setup

# Write out the cron job to a temporary file
echo "0 1 * * * /usr/bin/python /app/recommenderApp.py >> /var/log/cron.log 2>&1" > /etc/cron.d/recommender_cron

# Give execution rights to the cron job file
chmod 0644 /etc/cron.d/recommender_cron

# Apply cron job
crontab /etc/cron.d/recommender_cron

# Create the log file to be able to run tail
touch /var/log/cron.log

# Start cron
echo "Starting cron service"
service cron start

# Start the Flask server in the background
echo "Starting server.py"
python /app/recommenderApp.py &

# Continuously keep the container running and display the log
tail -f /var/log/cron.log
