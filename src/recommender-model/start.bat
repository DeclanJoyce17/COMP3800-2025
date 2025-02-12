@echo off

REM Cron job is running every minute
echo Cron job has started and is running every minute!

REM Function to handle initial setup (train and inference)
:initial_setup
echo Executing training and initial inference
python query.py
python dockerTrain.py
python dockerInference.py

REM Check if it's the first run
call :initial_setup

REM Create the log file
type NUL > C:\recommenderApp\cron.log

REM Schedule the task to run every minute
echo Scheduling the task
schtasks /create /tn "RecommenderCronJob" /tr "C:\recommenderApp\recommenderApp.bat" /sc minute /mo 1 /f

REM Start the Flask server in the background
echo Starting recommenderApp.py
start /B python recommenderApp.py

REM Continuously keep the script running and display the log
echo Displaying the log
powershell -command "Get-Content C:\recommenderApp\cron.log -Wait"

:eof
