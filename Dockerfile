# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures that pip doesn't store unnecessary cache, keeping the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code from your context to your container at /app
COPY . .

# Set the default command to run when the container starts.
# This will run the Acural Recursion trainer with the default settings from the config.
# Users can override this command, e.g., to run the baseline trainer or with different args.
CMD ["python", "acural_trainer.py"]
