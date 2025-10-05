# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This is done first to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Command to run your application
# Hugging Face Spaces exposes port 7860 by default
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]