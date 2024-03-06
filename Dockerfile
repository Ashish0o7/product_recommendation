# Use an appropriate base image for your Python environment
FROM python:3.12

# Update the package index and install necessary system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    # Add other dependencies if needed \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project files to the container
COPY . .

# Expose the port the app runs on (if needed)
EXPOSE 8000

# Specify the command to run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
