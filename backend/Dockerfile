# Use an official Python runtime as a parent image
# Matching your development environment (3.11 based on previous info)
FROM python:3.11-slim

# --- Add this section to update OS packages ---
# Update package lists and upgrade installed packages to patch vulnerabilities
# Run as root (default user initially)
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
# --- End added section ---

# Set the working directory in the container
WORKDIR /app

# --- Dependency Installation ---
# Copy ONLY the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
# --default-timeout=100 can help with slow downloads (like torch)
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# --- Application Code ---
# Copy the rest of the backend application code into the container
# .dockerignore file will ensure venv, .env etc. are excluded
COPY . .

# --- Runtime ---
# Make port 8000 available inside the container network (doesn't publish to host yet)
EXPOSE 8000

# Define the command to run your application when the container starts
# Use 0.0.0.0 to allow connections from outside the container's network interface
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]