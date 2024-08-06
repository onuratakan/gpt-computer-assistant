# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY --link . .

# Expose port 8080 
EXPOSE 8080

# Create a non-root user and switch to it
RUN useradd -m app_user
USER app_user

# Run the application
CMD ["python", "gpt_computer_assistant.py"]
