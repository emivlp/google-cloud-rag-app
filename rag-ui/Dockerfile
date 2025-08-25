# Use the official Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application code
COPY app.py ./

# Expose the port Streamlit will run on
EXPOSE 8080

# The command to run the Streamlit app
# It will automatically use the PORT environment variable provided by Cloud Run
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true"]
