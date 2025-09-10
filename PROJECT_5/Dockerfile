# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy everything to the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8080 for Elastic Beanstalk
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
