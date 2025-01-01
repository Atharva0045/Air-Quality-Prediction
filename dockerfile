FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the src directory contents into the container at /app
COPY src/ /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# run app.py when the container launches
CMD ["python", "app.py"]