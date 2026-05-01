# 1. Start with a lightweight, official Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the virtual container
WORKDIR /app

# 3. Install critical system tools (OpenCV needs these to process video)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file and install Python libraries
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt



# 5. Copy the rest of your SentinAI code into the container
COPY . .

# 6. Expose the port so we can see the web dashboard
EXPOSE 5000

# 7. Start the Master Command Center!
CMD ["python", "src/app.py"]