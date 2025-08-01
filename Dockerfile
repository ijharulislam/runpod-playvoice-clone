FROM python:3.11-slim

WORKDIR /

# Install dependencies
RUN pip install --no-cache-dir runpod

# Copy your handler file
COPY rp_handler.py /
COPY requirements.txt /
COPY playdiffusion /playdiffusion
RUN pip install --no-cache-dir -r requirements.txt

# Start the container
CMD ["python3", "-u", "rp_handler.py"]