FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Render assigns a dynamic port via the PORT environment variable.
# We fall back to 10000 if PORT is not set.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}
