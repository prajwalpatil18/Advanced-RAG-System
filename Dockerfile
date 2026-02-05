FROM python:3.11-slim

# Set base working directory
WORKDIR /app

# Install dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# 5. Expose port
EXPOSE 8000

# Run the app
CMD ["python", "main.py"]
