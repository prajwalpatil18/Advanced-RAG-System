FROM python:3.11-slim

# Set base working directory
WORKDIR /

# Install dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Switch to main folder
WORKDIR /

# 5. Expose port
EXPOSE 8000

# Run the app
CMD ["python", "main.py"]
