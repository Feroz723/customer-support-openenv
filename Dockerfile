FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py .
COPY tasks.py .
COPY grading.py .
COPY environment.py .
COPY inference.py .

# Environment variables should be set in the Hugging Face Space Settings (Variables and Secrets)

CMD ["python", "inference.py"]
