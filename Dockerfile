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

# Environment variables passed at runtime
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

CMD ["python", "inference.py"]
