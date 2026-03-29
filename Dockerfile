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
COPY api.py .

# HF Space requires port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
