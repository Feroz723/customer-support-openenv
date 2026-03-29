FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL project files
COPY . /app

# HF Space requires port 7860
EXPOSE 7860

# Run the FastAPI server — must stay alive
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
