FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Bind to 0.0.0.0 and port 7860 as requested by HF Spaces
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
