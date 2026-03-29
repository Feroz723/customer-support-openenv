FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Run the FastAPI server at server/app.py (port 7860 for HF)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
