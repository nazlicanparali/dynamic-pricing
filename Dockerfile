# Serves the API on top of the committed pipeline outputs.
#   docker build -t dynamic-pricing .
#   docker run -p 8000:8000 dynamic-pricing
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY pricing/ pricing/
COPY api/ api/
COPY outputs/ outputs/

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
