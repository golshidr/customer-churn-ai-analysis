# Scoring API image. Build after `make benchmark decide export` has written
# models/churn_model.joblib:
#   docker build -t churn-api .
#   docker run -p 8000:8000 churn-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_MODEL_PATH=/app/models/churn_model.joblib

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY models/churn_model.joblib ./models/churn_model.joblib

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "churn_analysis.api:app", "--host", "0.0.0.0", "--port", "8000"]
