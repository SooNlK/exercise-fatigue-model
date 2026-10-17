FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/api.txt

COPY src/ src/
COPY artifacts/model.joblib artifacts/model.joblib

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
