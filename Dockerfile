FROM python:3.12-slim
WORKDIR /app
COPY requirements-runtime-lock.txt .
RUN pip install --no-cache-dir -r requirements-runtime-lock.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
