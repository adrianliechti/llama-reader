FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install chromium --with-deps

COPY main.py .

EXPOSE 8000
VOLUME /app/.cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]