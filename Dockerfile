FROM python:3.12-slim

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_VERBOSITY=error
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

RUN python -m playwright install chromium --with-deps

COPY main.py .

EXPOSE 8000
VOLUME /app/.cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]