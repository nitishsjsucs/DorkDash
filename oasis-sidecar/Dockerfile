FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir camel-oasis && \
    pip install --no-cache-dir --force-reinstall "fastapi>=0.110" "uvicorn>=0.29,<1.0" aiohttp

COPY server.py .
COPY dasher_profiles.json .

EXPOSE 5050

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5050"]
