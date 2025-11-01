FROM mcr.microsoft.com/playwright/python:v1.53.0-noble

COPY . /app
WORKDIR /app

# Upgrade pip and install CPU-only PyTorch
RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Default command will be overridden by docker-compose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6789"]