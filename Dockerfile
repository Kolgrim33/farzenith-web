# FarZenith chat (Render, Cloud Run or anywhere Docker runs). No PyTorch: the model runs in NumPy.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV HF_HOME=/models
# Bake the model into the image so a cold start doesn't download it.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Kolgrim/farzenith-small-v1')"
# One BLAS thread: small matrices, and free hosts give a fraction of a CPU.
ENV HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY app.py .
CMD ["python", "app.py"]
