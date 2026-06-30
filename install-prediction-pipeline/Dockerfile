# Slim serving image: ONNX Runtime only (no LightGBM/sklearn needed at inference
# time because the whole preprocessing+model graph is baked into model.onnx).
FROM python:3.11-slim

WORKDIR /app

# Minimal runtime deps for serving — keeps the image small and cold-start fast.
RUN pip install --no-cache-dir \
        onnxruntime>=1.17 \
        fastapi>=0.110 \
        "uvicorn[standard]>=0.27" \
        pydantic>=2.0 \
        pandas>=2.0 \
        numpy>=1.24

COPY src/ ./src/
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/app/src
ENV IPM_ARTIFACTS=/app/artifacts

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if 'ok' in urllib.request.urlopen('http://localhost:8080/health').read().decode() else 1)"

CMD ["uvicorn", "ipm.serve:app", "--host", "0.0.0.0", "--port", "8080"]
