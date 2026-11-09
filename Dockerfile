FROM python:3.12-slim

# Run as a normal user, not root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# CPU-only PyTorch: the image is much smaller than the default (CUDA) build.
# Same version as training, so the checkpoint format matches.
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# Other dependencies (copied first, so Docker caches this layer when only code changes)
COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Code, tokenizer and exported model. Same folder layout as the project,
# because the code builds paths relative to the project root.
COPY --chown=user src/ src/
COPY --chown=user inference/ inference/
COPY --chown=user app/ app/
COPY --chown=user tokenizer/ tokenizer/
COPY --chown=user models/ models/

# Railway sets PORT; 8000 is the default for local runs.
# "exec" makes uvicorn the main process, so it receives the stop signal.
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
