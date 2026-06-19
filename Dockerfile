# Reproducible environment for the committed numbers (the AUROC=0.716 SQuAD gate).
# Pins the OS (Debian/python:3.10-slim), the exact deps (requirements.lock), CPU-only
# torch, and single-threaded BLAS — removing the OS/threading/BLAS nondeterminism the
# lockfile alone cannot. Large tensors are NOT baked in; fetch them at run time.
#
#   docker build -t llm-sae-difficulty .
#   docker run --rm llm-sae-difficulty make test          # fast tests + leakage audit
#   docker run --rm llm-sae-difficulty bash -lc \
#       "bash download_artifacts.sh && make squad"        # full SQuAD reproduction
FROM python:3.10-slim

# Single-threaded BLAS/OpenMP => deterministic linear algebra across hosts.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git make curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pinned deps first (better layer caching), CPU torch wheels.
COPY requirements.lock pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install -e . --no-deps

CMD ["bash", "-lc", "make help"]
