FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /workspace

# CPU-only torch build first (used only for NLI in E2/E3)
RUN python3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Native AVX flags for llama-cpp-python's ggml build, compiled for this host's CPU
RUN CMAKE_ARGS="-DGGML_NATIVE=ON" python3.11 -m pip install -r requirements.txt

RUN python3.11 -m spacy download en_core_web_sm

COPY . .

ENTRYPOINT ["python3.11"]
