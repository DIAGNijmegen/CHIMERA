# =====================================================================================
# Dockerfile for CHIMERA Task 2 Classification Prediction Algorithm
# =====================================================================================
#
# This Dockerfile is adapted from the CHIMERA Task 1 baseline but modified for Task 2.
# It builds upon the provided base image with CUDA, CUDNN, and ASAP.
# It installs Python 3.10 for reproducibility.
#
# Assumes:
#   - This Dockerfile is located in the root of the 'task2_baseline' project folder.
#   - A 'requirements.txt' file exists in the same root folder.
#   - The entry script is at 'task2_baseline/scripts/test_FE.py'.
#
# =====================================================================================

# --- 1. Base Image and User Setup ---
ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

# --- 2. Environment and System Dependencies ---
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff-dev \
    zlib1g-dev \
    curl \
    wget \
    git \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install ASAP
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN curl -L ${ASAP_URL} -o /tmp/ASAP.deb && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/ASAP.deb && \
    SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    echo "/opt/ASAP/bin" > "${SITE_PACKAGES}/asap.pth" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/ASAP.deb

# --- 3. Application Setup ---
RUN mkdir -p /input /output /opt/app && \
    chown -R user:user /input /output /opt/app

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

# --- 4. Install Python dependencies ---
COPY --chown=user:user task2_baseline/requirements.txt /opt/app/
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --requirement /opt/app/requirements.txt \
    && rm -rf /home/user/.cache/pip

# --- 5. Copy Project Code ---
COPY --chown=user:user . /opt/app/
ENV PYTHONPATH=/opt/app

# --- 6. Execution ---
CMD ["python", "task2_baseline/prediction_model/Aggregators/inference/inference.py"]



