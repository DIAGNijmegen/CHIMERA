# =====================================================================================
# Dockerfile for CHIMERA Task 2 Classification Prediction Algorithm
# =====================================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install ASAP
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN curl -L ${ASAP_URL} -o /tmp/ASAP.deb && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/ASAP.deb && \
    SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    echo "/opt/ASAP/bin" > "${SITE_PACKAGES}/asap.pth" && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/ASAP.deb

# Set up application environment
RUN mkdir -p /input /output /opt/app && \
    chown -R user:user /input /output /opt/app

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

# Install Python requirements
COPY --chown=user:user task2_baseline/requirements.txt /opt/app/
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --requirement /opt/app/requirements.txt \
    && rm -rf /home/user/.cache/pip

# Copy the full repository
COPY --chown=user:user . /opt/app/
ENV PYTHONPATH=/opt/app

# Set entrypoint to Task 2 inference
CMD ["python", "task2_baseline/inference.py"]
