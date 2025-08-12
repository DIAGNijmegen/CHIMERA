# =====================================================================================
# Dockerfile — CHIMERA Task 2 Inference (Dockerfile in task2_baseline/, context = CHIMERA/)
# =====================================================================================

ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# ---- user ----
ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} user \
 && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

# ---- system deps + Python 3.10 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff-dev zlib1g-dev curl wget git ca-certificates \
    python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# ---- ASAP (optional) ----
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN curl -L ${ASAP_URL} -o /tmp/ASAP.deb \
 && apt-get update && apt-get install -y --no-install-recommends /tmp/ASAP.deb \
 && python -c "import sysconfig, pathlib; pathlib.Path(sysconfig.get_paths()['purelib']).joinpath('asap.pth').write_text('/opt/ASAP/bin\n')" \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/ASAP.deb

# ---- dirs ----
RUN mkdir -p /input /output /opt/app /opt/ml/model \
 && chown -R user:user /input /output /opt/app /opt/ml/model

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH=/opt/app
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# ---- Python deps ----
COPY --chown=user:user task2_baseline/requirements.txt /opt/app/
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r /opt/app/requirements.txt \
 && rm -rf /home/user/.cache/pip

# ---- copy code (both common/ and task2_baseline/) ----
COPY --chown=user:user . /opt/app/

# ---- copy trained model + clinical processor into fixed path used by code ----
COPY --chown=user:user common/model/ABMIL_task2/s_checkpoint.pth       /opt/ml/model/s_checkpoint.pth
COPY --chown=user:user common/model/ABMIL_task2/clinical_processor.pkl /opt/ml/model/clinical_processor.pkl
COPY --chown=user:user common/model/ABMIL_task2/config.json            /opt/ml/model/config.json

# quick sanity
RUN ls -lah /opt/ml/model

# ---- default command ----
CMD ["python", "task2_baseline/inference.py"]
