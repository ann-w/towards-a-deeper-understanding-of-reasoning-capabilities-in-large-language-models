FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG CONDA_ENV=SmartPlay
ARG CONDA_VERSION="py39_4.12.0"

# Install system dependencies including git
RUN apt-get update -y && apt-get install -y \
    wget \
    bzip2 \
    software-properties-common \
    curl \
    git \
    pciutils \
    lshw \
    && rm -rf /var/lib/apt/lists/*

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$CONDA_VERSION-Linux-x86_64.sh -O miniconda.sh \
    && /bin/bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:${PATH}"

# Install Java 8 for Minedojo
RUN add-apt-repository ppa:openjdk-r/ppa -y \
    && apt-get update -y && apt-get install -y openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Create the SmartPlay conda environment
RUN conda env create --name $CONDA_ENV --file libs/smartplay/environment.yml \
    && conda clean -afy

# Ensure environment's Python and pip are used
ENV CONDA_DEFAULT_ENV=$CONDA_ENV
ENV PATH="/opt/conda/envs/${CONDA_ENV}/bin:${PATH}"
ENV PYTHONPATH="/app"


# Upgrade pip and install SmartPlay in editable mode from libs/smartplay
WORKDIR /app/libs/smartplay
RUN python -m pip install --upgrade pip==23.* && python -m pip install -e .

# Return to /app and install remaining dependencies
WORKDIR /app
RUN python -m pip install -r requirements.txt

CMD ["bash"]