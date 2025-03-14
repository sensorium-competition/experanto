FROM nvcr.io/nvidia/pytorch:25.02-py3
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install jupyterlab
RUN pip3 install torch torchvision torchaudio


EXPOSE 8888
# By default start running jupyter notebook
WORKDIR /notebooks
#RUN pip3 install -e ./mousehiera
#RUN git clone -b statshack https://github.com/KonstantinWilleke/experanto.git
#RUN pip3 install -e /src/experanto
ENTRYPOINT ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--port=8888", "--NotebookApp.token='1234'", "--notebook-dir='/notebooks'"]