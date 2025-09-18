# Base image with CUDA 12.8 + cuDNN
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="Amirhosein <amirhosein.najafy@gmail.com>"
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget curl build-essential \
    libgl1-mesa-glx libglib2.0-0 unzip ninja-build cmake && \
    rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# CUDA environment variables - use symlinked path for better compatibility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
#Adjust for your target GPU arch (A100=8.0, RTX40xx=8.9, H100=9.0)
ENV TORCH_CUDA_ARCH_LIST="9.0"  
# Set workspace
WORKDIR /workspace/resana

# Copy project requirements
COPY requirements-pip.txt ./

# Install PyTorch + torchvision + torchaudio with CUDA
RUN pip install --upgrade pip && \
    pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r requirements-pip.txt

# Clone DMD project
ARG REPO_URL=https://github.com/6amir6hosein6/DMD.git
RUN git clone -b master --single-branch $REPO_URL


# Build torch-linear-assignment with forced CUDA support
WORKDIR /workspace/resana
RUN git clone https://github.com/ivan-chai/torch-linear-assignment.git
WORKDIR /workspace/resana/torch-linear-assignment

# ✅ Patch setup.py to respect FORCE_CUDA
RUN sed -i "s/torch\.cuda\.is_available()/torch.cuda.is_available() or os.getenv('FORCE_CUDA') == '1'/g" setup.py

# Optional: Verify patch (non-fatal)
RUN grep -n "FORCE_CUDA" setup.py || echo "Patch applied (or no match)"

ENV FORCE_CUDA=1

# Install with verbose output to see if CUDA kernels are being compiled
RUN pip install -v .


WORKDIR /workspace/resana/DMD

# Clone fptools repo
RUN git clone https://github.com/youngjetduan/fptools.git

# Download pretrained model
RUN mkdir -p ./logs/DMD && \
    wget --no-check-certificate "https://cloud.tsinghua.edu.cn/f/fd5ca22af0eb44afa124/?dl=1" -O best_model.pth.tar && \
    mv best_model.pth.tar ./logs/DMD/best_model.pth.tar

# Prepare test data
RUN mkdir ./TEST_DATA && \
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1CrfRsF6nRqOX5vpa4_DL-UKvNgwJVWRi" -O SDTESTile.zip && \
    unzip myfile.zip "SDTEST/*" -d ./TEST_DATA

# Run dataset preprocessing
RUN python dump_dataset_mnteval.py

# Set default command
CMD ["bash"]