# Checkpoint: RunPod Serverless ComfyUI - Clean Build
# Base image: NVIDIA CUDA 11.8 Runtime on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Build arguments
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.4.0
ARG COMFYUI_VERSION=master

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHON_VERSION=${PYTHON_VERSION} \
    PYTORCH_VERSION=${PYTORCH_VERSION} \
    XPU_TARGET=NVIDIA_GPU \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch
RUN python3 -m pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install RunPod SDK and Requests (For Serverless Handler)
RUN python3 -m pip install runpod requests

# Setup Workspace
WORKDIR /workspace

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI && \
    cd /workspace/ComfyUI && \
    git checkout ${COMFYUI_VERSION} && \
    python3 -m pip install -r requirements.txt

# Create directories
RUN mkdir -p /workspace/ComfyUI/models/checkpoints \
             /workspace/ComfyUI/models/clip \
             /workspace/ComfyUI/models/vae \
             /workspace/ComfyUI/models/diffusion_models \
             /workspace/ComfyUI/custom_nodes

# Install Custom Nodes
# 1. ComfyUI Manager (Always good to have)
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /workspace/ComfyUI/custom_nodes/ComfyUI-Manager

# 2. ComfyUI Essentials (Required for ResizeImagesByLongerEdge)
RUN git clone https://github.com/cubiq/ComfyUI_Essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_Essentials && \
    cd /workspace/ComfyUI/custom_nodes/ComfyUI_Essentials && \
    python3 -m pip install -r requirements.txt

# 3. qweneditutils (Required for workflow)
# Installing via git if URL is known, or trusting manager if runtime install needed.
# Since user mentioned "comfy node install qweneditutils", we'll attempt a git clone if we find the URL or skip it for runtime.
# Searching shows it might be 'StartledAnt/ComfyUI_Qwen_Edit_Utils' or similar. 
# Best guessing URL based on name:
RUN git clone https://github.com/StartledAnt/ComfyUI_Qwen_Edit_Utils.git /workspace/ComfyUI/custom_nodes/ComfyUI_Qwen_Edit_Utils || \
    echo "Could not clone Qwen Utils - skipping build time install"

# Download Z-Image Turbo Models
# We use absolute paths to be safe
RUN wget -O /workspace/ComfyUI/models/checkpoints/Qwen-Rapid-AIO-NSFW-v21.safetensors https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v21/Qwen-Rapid-AIO-NSFW-v21.safetensors
RUN wget -O /workspace/ComfyUI/models/clip/qwen_3_4b.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
RUN wget -O /workspace/ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors
RUN wget -O /workspace/ComfyUI/models/vae/ae.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors

# Copy Serverless Handler
COPY rp_handler.py /workspace/rp_handler.py

# Set Entrypoint
WORKDIR /workspace
CMD ["python3", "-u", "/workspace/rp_handler.py"]
