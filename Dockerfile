# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.5.1-base

# install custom nodes into comfyui (first node with --mode remote to fetch updated cache)
# The workflow included custom nodes under `unknown_registry` but no aux_id (GitHub repo) was provided for any of them,
# so they cannot be installed automatically and were skipped:
# - CheckpointLoaderSimple (unknown_registry) - no aux_id provided, skipped
# - TextEncodeQwenImageEditPlus (unknown_registry) - no aux_id provided, skipped
# - TextEncodeQwenImageEditPlus (unknown_registry) - duplicate, no aux_id provided, skipped
# - ConditioningZeroOut (unknown_registry) - no aux_id provided, skipped
# - ResizeImagesByLongerEdge (unknown_registry) - no aux_id provided, skipped
# Install custom nodes using RunPod's CLI
RUN comfy-node-install qweneditutils
RUN comfy-node-install ComfyUI_Essentials

# Update ComfyUI to latest version (Critical for Z-Image Turbo model support)
# Install git and update repo
RUN apt-get update && apt-get install -y git && \
    git config --global --add safe.directory /comfyui && \
    cd /comfyui && \
    git checkout master && \
    git pull




# download models into comfyui
RUN comfy model download --url https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v21/Qwen-Rapid-AIO-NSFW-v21.safetensors --relative-path models/checkpoints --filename Qwen-Rapid-AIO-NSFW-v21.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors --relative-path models/clip --filename qwen_3_4b.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors --relative-path models/diffusion_models --filename z_image_turbo_bf16.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors --relative-path models/vae --filename ae.safetensors

# copy all input data (like images or videos) into comfyui (uncomment and adjust if needed)
# COPY input/ /comfyui/input/
