FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel

ARG MMYOLO_COMMIT=8c4d9dc503dc8e327bec8147e8dc97124052f693
ARG EFFDET_COMMIT=c6dff775a36cea0bf9b76c58e59f936411c5ce01

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.9"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ninja-build libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
      numpy==1.26.4 mmengine==0.10.7 \
      opencv-python-headless==4.10.0.84 Pillow==10.4.0 \
      pycocotools==2.0.7 scipy==1.11.4 PyYAML==6.0.2 \
      timm==0.9.16 omegaconf==2.3.0 thop==0.1.1.post2209072238 \
    && python -m pip install --no-cache-dir \
      mmcv==2.0.1 \
      -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
    && python -m pip install --no-cache-dir mmdet==3.3.0

RUN git clone --filter=blob:none https://github.com/open-mmlab/mmyolo.git /opt/mmyolo \
    && git -C /opt/mmyolo checkout --detach "${MMYOLO_COMMIT}" \
    && python -m pip install --no-cache-dir --no-deps -e /opt/mmyolo

COPY configs/patches/efficientdet-gradient-accumulation.patch /tmp/efficientdet-gradient-accumulation.patch
RUN git clone --filter=blob:none https://github.com/rwightman/efficientdet-pytorch.git /opt/efficientdet \
    && git -C /opt/efficientdet checkout --detach "${EFFDET_COMMIT}" \
    && git -C /opt/efficientdet apply --check /tmp/efficientdet-gradient-accumulation.patch \
    && git -C /opt/efficientdet apply /tmp/efficientdet-gradient-accumulation.patch \
    && python -m pip install --no-cache-dir --no-deps -e /opt/efficientdet

WORKDIR /workspace
ENV PYTHONPATH=/workspace:/opt/mmyolo:/opt/efficientdet
ENTRYPOINT ["python", "-m", "tools.backends.container_entry"]
