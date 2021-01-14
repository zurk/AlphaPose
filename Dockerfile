ARG CUDA_VERSION=10.1
# It looks like for pytorch3d we need to use CUDA_VERSION=10.1
ARG CUDNN_VERSION=7
# We use devel and not runtime because detectron2 needs to be compiled
# It requiers nvcc at least. Can be changed in future.
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

FROM ${BASE_IMAGE}

ENV LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
        gcc g++ cmake git wget ca-certificates locales make \
        python3.7 python3.7-dev python3.7-distutils && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3.7 && \
    pip3 install --no-cache-dir Cython torch==1.4.0 torchvision==0.5.0

RUN apt-get install -y --no-install-suggests --no-install-recommends libyaml-dev

COPY . /home/AlphaPose

WORKDIR /home/AlphaPose

RUN pip3 install git+https://github.com/yanfengliu/cython_bbox.git
RUN pip3 install git+https://github.com/cocodataset/cocoapi.git@8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9#subdirectory=PythonAPI
RUN python3.7 setup.py build develop --user

RUN apt-get install -y --no-install-suggests --no-install-recommends ffmpeg libsm6 libxext6
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-suggests --no-install-recommends python3.7-tk

