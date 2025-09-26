#!/bin/bash
set -o errexit

py_ver="$1"
location="$2"
FILENAME=Miniconda3-py${py_ver//.}_23.5.2-0-Linux-aarch64.sh

if [ "$1" = "3.7" ]
then
FILENAME=Miniconda3-py${py_ver//.}_23.1.0-1-Linux-aarch64.sh
fi

# 设置清华镜像（兼容 ARM64）
if [ "$location" = "us" ]; then
    MIRROR="https://repo.anaconda.com/miniconda"
else
    MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda"
fi

# 下载并安装
wget ${MIRROR}/${FILENAME}
mkdir -p /root/.conda
bash ${FILENAME} -b
rm -f ${FILENAME}

# 初始化 conda（可选）
# /root/miniconda3/bin/conda init bash