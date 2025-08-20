# hash:sha256:6a77661721e5a7b3e9e31c6da60c42b5a4d19207f0a7c8d2c929781eaf0c5a5c
FROM registry.codeocean.com/codeocean/py-r:python3.12.8-R4.4.2-JupyterLab4.3.5-RStudiorstudio-server-2024.12.1-563-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    eigenshuffle==0.1.0 \
    matplotlib==3.10.1 \
    numpy==1.26.4 \
    scipy==1.15.2
