# Docker file for det3
# Author: Peng YUN (pyun@cse.ust.hk)
# Copyright Peng YUN, RAM-Lab, HKUST
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# install mini conda 4.7.12.1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
# install pytorch 1.4
RUN conda install pytorch==1.4 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
RUN conda install -c open3d-admin open3d==0.10.0
RUN conda install -c numba numba
RUN conda install -c conda-forge tensorboardX==2.0 easydict==1.9 tqdm numpy
RUN apt-get update && apt-get -y install libgl1-mesa-glx

RUN conda install -c conda-forge scikit-image shapely

RUN apt-get update && apt-get install -y libboost1.62-dev
COPY kitti-object-eval-python /root/kitti-object-eval-python
COPY second.pytorch /root/second.pytorch
ENV PYTHONPATH=/root/second.pytorch/:/root/kitti-object-eval-python:/usr/app/

COPY spconv /root/spconv
COPY cmake /root/cmake
RUN cd /root/cmake && sh *.sh --prefix=/root/cmake --skip-license
RUN ln -s /root/cmake/bin/* /usr/bin/
RUN cmake --version
CMD cd /root/spconv/ && python setup.py bdist_wheel \
    && cd /root/spconv/dist && pip install *whl && cd / && /bin/bash
