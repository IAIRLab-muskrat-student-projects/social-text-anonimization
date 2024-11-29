FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y git apt-utils wget supervisor nano

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.04-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN /opt/conda/bin/conda update -n base -c defaults conda
RUN /opt/conda/bin/conda install -c conda-forge jupyterlab
# RUN /opt/conda/bin/conda install -c pytorch faiss-gpu 

RUN /opt/conda/bin/conda install pytorch==1.12 torchvision==0.13 cudatoolkit=10.2 -c pytorch
RUN /opt/conda/bin/conda install -c conda-forge "scikit-learn>=0.24"

RUN /opt/conda/bin/pip install --upgrade pip
RUN /opt/conda/bin/pip install "tensorflow==2.12.0"
RUN /opt/conda/bin/pip install "transformers==4.29.2"

RUN /opt/conda/bin/pip install setuptools wheel
RUN /opt/conda/bin/pip install --default-timeout=240 spacy[cuda102]

RUN /opt/conda/bin/pip install "bertopic==0.15.0"

RUN /opt/conda/bin/conda install ipykernel

RUN /opt/conda/bin/conda install -c plotly plotly=5.6.0
RUN /opt/conda/bin/conda install -c conda-forge jupyterlab_widgets
RUN /opt/conda/bin/conda install -c conda-forge "ipywidgets>=7.5"
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get -y install nodejs

RUN /opt/conda/bin/jupyter labextension install jupyterlab-plotly@5.6.0

RUN apt-get update && apt-get install -y sudo
RUN mkdir -p /mnt/ess_storage/DN_1/
RUN mkdir /home/jovyan && chmod 777 /home/jovyan 
RUN mkdir /home/jovyan/notebooks && chmod 777 /home/jovyan/notebooks

# RUN apt-get update && apt-get install -y libxml2 libxslt-dev bzip2 gcc

#RUN /opt/conda/bin/conda conda create -n rapids-22.02 -c rapidsai -c nvidia -c conda-forge \
#                             rapids=22.02 python=3.9 cudatoolkit=11.2 dask-sql
#RUN /opt/conda/bin/conda install -c plotly plotly=4.14.3
#RUN /opt/conda/bin/conda install -n rapids-0.18 -c plotly plotly=4.14.3

#RUN /opt/conda/bin/pip install natasha
RUN /opt/conda/bin/conda install -c conda-forge "gensim==3.8.3"
RUN /opt/conda/bin/pip install "pytorch-lightning==2.0.2"
RUN /opt/conda/bin/pip install "clean-text==0.6.0" "fasttext==0.9.2"
# RUN /opt/conda/bin/pip install keybert
# RUN python -m spacy download ru_core_news_sm
# RUN /opt/conda/bin/pip install pymystem3
# RUN /opt/conda/bin/pip install torch-lr-finder
# RUN python -m spacy download en_core_web_lg
RUN /opt/conda/bin/pip install "geopandas==0.13.0"
