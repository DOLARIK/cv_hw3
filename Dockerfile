FROM python:3.6

ENV JUPYTER_TOKEN dgupta12_psood_asg3

# Pick up some general dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        tree \
        curl \
        libfreetype6-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        git \
        tcl-dev \
        tk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install Pillow

# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    numpy matplotlib Cython

# Jupyter Lab
RUN pip install jupyter -U && pip install jupyterlab -U
EXPOSE 8888

# OpenCV 3.4.1    
RUN pip install opencv-python && \ 
    apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /assignment_3

# Production
COPY ./ /assignment_3
RUN pip install -r requirements.txt

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]