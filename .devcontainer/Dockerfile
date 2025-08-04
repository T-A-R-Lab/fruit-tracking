FROM ubuntu:20.04
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ARG USERNAME=USERNAME
ARG USER_UID=USER_UID
ARG USER_GID=USER_GID
ARG IMAGE_NAME=IMAGE_NAME

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    curl \
    git \
    nano \
    sudo \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-distutils python3.9-venv wget git && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# Install ffmpeg with timezone America/Santiago
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Santiago /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y ffmpeg
RUN apt-get install -y python3.9-tk


# Copy requirements file
COPY requirements.txt /requirements.txt

# Install tensorflow 2.5.0
ENV PATH=$PATH:/home/${USERNAME}/.local/bin
RUN pip install --upgrade pip && \
    pip install --upgrade --no-cache-dir -r requirements.txt

# -- Instalation of Object Detection API dependencies --
# Install protobuf-compiler
RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# Clone TensorFlow models repository just if not already present
RUN if [ ! -d "/ws/TensorFlow" ]; then \
    git clone "https://github.com/tensorflow/models" /ws/TensorFlow; \
    fi

RUN cd /ws/TensorFlow/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    pip install . 

# -- Instalation of Stone-Soup from source--
RUN if [ ! -d "/ws/Stone-Soup" ]; then \
    git clone "https://github.com/dstl/Stone-Soup.git" /ws/Stone-Soup; \
    fi

RUN cd /ws/Stone-Soup && \
    python3 -m pip install -e .[dev,video,tensorflow] --no-deps && \
    python3 -m pip install stonesoup[video]


RUN echo "Building..."
RUN echo "+ USERNAME=${USERNAME}" 
RUN echo "+ USER_UID=${USER_UID}"
RUN echo "+ USER_GID=${USER_GID}"

# # # # # # # # # # # # # # # # # # # #
# Create the user with same GID and UID as the host:
# # # # # # # # # # #  # # # # # # # # 
# Change the primary group of any user using the group to be deleted
RUN if getent group ${USER_GID}; then \
    for user in $(getent passwd | awk -F: -v gid=${USER_GID} '$4 == gid {print $1}'); do \
    usermod -g users $user; \
    done; \
    fi

# Delete existing user if it exists
RUN if getent passwd ${USER_UID}; then \
    userdel -r $(getent passwd ${USER_UID} | cut -d: -f1); \
    fi

# Delete existing group if it exists
RUN if getent group ${USER_GID}; then \
    groupdel $(getent group ${USER_GID} | cut -d: -f1); \
    fi

# Create the group with the specified GID
RUN groupadd -g ${USER_GID} ${USERNAME}

# Create the user with the specified UID and add to the sudo group
RUN useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# # # # # # # # # # # # # # # # # # # # # 

USER ${USERNAME}
WORKDIR /ws
CMD ["/bin/bash"]
