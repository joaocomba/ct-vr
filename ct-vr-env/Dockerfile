FROM vrex141/qt5.12_cpp17
# MAINTAINER frostasm <frostasm@gmail.com>

USER root
ENV HOME /home/user

RUN apt-get update && \
	apt-get install -y wget && \
	apt-get install -y unzip && \
	apt-get install -y libssl-dev && \
	apt-get install -y ffmpeg && \
	apt-get install -y python3-pip && \
	pip3 install -U pip && \
	pip install dicom2nifti && \
	pip install pandas

#CMAKE
ARG CMAKE_VERSION=3.17
ARG CMAKE_BUILD=1
ENV DEBIAN_FRONTEND=noninteractive
RUN cd /opt && \
	wget https://cmake.org/files/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.$CMAKE_BUILD.tar.gz && \
	tar -xzvf cmake-$CMAKE_VERSION.$CMAKE_BUILD.tar.gz && \
	cd cmake-$CMAKE_VERSION.$CMAKE_BUILD/ && \ 
	./bootstrap && \ 
	make -j$(nproc) && \ 
	make install

#MITK
RUN git clone https://github.com/MITK/MITK.git /opt/MITK && cd /opt/MITK && git checkout c0f884b289 && \ 
	mkdir /opt/MITK-build && \ 
	cd /opt/MITK-build && \
	cmake ../MITK && \
	make -j1
	# make -j$(nproc)

RUN pip install simpleitk
RUN pip install pyyaml

ENV MITK_BUILD_PATH=/opt/MITK-build
ENV QT_ROOT_PATH=/opt/Qt/5.12.4
ENV DEBIAN_FRONTEND=noninteractive
#COPY TEMPORAL FILES PHNN, MODELS, IMAGES -> MAKE TO SHARED DIRECTORY
RUN git clone https://github.com/ct-vr/video-generator.git /opt/video-generator && \
	git clone https://github.com/ct-vr/camera-shots-generator.git /opt/camera-shots-generator && \
	git clone https://github.com/noamlr/vglrun-support /opt/vglrun-support && \	
	cd /opt/camera-shots-generator && cmake -B build -S . && cd build && make && \
	cd /opt/video-generator && cmake -B build -S . && cd build && make 
COPY p-hnn-lung-segmentation.zip /opt/
RUN cd /opt && unzip p-hnn-lung-segmentation.zip && rm p-hnn-lung-segmentation.zip
COPY iftInterp /bin/ 
COPY pipeline/ /opt/pipeline/

# VirtualGL
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y lightdm && \
	systemctl enable lightdm && \
	cd /opt/vglrun-support && \
	dpkg -i virtualgl_*.deb && \
	dpkg -i turbovnc_*.deb && \
	sed -i 's/$host:/unix:/g' /opt/TurboVNC/bin/vncserver

ENV PATH ${PATH}:/opt/VirtualGL/bin:/opt/TurboVNC/bin
ENV DISPLAY :0

#CONDA
ARG ANACONDA_VERSION=2020.11
RUN wget https://repo.continuum.io/archive/Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh -O anaconda.sh && \
	chmod +x anaconda.sh && \
	./anaconda.sh -b -p /opt/conda && \
	rm anaconda.sh && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	ln -s /opt/conda/bin/conda /usr/bin/conda

ENV PATH /opt/conda/envs/env/bin:$PATH
COPY script.sh /opt/script.sh
SHELL ["/bin/bash", "-c"] 
COPY ct-vr_env.sh /opt/ct-vr_env.sh
RUN conda init bash && \
	chmod +x /opt/script.sh  && \
	/opt/script.sh && \
	conda create -n ct-vr python=3.6 && \
	echo "conda activate ct-vr" > ~/.bashrc && \
	echo "conda activate ct-vr" > /root/.bashrc && \
	chmod +x /opt/ct-vr_env.sh  && \ 
	/opt/ct-vr_env.sh


# Reconfigure locale
RUN locale-gen en_US.UTF-8 && dpkg-reconfigure locales

WORKDIR /opt/pipeline
