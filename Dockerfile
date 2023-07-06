# Set base image (host OS)
FROM condaforge/mambaforge
#continuumio/miniconda3
#python:3.8.13-buster
#bitnami/pytorch
#python:3.8.5-slim-buster 

RUN apt-get update && apt-get install libgl1 -y

RUN conda create -n myenv python=3.11
#RUN echo "source activate env" > ~/.bashrc
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
#ENV PATH /opt/conda/envs/env/bin:$PATH
#ENV PYTHONPATH "${PYTHONPATH}:/opt/conda/envs/env/bin"

# By default, listen on port 5000
EXPOSE 80/tcp

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

#RUN conda config --add channels conda-forge
# && \
#    conda config --set channel_priority strict

# Install any dependencies
RUN mamba install Flask==2.3.2 Flask-Cors==3.0.10 \
    geopandas==0.13.2 gunicorn==20.1.0 pytorch-cpu==2.0.0 torchvision==0.15.2 \
    rasterio==1.3.6 requests==2.31.0 Pillow==9.5.0 numpy==1.24.1 \
    albumentations==1.3.1 shapely==2.0.1 geojson==3.0.1 tqdm==4.65.0
#GDAL>=3.1 

# Copy the content of the local src directory to the working directory
COPY ./ .

# Specify the command to run on container start
#CMD [ "python", "./main.py" ]
#CMD /bin/bash -c "source activate env && python3 main.py"
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "main.py"]