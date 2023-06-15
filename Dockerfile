# Start with the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Copy the YAML file containing additional packages
COPY environment.yml .


RUN mkdir -m a=rw ./data

# Set the noninteractive environment variable
ENV DEBIAN_FRONTEND=noninteractive

# install tkinter
RUN apt-get update && apt-get install -y \
    tk \
    libx11-dev

# allow screen sharing
RUN apt-get update && apt-get install -y xorg
RUN apt-get update && apt-get install -y xfonts-base


# jupyter
RUN pip install jupyter

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini

RUN chmod +x /usr/bin/tini


RUN pip install --upgrade jupyter_http_over_ws>=0.0.7 && jupyter serverextension enable --py jupyter_http_over_ws

ENV MLFLOW_TRACKING_URI=file:///workspace/mlflow
ENV TENSORBOARD_LOG_DIR=/workspace/tensorboard


EXPOSE 8888
EXPOSE 5000
EXPOSE 6006

# Create a Conda environment named "EM+" and clone from the base environment
RUN conda create --name EM+ --clone base

# Activate the "EM+" environment
SHELL ["conda", "run", "-n", "EM+", "/bin/bash", "-c"]

# Update the "EM+" environment with packages from environment.yml
RUN conda env update --file environment.yml --prune


RUN apt-get update && apt-get install -y supervisor

# Set the default command to launch Jupyter Notebook
ENTRYPOINT ["/usr/bin/tini", "--"]

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord"]

# # Create and activate a Conda environment
# RUN conda env create --file environment.yml && \
#     echo "source activate myenv" >> ~/.bashrc