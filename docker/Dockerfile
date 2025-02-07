FROM condaforge/mambaforge

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# Install ssh (missing dependency to run conda envs)
RUN apt-get update && \
    apt-get install -y ssh build-essential

# Upgrade mamba
RUN mamba upgrade -y mamba

# Copy environment and requirements files into docker env
COPY environment.yml .

# Update environment file with new environment name
RUN mamba env create --file environment.yml
SHELL ["mamba", "run", "-n", "proximalGalerkin", "/bin/bash", "-c"]

RUN echo "source activate proximalGalerkin" > ~/.bashrc
