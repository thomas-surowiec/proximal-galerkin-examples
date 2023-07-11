# proximal-galerkin-examples
Code for the paper Proximal Galerkin: A structure-preserving finite element method for pointwise bound constraints by Brendan Keith and Thomas M. Surowiec
Github repository set up by Jørgen S. Dokken

## Installation
Either
- Use Anaconda and the conda [environment file](environment.yml)
- Use the docker image from: [ghcr.io/thomas-surowiec/proximal-galerkin-methods](https://github.com/thomas-surowiec/proximal-galerkin-examples/pkgs/container/proximal-galerkin-examples)
- Build the docker image locally with `docker build -t proximal -f docker/Dockerfile .` and run it with for instance `docker run -ti -v ${PWD}:/root/shared -w /root/shared --entrypoint=/bin/bash --rm proximal`

## Running the examples
FEniCSx:
Examples 1-3 can be executed with FEniCSx running `python3 code/obstacle.py --example=x` where `x` stands for the example number. For more options call
`python3 code/obstacle.py --help`
