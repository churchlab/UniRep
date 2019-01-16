# Developing with Docker

## Build Docker image for GPU machine.


Run from root mlpe-gfp-pilot directory

    $ docker build -f docker/Dockerfile.gpu -t unirep-gpu .

## Build CPU-only Docker image

    $ docker build -f docker/Dockerfile.cpu -t unirep-cpu .

# Start Docker machine
From repository root, run `docker/run_cpu_docker.sh`
