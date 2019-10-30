# Developing with Docker

## Build Docker image for GPU machine.


Run from root UniRep directory

    $ docker build -f docker/Dockerfile.gpu -t unirep-gpu .

## Build CPU-only Docker image

    $ docker build -f docker/Dockerfile.cpu -t unirep-cpu .

# Start Docker machine
From repository root, run `docker/run_cpu_docker.sh`

# On GPU
`docker/run_gpu_docker.sh`

# On GPU, with docker version < 19.03
`docker/run_gpu_docker_DEPRECATED.sh`
