# UniRep, a mLSTM "babbler" deep representation learner for protein engineering informatics.

We present an interface for training, inferencing reprepresentations, generative modelling aka "babbling", and data management. All three architectures (64, 256, and 1900 units) are provided along with the trained architectures, random initializations used for evotuning (to ensure reproducibility) and the evotuned parameters.

## Quick-start

First clone or fork this repository and navigate to the repository's top directory (`cd UniRep`). We recommend developing with docker.

### CPU-only support
0. You will need to install [docker](https://www.docker.com/why-docker) to get started.
1. Build docker: `docker build -f docker/Dockerfile.cpu -t unirep-cpu .` This step pulls the Tensorflow 1.3 CPU image and installs a few required python packages. Note that Tensorflow pulls from Ubuntu 16.04.
2. Run docker: `docker/run_cpu_docker.sh`. This will launch Jupyter. Copy and paste the provided URL into your browser. Note that if you are running this code on a remote machine you will need to set up port forwarding between your local machine and your remote machine. See this [example](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh) (note that in our case jupyter is serving port 8888, not 8889 as the example describes).
3. Open up the `unirep_tutorial_64_unit.ipynb` or `unirep_tutorial.ipynb` notebooks and get started. The 64-unit model should be OK to run on any machine. The full-sized model used in `unirep_tutorial.ipynb` will require a machine with more than 16GB of RAM.

### GPU support
0. System requirements: NVIDIA CUDA 8.0 (V8.0.61), NVIDIA cuDNN 6.0.21, NVIDIA GPU Driver 410.79 (though == 361.93 or >= 375.51 should work. Untested), nvidia-docker. We use the AWS [Deep Learning Base AMI for Ubuntu](https://aws.amazon.com/marketplace/pp/B077GCZ4GR), which has these requirements pre-configured. 
1. Build docker: `docker build -f docker/Dockerfile.gpu -t unirep-gpu .` This step pulls the Tensorflow 1.3 GPU image and installs a few required python packages. Note that Tensorflow pulls from Ubuntu 16.04.
2. Run docker: `docker/run_gpu_docker.sh`. This will launch Jupyter. Copy and paste the provided URL into your browser. Note that if you are running this code on a remote machine you will need to set up port forwarding between your local machine and your remote machine. See this [example](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh) (note that in our case jupyter is serving port 8888, not 8889 as the example describes).
3. Open up the `unirep_tutorial_64_unit.ipynb` or `unirep_tutorial.ipynb` notebooks and get started. The 64-unit model should be OK to run on any machine. The full-sized model used in `unirep_tutorial.ipynb` will require a machine with more than 16GB of GPU RAM.

- unirep_tutorial_64_unit.ipynb Start here for information about installation and usage of the babbler interface. unirep_tutorial.ipynb uses the full-sized model which should only be run on a workstation with a more than 16G of RAM. 
- unirep.py Interface for most use cases.
- custom_models.py Custom implementations of GRU, LSTM and mLSTM cells as used in representation training on UniRef50
- data_utils.py Convenience functions for data management.
- pbab3.yml, pip_requirements.txt, requirements.txt Environment files.
- formatted and seqs.txt Tutorial files.
- {64,256,1900}_weights Parameters trained on UniRef50. See tutorial for usage.
- {64,256,1900}_weights_random Random initializations (used for random initialization for evotuning where applicable) for reproducibility. See tutorial for usage.
- evotuned/{unirep,random_init} The weights, as a tensorflow checkpoint file, after 13k unsupervised weight updates on fluorescent protein homologs obtained with JackHMMer of pre-trained UniRep and a random initialization NOT pretrained. Can be loaded into the interface with the tf.Saver module (see https://github.com/tensorflow/docs/tree/r1.3/site/en/api_docs for usage).
- 64_predict_42_weights Tutorial file of the mLSTM trained to predict the meaning of life.
