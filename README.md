# UniRep, a mLSTM "babbler" deep representation learner for protein engineering informatics.

We present an interface for training, inferencing reprepresentations, generative modelling aka "babbling", and data management. All three architectures (64, 256, and 1900 units) are provided along with the trained architectures, random initializations used for evotuning (to ensure reproducibility) and the evotuned parameters.

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
