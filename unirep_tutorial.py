
# coding: utf-8

# ## How to use the UniRep mLSTM "babbler". This version demonstrates the 1900-unit architecture. On a laptop, unirep_tutorial_64.ipynb will be more RAM friendly.

# First, install the environment with your preferred method. We have tested the .yml conda load method on a conda python installation in Ubuntu 15.04, but not the others. We used pip to dump to requirements.txt which should be compatible with virtual env if you use python 3.6. 
# 
# To install something probably close enough to this environment you can use conda as follows:
# ```
# conda create -n unirep python=3.6.0 tensorflow=1.3.0 jupyter pandas
# source activate unirep
# ```
# 
# The most important version requirement is **tensorflow 1.3.0**. The others are likely fungible.

# Download the weights files if this hasn't been done already.

# In[ ]:


get_ipython().system('aws s3 sync --no-sign-request --quiet s3://unirep-public/1900_weights/ 1900_weights/')


# In[ ]:


import tensorflow as tf
import numpy as np
from unirep import babbler1900


# In[ ]:


# Initialize the babbler. You need to provide the batch size you will use and the path to the weight directory.
batch_size = 12
b = babbler1900(batch_size=batch_size, model_path="./1900_weights")


# The babbler needs to receive data in the correct format, a (batch_size, max_seq_len) matrix with integer values, where the integers correspond to an amino acid label at that position, and the end of the sequence is padded with 0s until the max sequence length to form a non-ragged rectangular matrix. We provide a formatting function to translate a string of amino acids into a list of integers with the correct codex:

# In[ ]:


seq = "MRKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


# In[ ]:


np.array(b.format_seq(seq))


# We also provide a function that will check your Amino Acid sequences don't contain any characters which will break the UniRep model.

# In[ ]:


b.is_valid_seq(seq)


# If you are into it, you could do your own data flow as long as you ensure that the data format is obeyed. Alternatively, you could use the data flow we implemented for babbler training, which happens in the tensorflow graph. It reads from a file of integer sequences, shuffles them around, collects them into groups of similar length (to minimize padding waste) and pads them to the max_length. We'll show you how to do that below:

# In[ ]:


# Before you can train your model, sequences need to be saved in the correct format
# suppose we have a new-line seperated file of AA sequences, seqs.txt, and we want to format them.
with open("seqs.txt", "r") as source:
    with open("formatted.txt", "w") as destination:
        for i,seq in enumerate(source):
            seq = seq.strip()
            if b.is_valid_seq(seq):
                formatted = ",".join(map(str,b.format_seq(seq)))
                destination.write(formatted)
                destination.write('\n')
            else:
                raise ValueError("Sequence {0} is not a valid sequence.".format(i))


# In[ ]:


# This is what the integer format looks like
get_ipython().system('head -n1 formatted.txt')


# Notice that by default format_seq does not include the stop symbol (25) at the end of the sequence. This is the correct behavior if you are trying to train a top model, but not if you are training a babbler.

# Now we can use my custom function to bucket, batch and pad sequences  from formatted.txt (which has the correct integer codex after calling)
# babbler.format_seq(). The bucketing occurs in the graph. 
# 
# What is bucketing? Specify a lower and upper bound, and interval. All sequences less than lower or greater than upper will be batched together. Interval defines the "sides" of buckets between these bounds. Don't pick a small interval for a small dataset because
# the function will just repeat a sequence if there are not enough to
# fill a batch. All batches are the size you passed when initializing the babbler.
# What else is this doing? 
# - Shuffling the sequences by randomly sampling from a 10000 sequence buffer
# - Automatically padding the sequences with zeros so the returned batch is a perfect rectangle
# - Automatically repeating the dataset (you will need synthetic epochs)

# In[ ]:


bucket_op = b.bucket_batch_pad("formatted.txt", interval=1000) # Large interval


# Inconveniently, this does not make it easy for a value to be associated with each sequence and not lost during shuffling. You can get around this by just prepending every integer sequence with the sequence label (eg, every sequence would be saved to the file as "{brightness value}, 24, 1, 5,..." and then you could just index out the first column after calling the bucket_op. Please reach out if you have questions on how to do this.

# In[ ]:


# Now that we have the bucket_op, we can simply sess.run() it to get
# a correctly formatted batch

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch = sess.run(bucket_op)
    
print(batch)
print(batch.shape)


# You can look back and see that the batch_size we passed to __init__ is indeed 12, and the second dimension must be the longest sequence included in this batch. Now we have the data flow setup (note that as long as your batch looks like this, you don't need my flow), so we can proceed to implementing the graph. The module returns all the operations needed to feed in sequence and get out trainable representations.

# In[ ]:


final_hidden, x_placeholder, batch_size_placeholder, seq_length_placeholder, initial_state_placeholder = b.get_rep_ops()


# In[ ]:


# final_hidden should be a batch_size x rep_dim matrix
# Lets say I want to train a basic feed-forward network as the top
# model, doing regression with MSE loss, Adam optimizer
y_placeholder = tf.placeholder(tf.float32, shape=[None,1], name="y")
initializer = tf.contrib.layers.xavier_initializer(uniform=False)

with tf.variable_scope("top"):
    prediction = tf.contrib.layers.fully_connected(
        final_hidden, 1, activation_fn=None, 
        weights_initializer=initializer,
        biases_initializer=tf.zeros_initializer()
    )

loss = tf.losses.mean_squared_error(y_placeholder, prediction)
    
# You can specifically train the top model first
learning_rate=.001
top_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="top")
optimizer = tf.train.AdamOptimizer(learning_rate)
top_only_step_op = optimizer.minimize(loss, var_list=top_variables)
all_step_op = optimizer.minimize(loss)


# In[ ]:


# Notice that one of the placeholder is seq_length_placeholder.
# We need to compute the lengths of the sequences in each batch so
# that we can index out the the correct final hidden.
def nonpad_len(batch):
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=1)
    return lengths

nonpad_len(batch)


# In[ ]:


# toy example where we learn to predict 42 just training the top
y = [[42]]*batch_size
num_iters = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        batch = sess.run(bucket_op)
        length = nonpad_len(batch)
        loss_, __, = sess.run([loss, top_only_step_op],
                             feed_dict={
                                 x_placeholder: batch,
                                 y_placeholder: y,
                                 batch_size_placeholder: batch_size,
                                 seq_length_placeholder:length,
                                 initial_state_placeholder:b._zero_state
                             }
                             )
        print("Iteration {0}: {1}".format(i, loss_))


# Below we train both a top model and the mLSTM. This won't work on a 16G RAM laptop. Joint recurrent/ top model training has been tested on p3.2xlarge, which has 16G of GPU RAM and 64G of system RAM. To see a demonstration of joint training on your laptop, please run  unirep_tutorial_64_unit.ipynb, which is the same architecture and interface except for 64 hidden units in 4 stacked layers.

# In[ ]:



y = [[42]]*batch_siz
num_iters = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        batch = sess.run(bucket_op)
        length = nonpad_len(batch)
        loss_, __, = sess.run([loss, all_step_op],
                             feed_dict={
                                 x_placeholder: batch,
                                 y_placeholder: y,
                                 batch_size_placeholder: batch_size,
                                 seq_length_placeholder:length,
                                 initial_state_placeholder:b._zero_state
                             }
                             )
        print(f"Iteration {i}: {loss_}")


# In[ ]:




