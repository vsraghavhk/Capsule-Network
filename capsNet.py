<<<<<<< HEAD
import os
import matplotlib 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

# Shutting up the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Getting input data from MNIST
mnist = input_data.read_data_sets("/tmp/data/")
print("MNIST data extracted.")

# 28x28 images with 1 channel
x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)

# todo: Test model with 3 channel image dataset
# 28x28 images with 3 channels
# X = tf.placeholder(shape=[None, 28, 28, 3], dtype=tf.float32, name="X")

y = tf.placeholder(shape=[None], dtype=tf.int64)

''' Parameters '''
### PrimaryCaps
# No.of maps in PrimaryCaps
PCaps_maps = 32
# No. of capsules in total
PCaps_capsules = PCaps_maps*6*6 # 1,152
# No.of Dimensions to the PrimaryCaps layer
PCaps_dims = 8

### DigitCaps
# 10 classes, 10 capsules
DCaps_capsules = 10
# 16 dimensions as defined in the paper
DCaps_dims = 16

### Decoder
fc1_dim = 512
fc2_dim = 1024
output_dim = 784

# Margin Loss
mplus = 0.9
mminus= 0.1
ml_lambda = 0.5
scale_down_factor = 0.0005
    # To let Margin loss dominate training

''' Convolution Layer '''
# Defining Convolution Layer
conv = tf.layers.conv2d(x, 
    filters=256,
    kernel_size=9,
    strides=1,
    padding="valid",
    activation=tf.nn.relu 
)

''' Primary Capsules '''
# Defining PrimaryCaps Layer
PCaps = tf.layers.conv2d(conv, 
    filters=PCaps_capsules * PCaps_dims, # 9,216
    kernel_size=9,
    strides=2,
    padding="valid",
    activation=tf.nn.relu
)

# Reshaping output from layer to get 8D vectors
PCaps_raw = tf.reshape(PCaps, [-1, PCaps_capsules, PCaps_dims])

# Squash function
    # s is input vectors. s_j from formula
    # Epsilon?
def squash(s, axis=-1, name=None):
    norm_squared = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    result = norm_squared/(1+norm_squared) * s/norm_squared
    return result

# Squashing output of PrimaryCaps
PCaps_output = squash(PCaps_raw)

print("Shape of Squashed output :\t \t \t: ", PCaps_output)

''' Digit Capsules ''' 
# For predicted output vectors
    # Standard Deviation for W initialisation
stddev = 0.1
# Trainable Weight with shape [1, 1152, 10, 16, 8]
W_initial = tf.random_normal(
    shape=(1, PCaps_capsules, DCaps_capsules, DCaps_dims, PCaps_dims),
    stddev=stddev,
    dtype=tf.float32,
)
W = tf.Variable(W_initial)

# we need two arrays.
# First. Repeat W once per every instance
    # on batch_size
batch_size = tf.shape(x)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1])

# Second. array of shape [batch_size, 1152, 10, 8,1]
    # containing o/p of first layer capsules
    # repeated 10 times for 10 classes
    # PCaps_output has shape [batch_size, 1152, 8]
    # Expand PCaps_output twice to get shape [batch_size, 1152, 1, 8, 1]
        # Then repeat it 10 times on third dimension

# Expanding
PCaps_output_expanded = tf.expand_dims(PCaps_output, -1)

# Single Tile
PCaps_output_tile = tf.expand_dims(PCaps_output_expanded, 2)

# Tiled output of PCaps (same as u in Equation 2b)
PCaps_output_tiled = tf.tile(PCaps_output_tile, 
    [1, 1, DCaps_capsules, 1, 1])

# print(W_tiled)
# print(PCaps_output_tiled)

# caps2_predicted = u
# DigitCaps predictions. u_cap from Equation 2b 
u_cap = tf.matmul(W_tiled, PCaps_output_tiled)

print("Shape of Predictions (u) :\t \t \t: ", u_cap)

''' Routing by Agreement '''
# 2 Extra dimensions to make it compatible
    # with DCaps_predicted
    # b is initial logits 
b = tf.zeros([batch_size, PCaps_capsules, DCaps_capsules, 1, 1])

#Coupling Coefficients a.k.a weights
c = tf.nn.softmax(b, dim=2)

# Weighted Predictions. 
weighted_preds = tf.multiply(c, u_cap)

# Weighted Sum. Equation 2a
s = tf.reduce_sum(weighted_preds, axis=1, keepdims=True)

print("Shape of Weighted sum (s) :\t \t \t: ", s)

### First Iteration
# DigitCaps output First Iteration 
v = squash(s, axis=-2)
print("Shape of DigitCaps output (v) :\t \t \t: ", v)

# Second Iteration
# Tiling v to match with u for agreement a
v_tiled = tf.tile(v, [1, PCaps_capsules, 1, 1, 1])

''' Agreeement '''
### Agreement a
a = tf.matmul(u_cap, v_tiled, transpose_a=True)
print("Shape of Agreement after first iteration :\t: ", a)

# Updating initial logits 
    # with recent agreements
b_2 = tf.add(b, a)
# Repeat first iteration process of agreement
b_2 = tf.nn.softmax(b_2, dim=2)
# Weighted predictions
weighted_preds_2 = tf.multiply(b_2, u_cap)
# s
s_2 = tf.reduce_sum(weighted_preds_2, axis=1, keepdims=True)
# Output of second iteration
v_2 = squash(s_2, axis=-2)
# Output of DigitCaps!
DCaps_output = v_2
print("Shape of output of DigitCaps :\t \t \t: ", DCaps_output)

# Safe(?) Norm to compute y probabilities
def norm(s, axis=-1, keepdims=False):
    with tf.name_scope("safe_norm", default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, 
            keepdims=keepdims)
        return tf.sqrt(squared_norm) # Add epsilon

# Getting y probabilities by applyng safe norm on second last data. 
    # i.e ???
y_prob = norm(DCaps_output, axis=-2)
# Max probability vector
y_max = tf.argmax(y_prob, axis=2)
print("Shape of highest probability vector is: \t: ", y_max)

# Squeezing to remove the last two dimensions to get prediction vector
y_prediction = tf.squeeze(y_max, axis=[1, 2])
print("Shape of prediction vector y is: \t \t: ", y_prediction)

''' Margin Loss ''' 
# L_k = T_k max(0, m+ - \\v_k\\)^2 + lambda(1-T_k) max(0, \\v_k\\)
Tk = tf.one_hot(y, depth=DCaps_capsules)
print("Shape of output of DigitCaps :\t \t \t: ", DCaps_output)
# 16d output is in second last dimension
DCaps_output_norm = norm(DCaps_output, axis=-2, keepdims=True)
# First term of margin loss
first_term = tf.square(tf.maximum(0., mplus - DCaps_output_norm))
first_term = tf.reshape(first_term, shape=(-1, 10))
# Second term of margin loss
second_term = tf.square(tf.maximum(0., DCaps_output_norm - mminus))
second_term = tf.reshape(second_term, shape=(-1, 10))
# Final Margin Loss
L = tf.add(Tk * first_term, ml_lambda * (1.0 - Tk) * second_term)
# reducing mean
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1))
print("Shape of margin loss is: \t \t \t: ", margin_loss)

''' Reconstruction '''
# Masking everything except 
# output vector of capsule for 
    # corresponding target class
# Creating the mask for labels 
label_mask = tf.placeholder_with_default(False, shape=())
# Applying mask for false and leaving out for true (target class)
recon_targets = tf.cond(label_mask, lambda: y,      # just y ?
    lambda: y_prediction)                           # just y_prediction ?
# Preparing the mask by one-hot and reshaping to fit the DCaps_output
recon_mask = tf.one_hot(recon_targets, depth=DCaps_capsules)
recon_mask_reshaped = tf.reshape(recon_mask, [-1, 1, DCaps_capsules, 1, 1])

# Masking the capsule outputs except for corresponding classes
DCaps_output_reconstructed = tf.multiply(DCaps_output, recon_mask_reshaped)
print("Shape of reconstructed output: \t \t \t: ", DCaps_output_reconstructed)

# Flattening the entire reconstructed DCaps_output
DCaps_output_reconstructed_flat = tf.reshape(DCaps_output_reconstructed, [-1, DCaps_capsules*DCaps_dims])
print("Shape of Reconstructed Dcaps output: \t \t: ", DCaps_output_reconstructed_flat)

''' Decoder '''
# Decoder or Final output layer

# Decoder_input = reconstruct(y, y_pred)

# 2 Fully connected layers with ReLU
fc1 = tf.layers.dense(DCaps_output_reconstructed_flat, 
    fc1_dim, 
    activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1, 
    fc2_dim,
    activation=tf.nn.relu)

# 1 Fully connected layer with output Sigmoid
fc3 = tf.layers.dense(fc2, output_dim,
    activation=tf.nn.sigmoid)

''' Reconstruction Loss '''
x_flat = tf.reshape(x, [-1, recon_output])
# Mean of square of difference
recon_loss = tf.reduce_mean(
    tf.square(
        x_flat - fc3 # fc3 = output of decoder
    )
)

''' Total Loss '''
# Total loss = Margin loss + sdf * recon loss
# "We scale down this reconstruction loss 
    #  so margin loss dominates during training."
    # Page 4, last paragraph
total_loss = tf.add(margin_loss + scale_down_factor * recon_loss)



# todo: Loss    
    # todo: Margin Loss
    # todo: Reconstruction Loss
        # Do not forget to Mask!
# todo: Training and Evaluation
