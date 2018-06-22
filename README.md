# Neural-Nets

In this tutorial we want to play with a few Neural Networks architecture for classification problems.

# Playing with "Keras + TensorFlow":

Step I: Setup an AWS instance with Deep Learning AMI which comes with a number of deep learning frameworks pre-installed in seperate virtual environments: MXNET, TensorFlow, Caffe, Caffe2, PyTorch, Keras, Chainer, Theano, and CNTK.  We selected g3.4xlarge (GPU instance) for the node with 122 GB memory.

Step II: After your instance is up and runningm try the following command for Theano(+Keras2) with Python2 (CUDA 9.0):
source activate tensorflow_p27

Step III: clone the github project and run the basic categorical classifier for Iris data.  The 3-layer NN topology had the highest accuracy among a few other simple architecture that we tried.  The reported accuracy was: 98.00% (3.06%)