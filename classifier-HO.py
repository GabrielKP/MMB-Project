# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# AS.200.313, Models of Mind and Brain, Prof. Honey
#
# Project draft, Gabriel Kressin
#
# # How does a neural-network learning with Hebbs rule compare to a neural-network learning with a Hebb-Decay rule and Oja's rule regarding accuracy, learning speed and other features in a digit classification task?
#
# This project builds and compares three networks featuring biologically plausible learning rules to classify digits from the MNIST 10-digit dataset (LeCun, Y., & Cortes, C., & Burges, C.J.C.). How do these networks compare to each other regarding accuracy, learning speed and other features?
#
# The networks are built to classify 28x28 pixel images of handwritten digits correctly to a digit from 0 to 9 and only differ in their learning rule:
#     
#     1. Plain Hebbian rule. 
#     2. Hebbian-Decay rule.
#     3. Oja's learning rule.
#
#
# The project consists of 3 stages:
#
# #### Stage 1: Definition
# First, the network and learning rules are explained and defined. Additionally the data is loaded in and taken a look at.
#
# #### Stage 2: Training
# Second, the networks are trained on the data and results are plotted. Based on the results additional investigations into training order, accuracy of specific numbers and learning speed are made.
#
# #### Stage 3: Conclusion
# Finally, a conclusion is drawn based on the results and following thre criterias:
# - Classification accuracy
# - Learning speed
# - Emerging other factors

# %% [markdown]
# # Stage 1: Definition
#
# In this stage the Neurons, Networks, learning Rules and activation Functions are defined and the Data is loaded in.
#
# ## The Neuron
#
# A neuron with the input $\mathbf{x}$ and the output $\mathbf{y}$ can be defined as
#
# \begin{equation}
#     \mathbf{y}  = f(\mathbf{wx})
# \end{equation}
#
# Whereas $\mathbf{w}$ is a vector of the weights of the input and $f$ is the so called 'activation function' - a potentially nonlinear function.
#
# To make the computations more efficient, multiple Neurons are stacked together in a 'Layer'. In that case, multiple weight Vectors $\mathbf{w}$ are 'stacked' on top of each other creating a weight matrix $\mathbf{W}$ and the output becomes a vector of outputs.
#
# The 'Layer' class below implements the above mentioned framework without specifying any details on activation function and how the neuron learns. The 'Layer' class takes learning rule and activation function as initializing arguments and then provides following functions:
# - compute: computes the outputs of neurons in the layer
# - learn: updates the weights for given samples
# - getWeights: returns the weights object
# - train: trains the Layer on a dataset
#
# For convenience, to build a multi-layer network the 'Network' class is defined. It is a Wrapper for multiple stacks of Layers and defines like a Layer multiple functions:
# - compute: computes the outputs of neurons in the layer
# - learn: updates the weights for given samples
# - getWeights: returns the weights object
# - train: trains the Layer on a dataset
#
# Furthermore, the cell below features all of the functions used in the project.
#
# To load the project properly you will need to have matplotlib, numpy and pandas installed.

# %%
# %matplotlib inline

# Imports

import gzip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


# Functions used in classes

def normalizeRows( x ):
    """
    Normalizes Rows
    """
    return x / np.linalg.norm( x, axis=1 )[ :, None ]


def softmax( x ):
    """
    Converts all values in vector x so that they add up to 1
    """
    xrel = x - np.max( x ) # Handle exploding Hebb Values
    return np.exp( xrel ) / np.sum( np.exp( xrel ), axis=0 )


def runTest( X, y, network ):
    """
    Computes for given X and y data the amount of correct predictions by the given network.
    Requires the predictions being higher then 0.
    If there is multiple predictions with the same value, the lowest digit of those is taken.
    """
    assert isinstance( network, Layer ) or isinstance( network, Network ), "Not given a 'Layer' or 'Network' object in network argument!"
    assert X.shape[0] == y.shape[0], "X shape does not match y shape!"

    # Convert Labels into digits
    y = asDigits( y )

    # Compute predictions
    preds = np.empty( y.shape )
    for i in range( X.shape[0] ):
        predvec = network.compute( X[i] )
        # Require predictions to be over 0
        if np.sum( predvec ) == 0:
            preds[i] = None
        else:
            preds[i] = np.argmax( softmax( predvec ) )

    # Compare
    comp = preds == y
    correct = sum( comp.astype( np.int ) ) / y.shape[0]
    indexWrong = np.where( comp == False )
    return correct, indexWrong


# Classes

class Layer():
    """
    A Layer object includes all neurons in a layer.
    It saves the weights in Matrix form.
    The weights can be updated whilst computing the results.
    """

    def __init__( self,
                    nInputs,
                    nNeurons,
                    activationFunction=( lambda x: x ),
                    learning=( lambda w, x, y, eta: w + eta * np.outer( y, x.T ) ),
                    random=False,
                    normalize=False ):
        """
        nInputs: amount of input neurons to layer
        nNeurons: amount of neurons in layer (==outputs)
        activationFunction: potentially nonlinear function for the activation of the neuron: standard: linear Neuron
        learning: learning Rule for Layers, standard: is simple Hebbian
        random: initialize weights randomly - will normalize weights to 1, standard: False
        normalize: should weights be normalized after every learning step, standard: False
        """
        self.layerShape = ( nNeurons, nInputs )
        self.aF = activationFunction
        self.learning = learning
        self.normalize = normalize

        if random:
            self.weights = normalizeRows( np.random.uniform( low=-1, high=1, size=self.layerShape ) )
        else:
            self.weights = np.zeros( self.layerShape )


    def setWeights( self, weights ):
        """
        Sets the Layers weights
        """
        assert self.weights.shape == weights.shape, "New Weights have wrong shape"
        self.weights = weights


    def getWeights( self ):
        """
        Gets the Layers weights
        """
        return self.weights


    def learn( self, x, y, eta=0.25 ):
        """
        X: input data
        y: output data
        eta: learning rate
        """
        self.weights = self.learning( self.weights, x, y, eta )
        if self.normalize:
            self.weights = normalizeRows( self.weights )
            # Deal with rows which where completely 0
            self.weights[ np.isnan( self.weights ) ] = 0


    def compulearn( self, x, y=None, eta=0.25 ):
        """
        Computes a Prediction with current weights from input x,
        Learns the weights depending on output:
        y: Label that should be learned, if None the computed result will be taken to learn
        """

        # 1. Compute Result
        res = self.compute( x )

        # 2. Learn the network
        if y is None:
            self.learn( x, res, eta )
        else:
            self.learn( x, y, eta )

        return res


    def compute( self, x ):
        """
        Computes a Prediction with current weights from imput x.
        """
        return self.aF( np.dot( self.weights, x ) )


    def train( self, Xt, yt, X_val, y_val, epochs, eta, permute, verbose=True, decay=1 ):
        """
        Trains the neural network on training set for given epochs
        Returns history of training accuracy over validation set for each epoch
        """
        assert Xt.shape[0] == yt.shape[0], "X shape does not match y shape!"
        assert X_val.shape[0] == y_val.shape[0], "X shape does not match y shape (Val Data)!"
        
        hist = []

        for x in range( epochs ):
            print( f"Epoch { x + 1 }: ", end='' )
            if permute:
                permutation = np.random.permutation( Xt.shape[0] )
            else:
                permutation = list( range( Xt.shape[0] ) )
    
            for i in permutation:
                self.learn( Xt[i], yt[i], eta )

            # Compute validation, save and print
            correct, _ = runTest( X_val, y_val, self )
            hist.append( correct )
            if verbose:
                print( f"Val: {correct:.4f} Eta: {eta}" )

            eta *= decay
                
        return hist


class Network:
    """
    Wrapperclass to hold multiple layers.
    """

    def __init__( self, compute=None, learn=None ):
        """
        compute: function which computes outputs to inputs
        learn: function which learns the networks
        """
        self.compute = compute
        self.learn = learn


    def setCompute( self, compute ):
        """
        Sets the compute function
        """
        self.compute = compute


    def compute( self, x ):
        """
        Computes a Prediction with current weights from imput x.
        """
        assert self.compute is not None, "compute not set!"
        return self.compute( x )


    def setLearn( self, learn ):
        """
        Sets the learn function
        """
        self.learn = learn


    def learn( self, x, y, eta=0.25 ):
        """
        X: input data
        y: output data
        eta: learning rate
        """
        assert self.learn is not None, "learn not set!"
        self.learn( x, y, eta )


#     def train( self, X, y, epochs, eta, seed=None, plot=False ):
#         """
#         Trains the neural network on training set for given epochs
#         """
#         assert X.shape[0] == y.shape[0], "X shape does not match y shape!"

#         # Set seed
#         np.random.seed( seed )
#         for x in range( epochs ):
#             print( f"Epoch { x + 1 }: ", end='' )
#             for i in np.random.permutation( X.shape[0] ):
#                 self.learn( X[i], y[i], eta )

#             # Pick last 10% and compute the hit rate on them
#             lindex = int( X.shape[0] * 0.9 )
#             correct, _ = runTest( X[lindex:], y[lindex:], self )
#             print( f"Val: { correct }" )

#         # @todo: plot



# Other Functions

def runPrintTest( X, y, network, name="" ):
    """
    runs a test given X and y with network and prints the result,
    returns amount of correct classified elements and indices of wrong ones
    """
    correct, indicesWrong = runTest( X, y, network )
    print( f"{name} {correct}/{y.shape[0]} correct: { correct/y.shape[0] * 100 } %" )
    return correct, indicesWrong


def trainNewNetworksAndTest( X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            runs,
                            epochs,
                            learningRules,
                            decay=1,
                            permute=True,
                            N_INPUT=28*28,
                            N_OUTPUT=10,
                            eta=0.1,
                            verbose=True
                           ):
    """
    Trains 3 new networks and returns dictionaries for accuracies, wrongly classified indices and
    history of Validationa accuracies for each epoch, run and network.
    X_train: Training data
    y_train: Training labels
    X_val: Validation data
    y_val: Validation labels
    X_test: Testing data
    y_test: Tesing labels
    decay: Decay learning coefficient eta by this rate, set to 1 for no decay
    runs: Divisible by 2, Amount of different testruns for each network
    epochs: Amount of iterations through testset for a single network
    learningRules: 3 learning rules to be used
    N_INPUT: input number of neurons
    N_OUTPUT: output number of neurons
    eta: learning rate
    """
    r_hebb, r_decay, r_ojas = learningRules
    # Create a dictionaries with all the networks and activationFunctions
    accuracies = { 'hebb': [], 'deca': [], 'ojas': [] }
    wrongIndices = { 'hebb': [], 'deca': [], 'ojas': [] }
    valHistory = { 'hebb': [], 'deca': [], 'ojas': [] }


    for run in range( runs ):
        print( f"Run Number {run + 1}" )
        # Initialize Networks
        hebb = Layer( N_INPUT, N_OUTPUT, learning=r_hebb )
        deca = Layer( N_INPUT, N_OUTPUT, learning=r_decay )
        ojas = Layer( N_INPUT, N_OUTPUT, learning=r_ojas, normalize=True )
        # Train
        print( "Hebb" )
        np.random.seed( run )
        hisHebb = hebb.train( X_train, y_train, X_val, y_val, decay=decay, permute=permute, epochs=epochs, eta=eta, verbose=verbose )
        print( "Decay" )
        np.random.seed( run )
        hisDeca = deca.train( X_train, y_train, X_val, y_val, decay=decay, permute=permute, epochs=epochs, eta=eta, verbose=verbose )
        print( "Oja")
        np.random.seed( run )
        hisOjas = ojas.train( X_train, y_train, X_val, y_val, decay=decay, permute=permute, epochs=epochs, eta=eta, verbose=verbose )
        # Run test after training
        hebb_post_acc, hebb_post_iWrong = runTest( X_test, y_test, hebb )
        deca_post_acc, deca_post_iWrong = runTest( X_test, y_test, deca )
        ojas_post_acc, ojas_post_iWrong = runTest( X_test, y_test, ojas )
        # Save data in the dictionaries
        accuracies['hebb'].append( hebb_post_acc )
        accuracies['deca'].append( deca_post_acc )
        accuracies['ojas'].append( ojas_post_acc )
        wrongIndices['hebb'].append( hebb_post_iWrong )
        wrongIndices['deca'].append( deca_post_iWrong )
        wrongIndices['ojas'].append( ojas_post_iWrong )
        valHistory['hebb'].append( hisHebb )
        valHistory['deca'].append( hisDeca )
        valHistory['ojas'].append( hisOjas )

    print( "Done" )
    return accuracies, wrongIndices, valHistory


def readImages( path ):
    """
    Reads images from idx dataformat into np array
    Code partly from: https://stackoverflow.com/a/53570674
    """
    with gzip.open( path ) as f:
        f.read(4)   # Jump Magic Number
        nImages = int.from_bytes( f.read(4), "big" )
        x = int.from_bytes( f.read(4), "big" )
        y = int.from_bytes( f.read(4), "big" )
        print( f"Images: {nImages}; Size: x:{x}, y:{y};" )

        # Read the data in
        buf = f.read( x * y * nImages )
        data = np.frombuffer( buf, dtype=np.uint8 ).astype( np.int64 )
        return data.reshape( nImages, x * y )


def readLabels( path ):
    """
    Reads labels from idx dataformat into np aray
    Code partly from: https://stackoverflow.com/a/53570674
    """
    with gzip.open( path ) as f:
        f.read(4)   # Jump Magic Number
        nLabels = int.from_bytes( f.read(4), "big" )
        print( f"Labels: {nLabels};" )

        # Read the labels in
        buf = f.read( nLabels )
        labels = np.frombuffer( buf, dtype=np.uint8 ).astype( np.int64 )
        return labels


def plotData( images, labels, n ):
    """
    Prints n random images with their labels from given images
    Code adapted from: https://azure.microsoft.com/de-de/services/open-datasets/catalog/mnist/
    """
    # Get images in right format:
    images = np.reshape( images, ( images.shape[0], 28, 28 ) )
    # Convert labels to digits:
    labels = asDigits( labels )
    plt.figure( figsize=( 16, 6 ) )
    for i, x in enumerate( np.random.permutation( images.shape[0] )[:n] ):
        plt.subplot( 1, n, i + 1 )
        plt.axhline( "" )
        plt.axvline( "" )
        plt.text( x=10, y=-10, s=labels[x], fontsize=21 )
        plt.imshow( images[x], cmap=plt.cm.Greys )


def printStats( xs, topFive=False ):
    """
    prints basic information about a numpy array along axis 1
    """
    if topFive:
        print( f"Top 5 entries:\n { xs[ 0:5 ] }" )
    print( f"Mean: { np.mean( xs, axis = 0 ) }" )
    print( f"Max : { np.amax( xs, axis = 0 ) }" )
    print( f"Min : { np.amin( xs, axis = 0 ) }" )


def asDigits( labels ):
    """
    Turns One-Hot-Vector encodings to digits, returns a numpy array
    """
    return np.argmax( labels, axis=1 )

    
def plotDistribution( labels, title="" ):
    """
    Plots distribution of digits in dataset
    Code from: https://stackoverflow.com/a/51475132
    and https://stackoverflow.com/a/53073502
    """
    bins = [0,1,2,3,4,5,6,7,8,9,10]
    heights, _ = np.histogram( asDigits( labels ), bins )
    percent = [ i / sum( heights ) * 100 for i in heights ]
    
    f, ax = plt.subplots( 1, 1 )
    ax.bar( bins[:-1], percent, width=0.8, color="grey" )
    ax.set_ylim( [0, 15] )
    
    # axis labels
    ax.set_ylabel( "Percentage of entire dataset in %" )
    ax.set_xlabel( "image labels" )
    
    # x ticks
    ax.set_xticks( bins[:-1] )
    
    # numbers above bars
    for i, v in enumerate( percent ):
        plt.text( bins[i] - 0.4, v + 0.4, f"{v:.2f}%", rotation=45)
    plt.title( title )
    plt.show()
    
def computeAccPerLabel( y, wrongIndices ):
    """
    Computes Accuracy for given testset y and wrongly marked indices
    """
    y = asDigits( y )
    bins = [0,1,2,3,4,5,6,7,8,9,10]
    distr, _ = np.histogram( y, bins )
    
    # Get labels for wrong indices
    wrongIndiceLabels = y[wrongIndices]
    wrongs, _ = np.histogram( wrongIndiceLabels, bins )
    
    return ( distr - wrongs ) / distr

def plotNumberAccFromWrong( y, wrongHebb, wrongDeca, wrongOjas, title ):
    """
    Plots a bar graph with labels on x axis and accuracy on y axis with bars for all learning rules
    """
    percentHebb = computeAccPerLabel( y, wrongHebb ) * 100
    percentDeca = computeAccPerLabel( y, wrongDeca ) * 100
    percentOjas = computeAccPerLabel( y, wrongOjas ) * 100

    width = 1
    bins = np.array( range(11) )

    f, ax = plt.subplots( 1, 1, figsize=(15, 5) )
    ax.bar( bins[:-1] * 4, percentHebb, width=width )
    ax.bar( bins[:-1] * 4 + width, percentDeca, width=width )
    ax.bar( bins[:-1] * 4 + width * 2, percentOjas, width=width )
    ax.set_ylim( [0, 115] )

    # axis labels
    ax.set_ylabel( "Accuracy in %" )
    ax.set_xlabel( "image labels" )

    # x ticks
    ax.set_xticks( bins[:-1] * 4 + width )
    ax.set_xticklabels( bins[:-1] )

    # numbers above bars
    offsetx = -0.2
    offsety = 1.5
    for i, v in enumerate( percentHebb ):
        plt.text( bins[i] * 4 + offsetx, v + offsety, f"{v:.2f}%", rotation=90, fontsize=9 )
    for i, v in enumerate( percentDeca ):
        plt.text( bins[i] * 4 + width + offsetx, v + offsety, f"{v:.2f}%", rotation=90, fontsize=9 )
    for i, v in enumerate( percentOjas ):
        plt.text( bins[i] * 4 + width * 2 + offsetx, v + offsety, f"{v:.2f}%", rotation=90, fontsize=9 )

    plt.legend( ["Hebbian", "Decay", "Oja"] )
    plt.title( title )
    plt.show()

def plotAccuracies( accs, learningRuleNames, title ):
    plt.figure( figsize=( 10, 7 ) )
    plt.title( title )
    plt.ylim( [0, 100] )
    bWidth = 0.8
    xs = range( len( accs ) )

    percent = [ x * 100 for x in accs ]
    # plot the bars
    bars = plt.bar( xs, percent , width=bWidth, align='center' )

    # Colors
    bars[1].set_color( 'orange' )
    bars[2].set_color( 'green' )

    # Set x axis and labels
    plt.xticks( xs, learningRuleNames )
    plt.xlabel( 'Learning Rules' )
    plt.ylabel( 'Accuracy in %' )

    # numbers above bars
    for i, v in enumerate( percent ):
        plt.text( xs[i] - 0.09, v + 0.7, f"{v:.2f}%" )


# %% [markdown]
# ## Learning rules
#
# The learning rules define how exactly a neuron updates its weights given a specific input and output. All learning rules in this project are 'biologically plausible' in the sense of them being local learning rules.
#
# ### Plain Hebb rule
#
# Hebbs Rule can be summarized as "What fires together, wires together".
# The weights $\mathbf{W}$ are updated according to the given input, if the neuron was supposed to be activated. In other words, given a pair $(\mathbf{x}, \mathbf{y})$ the updated weights $\mathbf{\hat{W}}$ are computed:
#
# \begin{equation}         
# \mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}\mathbf{x}^{T}
# \end{equation} 
#
# ### Hebb with decay rule
#
# Hebbs plain Rule has a big drawback: the weights explode indefinitly to infinity. To stop that from happening a decay term is introduced (Amato et al. 2019, p.3). This leads to following equation:
#
# \begin{equation}         
# \mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}( \mathbf{x} - \mathbf{W} )
# \end{equation}
#
# ### Oja's Rule
#
# Another way to stop the weight explosion is by normalizing the weights of each neuron to 1. Additionally the 'forgetting' part is limited to the correct outputs. This gives rise to Oja's rule. This leads to other interesting effects, such as that after enough learning attempts the weights of a single neuron represent the first principal component towards the learnt activation. Besides normalizing the weights of each Neuron to 1 after each learning iteration, Oja's rule defines:
#
# \begin{equation}         
# \mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}( \mathbf{x} - \mathbf{y} \mathbf{W} )
# \end{equation}
#
# ### Rules in Python
#
# Finally, the implementation of the rules in Python! For this project the learning rules are implemented as lambda functions. It is important to keep in mind that they need to work for multiple Neurons stacked on top of each other.

# %%
# Learning rules in Python
r_hebb = lambda W, x, y, eta: W + eta * np.outer( y, x.T )
r_decay = lambda W, x, y, eta: W + eta * ( ( x - W ) * y[ :, None ] )
r_ojas = lambda W, x, y, eta: W + eta * ( ( x - W * y[ :, None ] ) * y[ :, None ] )

learningRules = ( r_hebb, r_decay, r_ojas, )
learningRuleNames = ( "Hebbian", "Decay", "Oja", )
lRs = { "Hebbian": r_hebb, "Decay": r_decay, "Oja": r_ojas }

# %% [markdown]
# ## Activation functions
#
# For correctness of the above mentioned learning rules a linear Neuron is assumed. To evaluate the output of all neurons, the output vector  of all activations $\mathbf{y}$ is fed through a softmax which converts them to a value between 0 and 1, so that they all sum together to 1.
#
# #### Linear activation function
#
# \begin{equation}
#     f(x) = x
# \end{equation}
#
# #### The Softmax
#
# \begin{equation}
#     f(y_i) = \frac{e^{y_i}}{\sum_y e^{y}}
# \end{equation}

# %% [markdown]
# ## The Data
#
# The MNIST Database provides 60.000 training examples and 10.000 test examples without needing to preprocess or format them.
#
# First, the data needs to be loaded in, there is 2 things to keep in mind:
# - The labels are converted into One-Hot-Encodings. ( e.g. 1 -> [0,1,0,0,...], 2 -> [0,0,1,0,...] )
# - The images have pixel values from 0 to 255, so the data is divided by 255 to have all data between 0 and 1.
# - 92% of the training examples will be used for training, 8% for validation during training

# %%
print( "Train & Validation" )
data = ( readImages( "data/train-images-idx3-ubyte.gz" ) / 255 )
labels = np.array( [ np.array( [ 1 if x == label else 0 for x in range(10) ] ) for label in readLabels( "data/train-labels-idx1-ubyte.gz" ) ] )

# Pick last 8% as validation dataset
lindex = int( data.shape[0] * 0.92 )
X_train = data[:lindex]
y_train = labels[:lindex]
X_val = data[lindex:]
y_val = labels[lindex:]

print( "\nTest" )
X_test = ( readImages( "data/t10k-images-idx3-ubyte.gz" ) / 255 )
y_test = np.array( [ np.array( [ 1 if x == label else 0 for x in range(10) ] ) for label in readLabels( "data/t10k-labels-idx1-ubyte.gz" ) ] )

# %% [markdown]
# This is how the train data looks like:

# %%
plotData( X_train, y_train, 20 )

# %% [markdown]
# Validation Data:

# %%
plotData( X_val, y_val, 20 )

# %% [markdown]
# And the test data:

# %%
plotData( X_test, y_test, 20 )

# %% [markdown]
# The numbers are distributed as follows for all data sets:

# %%
plotDistribution( y_test, "Test data distribution" )
plotDistribution( y_val, "Validation data distribution" )
plotDistribution( y_train, "Train data distribution" )

# %% [markdown]
# # Stage 2: Training
#
# In this section the three networks are trained.

# %% [markdown]
# ### First training
#
# First, the three networks are initialized to weights of 0, and trained on all on the same random permutations of the training data for 10 epochs in 10 different runs. I found that the performance was best with a learning rate of 0.1 and a decay rate for 0.4. Be warned, training may take some time. Furthermore there is a runtime warning which does not affect the outcome of the training.

# %%
epochs = 10
runs = 4
accuracies, wrongIndices, valHistory = trainNewNetworksAndTest( X_train,
                                                               y_train,
                                                               X_val,
                                                               y_val,
                                                               X_test,
                                                               y_test,
                                                               runs,
                                                               epochs,
                                                               learningRules,
                                                               eta=0.1,
                                                               decay=0.4
                                                              )

# %% [markdown]
# The runs are averaged:

# %%
# Average the accuracies
avgAccs = dict()
for network in accuracies.keys():
    avgAccs[network] = np.average( accuracies[network] )

# Average the epoch accuracies
avgValHis = dict()
for network in valHistory.keys():
    avgValHis[network] = np.average( valHistory[network], axis=0 )
    
# Average 

# %% [markdown]
# ...and then visualized:

# %%
# Testset accuracy    
plotAccuracies( avgAccs.values(), learningRuleNames, f"Average classification accuracy on Testset after {epochs} Epochs" )

# %%
# Run Validation accuracy throughout epochs
plt.figure( figsize=( 15, 10 ) )
xs = range( 1, epochs + 1 )

for r in range( runs ):
    plt.subplot( int( runs / 2 ), 2, r + 1 )
    plt.ylim( [0, 100] )
    plt.plot( xs, np.array( valHistory['hebb'][r] ) * 100 )
    plt.plot( xs, np.array( valHistory['deca'][r] ) * 100 )
    plt.plot( xs, np.array( valHistory['ojas'][r] ) * 100 )
    plt.title( f"Run {r}" )

    # Axes
    plt.xlabel( 'Epochs' )
    plt.xticks( xs )
    plt.ylabel( 'Accuracy in %' )

# Legend
plt.suptitle( "Classification Accuracy on Validation Data by epoch for each run" )
plt.legend( learningRuleNames )
plt.tight_layout();

# %%
# Average Validation accuracy throughout epochs
plt.figure( figsize=( 10, 7 ) )
plt.title( "Average classification Accuracy on Validation Data by epoch" )
plt.ylim( [0, 1] )
xs = range( 1, epochs + 1 )

plt.plot( xs, avgValHis['hebb'] )
plt.plot( xs, avgValHis['deca'] )
plt.plot( xs, avgValHis['ojas'] )

# Axes
plt.xlabel( 'Epochs' )
plt.xticks( xs )
plt.ylabel( 'Accuracy in %' )

# Legend
plt.legend( learningRuleNames );

# %%
run = 0
plotNumberAccFromWrong( y_test,
                       wrongIndices['hebb'][run],
                       wrongIndices['deca'][run],
                       wrongIndices['ojas'][run],
                       f"Accuracy per number for Run {run}"
                      )
plotAccuracies( [ accuracies['hebb'][run], accuracies['deca'][run], accuracies['ojas'][run] ],
               learningRuleNames,
               f"Accuracy of Networks in Run {run}"
              )
# @todo avergae

# %% [markdown]
# There are some interesting observations for these results:
# 1. The Oja network has the best accuracy
# 2. The Hebbian network has a steady accuracy from the first epoch on
# 3. The Hebbian and the Decay rule have difficulties classifying the 5.
# 4. The Decay Rule works shows great variability in classification accuracy on the validation set

# %% [markdown]
# The second observation is easily explained: Whereas the other rules have some sort of "forgetting" term, the plain Hebbian rule only adds input to expected output relations to the weights. Because these are independent on the current weights of the network, and because addition is associative, the order of training examples does not matter. This is also the reason the accuracy for the plain Hebbian network stays exactly the same through time, as after each epoch the weights just changed in the exact same proportions as in the epoch before.
#
# The second observation highly suggests that the deciding factor for classification accuracy is training order. But how does training order affect the classification accuracy?

# %% [markdown]
# ### The effect of training order on classification accuracy
#
# If training order is the
#
# First a few different training orders are created:
# 1. One number in the end
# 2. As even as possible
# 3. "Difficult" numbers such as 5 towards the end
# 4. 

# %%
# Split testset into different numbers
numsData = []
numsLabels = []
dig = asDigits( y_train )
for i in range( 10 ):
    numsData.append( X_train[dig == i] )
    numsLabels.append( y_train[dig == i] )

# %%
# Distribute numbers evenly
X_trainEven = np.zeros( X_train.shape )
y_trainEven = np.zeros( y_train.shape )
turn = 0
numIdx = 0
i = 0
while i < X_train.shape[0]:
    for c in range( 10 ):
        if numIdx < numsLabels[c].shape[0]:
            X_trainEven[i] = numsData[c][numIdx]
            y_trainEven[i] = numsLabels[c][numIdx]
            i += 1
    numIdx += 1

# Reverse so that uneven "excess" numbers are at the beginning
X_trainEven = np.flip( X_trainEven, axis=0 )
y_trainEven = np.flip( y_trainEven, axis=0 )

# %%
numsLabels[1].shape

# %%
accuraciesEven, wrongIndicesEven, valHistoryEven = trainNewNetworksAndTest( X_trainEven,
                                                                           y_trainEven,
                                                                           X_val,
                                                                           y_val,
                                                                           X_test,
                                                                           y_test,
                                                                           runs=10,
                                                                           epochs=3,
                                                                           permute=False,
                                                                           eta=0.1,
                                                                           decay=0.4
                                                                          )

run = 0
plotNumberAccFromWrong( y_test,
                       wrongIndicesEven['hebb'][run],
                       wrongIndicesEven['deca'][run],
                       wrongIndicesEven['ojas'][run],
                       f"Accuracy per number for Run {run}"
                      )
plotAccuracies( [ accuraciesEven['hebb'][run], accuraciesEven['deca'][run], accuraciesEven['ojas'][run] ],
               learningRuleNames,
               f"Accuracy of Networks in Run {run}"
              )

# %% [markdown]
# ### Why do the networks struggle with the 5?
#
# Which labels does the 5 get?
# Visualization of weights

# %% [markdown]
# ## Learning Speed Comparison
#
# Although the Oja network shows a better accuracy then the Hebbian network, the hebbian network performs at maximal classification accuracy just after one epoch. But how much faster exactly does the hebbian network learn?
#
# To answer the question an Ojas, Decay and Hebbian networks are trained in the same random order. To monitor their learning speed after every N steps they are tested on the test-dataset. The experiment is repeated 10 times and the results are averaged.

# %%
# WARNING: This may takes an hour of running time with current parameters!
runs = 3
N_INPUT = 28 * 28
N_OUTPUT = 10
eta=0.1
decay=0.4
N = 1000
offset = X_train.shape[0] % N
resultLength = int( X_train.shape[0] / N )

# Result has shape (learningRules, runs, resultLength)
results = { lR: [] for lR in lRs }
for r in range( runs ):
    print( f"Run {r}" )

    # Init networks and result arraays
    networks = dict()
    for lR in lRs:
        # Network
        if lR == "Oja":
            networks[lR] = Layer( N_INPUT, N_OUTPUT, learning=lRs[lR], normalize=True )
        else:
            networks[lR] = Layer( N_INPUT, N_OUTPUT, learning=lRs[lR] )
        # Results
        results[lR].append( np.zeros( resultLength ) )

    # Train and test networks
    rcount = 0
    for i, idx in enumerate( np.random.permutation( X_train.shape[0] ) ):
        for lR in lRs:
            networks[lR].learn( X_train[idx], y_train[idx], eta=eta )
        if ( ( i + offset ) % N == 0 ):
            for lR in lRs:
                results[lR][-1][rcount], _ = runTest( X_test, y_test, networks[lR] )
            rcount += 1
        if ( i % 5500 == 0 ):
            print( "+", end="" )
    print( "" )
    eta *= 0.4

# %%
# Average results
avgResults = dict()
for lR in results:
    avgResults[lR] = np.average( results[lR], axis=0 )


# %%
def plotLineGraph( dic, title, xlabel, ylabel, lRs, ticks=0, offset=0, ):
    """
    Takes dict as input, plots line graph
    """
    # Average Validation accuracy throughout epochs
    plt.figure( figsize=( 10, 7 ) )
    plt.title( title )
    plt.ylim( [0, 100] )
    xs = np.array( range( 0, len( list( dic.values() )[0] ) ) ) * ticks + offset

    for lR in lRs:
        plt.plot( xs, np.array( dic[lR] ) * 100 )

    # Axes
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    plt.grid( True, axis="y" )

    # Show offset value in xticks
    c = plt.xticks()
    print( c )
    # Legend
    plt.legend( lRs.keys() );

plotLineGraph( avgResults, "Test Accuracy during one Epoch", "Training Examples", "Accuracy in %", lRs, N, offset )

# %% [markdown]
# # Stage 3: Comparison
#
# Now that everything is defined and the architectures are explored, let's discuss the results regarding following three points:
#
# 1. Which model has a higher classification accuracy?
# 2. Which model learns faster?
# 3. Interesting other effects
#

# %% [markdown]
# ### Accuracy
#
# In this section the clear winner is the plain simple Hebbian learning rule paired with linear or relu as activation function. Still, the plain Hebbian learning rule is quite impractical in real world usage, due to the weight explosion.
#
# Not taking the plain Hebbian learning rule into account, the Oja learning rule performs on average better then the Hebbian Decay rule.
#
# @todo: include code cell from above showing results (or a table or something like that)

# %% [markdown]
# ### Learning speed
#
# Also here, the plain Hebbian learning rule is clearly the fastest. As it is independent from the ordering in training it does not matter how often it is trained, it's best classification accuracy is reached after one Epoch already
#
# @todo: include code cell creating a plot of accuracy dependent on Epochs

# %% [markdown]
# ### Emerging other effects
#
# In this category, the Oja Rule clearly wins. Due to the way it works, the weights of the Neurons resemble the first principle component of the data for the digit they are tuned to (@todo: include citation). This could be very helpful for image further analysis, for instance finding, which pixels are the most important ones.
#
# @todo: include code cell showing the principal component for each of the numbers - "which pixel is most important for a a digit"

# %% [markdown]
# ## Conclusion: How does a neural-network-classifier learning with Hebbs rule compare to a neural-network-classifier learning with Oja's rule?
#
# Whereas the network with the plain Hebbian learning rule has quite an impressive learning rate and accuracy, it is impractical due to it's weight explosion.
#
# The interesting part is the comparison of Oja's rule and Hebb's Decay Rule. In terms of accuracy, both rules provide for a image classification around 60%, with neither being completely dominant over the other. Furthermore, both rules highly depend on the order of the presented training data, making their learning rate quite unpredictable und not very stable over time. The key difference is, that Oja's network also gives you acces to the first principal component of the input data, making it very valuable for input data analysis.
# ( I can only really write more with the other results in )
#
#
# @todo: relate to state of the art

# %% [markdown]
# ## Future Work
#
# There is many interesting things to have a longer look at:
#
# - How would a spiking neural network with the same learning rules do, compared to the models shown here.
# - Is there a better and actually effective way to train the intermediate layers?
# - How would a convoutional architecture perform?
# - How would recurrance affect the prediction accuracy?

# %% [markdown]
# # References
#
# LeCun, Y., Cortes, C., & Burges, C.J.C., The MNIST Database of Handwritten Digits \[Accessed 26.11.2020 18:00 CET\]. http://yann.lecun.com/exdb/mnist/
#
# Amato, G., Carrara, F., Falchi, F., Gennaro, C., & Lagani, G.(2019). Hebbian Learning Meets Deep Convolutional Neural Networks. In: Ricci E., Rota Bulò S., Snoek C., Lanz O., Messelodi S., Sebe N. (eds) Image Analysis and Processing – ICIAP 2019. ICIAP 2019. Lecture Notes in Computer Science, vol 11751. Springer, Cham. https://doi.org/10.1007/978-3-030-30642-7_29
#
# @todo: Oja's paper

# %% [markdown]
# # Old Code Parking Lot
#
# Do not regard anything after this section, I am just parking old code in case I need it again - it will not be present in the final project.

# %% [markdown]
# ### Multi Layer Networks
#
# Now, a two layer network is created.
#
# Even though it seems simple, training now becomes non-trivial as it is unclear which output intermediat neurons should try to learn. There seem to be two approaches:
#
# 1. Initialize weights randomly, let the network learn just what it is producing and hope it creates enough diversity in outputs for a good categorization.
# 2. Force the intermediate Neurons to learn specific patterns, which arguably is not as biologically plausible anymore. A version of this was done in Amato et al. (2019), which can be seen in the chapter "3.3 Supervised Hebbian Learning".
#
# At the moment I am not getting the first approach to work properly with an ojas based network. Probably I will include both approaches in the end.
# BIG@todo: get multilayer working!

# %%
N_INPUT = 28 * 28
N_L1 = 1000
N_OUTPUT = 10

h_l1 = Layer( N_INPUT, N_L1, random=True, normalize=True, learning=r_ojas )
h_out = Layer( N_L1, N_OUTPUT, learning=r_ojas, normalize=True, random=True )

twoLayer = Network()
twoLayer.setCompute( lambda x: h_out.compute( h_l1.compute( x ) ) )
twoLayer.setLearn( lambda x, y, eta: h_out.learn( ( h_l1.compulearn( x, eta=eta ) ), y, eta=eta ) )

# %%
runPrintTest( X_test, y_test, twoLayer,"Two Layer Ojas:" );

# %% tags=[]
twoLayer.train( X_train[:20000], y_train[:20000], epochs=5, eta=0.1 )

# %%
print( runTest( X_test, y_test, twoLayer )[0] / X_test.shape[0] * 100 )

# %%
N_INPUT = 28 * 28
N_OUTPUT = 10
epochs = 1  # How often the Training Set is iterated over, Set lower to save significant amount of time
trials = 1  # Amount of different testRuns, set lower for maximum time saving

# Function to create empty arrays - to save space below
eA = lambda: [ [] for _ in range( len( activationFunctions ) ) ]

# Create a dictionary with all the networks and activationFunctions
oneLNacc = { 'hebb': eA(), 'deca': eA(), 'ojas': eA() }
oneLNind = { 'hebb': eA(), 'deca': eA(), 'ojas': eA() }


for trial in range( trials ):
    print( f"Trial Number {trial + 1}" )
    for i, aF in enumerate( activationFunctions ):
        print( activationFunctionNames[i] )
        # Initialize Networks
        hebb = Layer( N_INPUT, N_OUTPUT, learning=r_hebb, activationFunction=aF )
        deca = Layer( N_INPUT, N_OUTPUT, learning=r_decay, activationFunction=aF )
        ojas = Layer( N_INPUT, N_OUTPUT, learning=r_ojas, activationFunction=aF)
        # Run test before
        hebb_pre_acc, hebb_pre_iWrong = runTest( X_test, y_test, hebb )
        deca_pre_acc, deca_pre_iWrong = runTest( X_test, y_test, deca )
        ojas_pre_acc, ojas_pre_iWrong = runTest( X_test, y_test, ojas )
        # Train
        print( "Hebb" )
        np.random.seed( trial )
        hebb.train( X_train, y_train, epochs=epochs, eta=0.1, seed=None )
        print( "Decay" )
        np.random.seed( trial )
        deca.train( X_train, y_train, epochs=epochs, eta=0.1, seed=None )
        print( "Oja")
        np.random.seed( trial )
        ojas.train( X_train, y_train, epochs=epochs, eta=0.1, seed=None )
        # Run test after
        hebb_post_acc, hebb_post_iWrong = runTest( X_test, y_test, hebb )
        deca_post_acc, deca_post_iWrong = runTest( X_test, y_test, deca )
        ojas_post_acc, ojas_post_iWrong = runTest( X_test, y_test, ojas )
        # Save data in the dictionaries
        oneLNacc['hebb'][i].append( hebb_post_acc / nTest )
        oneLNind['hebb'][i].append( hebb_post_iWrong )
        oneLNacc['deca'][i].append( deca_post_acc / nTest )
        oneLNind['deca'][i].append( deca_post_iWrong )
        oneLNacc['ojas'][i].append( ojas_post_acc / nTest )
        oneLNind['ojas'][i].append( ojas_post_iWrong )


print( "Done" )

# %%
N_INPUT = 28 * 28
N_OUTPUT = 10

np.random.seed( 1 )

oneLNhebb = Layer( N_INPUT, N_OUTPUT )
oneLNdeca = Layer( N_INPUT, N_OUTPUT, learning=r_decay )
oneLNojas = Layer( N_INPUT, N_OUTPUT, learning=r_ojas, normalize=True )

# %%
print( "Before Training" )
runPrintTest( X_test, y_test, oneLNhebb, "Hebb: " )
runPrintTest( X_test, y_test, oneLNdeca, "Decay:" )
runPrintTest( X_test, y_test, oneLNojas, "Oja's:" );

# %%
# Training
print( "Hebb" )
oneLNhebb.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
print( "\nDecay" )
oneLNdeca.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
print( "\nOjas" )
oneLNojas.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
# Note: When initializing the Oja's Network there will be a runtime warning for true_divide. This is handled.

# %%
print( "After Training" )
runPrintTest( X_test, y_test, oneLNhebb, "Hebb: " )
runPrintTest( X_test, y_test, oneLNdeca, "Decay:" )
runPrintTest( X_test, y_test, oneLNojas, "Oja's:" );


# %% [markdown]
# These resulst are not as bad already. As expected the order of training samples does not matter to the Hebbian Network, but to the Decay and Oja Network!
#
# Now we can have a look at how the activation functions change our networks prediction.

# %%
# Legacy, just in case I need it again
# class singleNeuron():

#     def __init__( self, inputd ):
#         # self.weights = np.random.uniform( low=-1, high=1, size=( 1, inputd ) )
#         self.weights = np.zeros( ( 1, inputd ) )
#         self.bias = np.random.uniform( low=-10, high=10, size=( 1 ) )
#         print( f"weights: {self.weights}" )
#         print( f"bias: {self.bias}" )

#     def run( self, inp ):
#         """
#         Computes Neuron output for input
#         """
#         return int( np.dot( self.weights, inp ) > 0 )

#     def getLoss( self, x, y ):
#         """
#         Computes Loss for x input and y output values
#         """
#         return 1 - ( self.run( x ) == y )

#     def train( self, train_features, train_labels, learning_rate = 0.1 ):
#         """
#         Trains the weights of the neuron
#         """
        
#         def helper( x, y ):
#             self.weights = self.weights + learning_rate * y * x
#             print( self.weights )
#             if self.run( x ) == y:
#                 return True
#             return False

#         # Train until everything in trainingset is classified correctly
#         x = 0
#         while not all( [ helper( train_features[i], train_labels[i] ) for i in range( len( train_features ) ) ] ):
#             print( self.weights )
#             x = x + 1
#             if x == 1:
#                 break
#         print( self.weights )

# %% [markdown]
# ## Simple Test
#
# To make sure the learning rules work, a very simple test.
#
# This is a dataset from a [blog](https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2) discussing simple classifiers: [data](https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt).
#
# I use this data, extract only the oranges and apples and then do binary classification.
#
# The data is split into train and test. Additionally it is scaled to be between -1 and 1.

# %%
# Functions

def quicktest( s, test_features, test_labels, verbose=False ):
    correct = 0
    for i in range( len( test_features ) ):
        res = "Apple" if s.compute( test_features[i] ) else "Orange"
        if test_labels[i]:
            if verbose:
                print( f"Example { i + 1 }, Apple, Classification: { res }" )
            correct += "Apple" == res
        else:
            if verbose:
                print( f"Example { i + 1 }, Orange, Classification: { res }" )
            correct += "Orange" == res
    print( f"{ correct }/{ len( test_features ) } correct!" )

def scaleTo( xs ):
    xs = np.array( xs )
    xs = xs - np.mean( xs, axis = 0 )
    return ( xs / ( max( abs( np.min( xs, axis = 0 ) ), np.max( xs, axis = 0 ) ) ) )

# Read the fruit data in
data = pd.read_table( "data/fruit_data_with_colors.txt" )

# Exctact apples and oranges
fruits = data[( (data.fruit_name == "apple") | (data.fruit_name == "orange" ) )]

# Extract features and labels; labels: 1 == apple, 0 == orange
featurenames = ['mass', 'width', 'height', 'color_score']
fruit_features = fruits[ featurenames ].to_numpy()

printStats( fruit_features )

for i in range( len( featurenames ) ):
    fruit_features[:,i] = scaleTo( fruit_features[:,i] )

print( "After Scaling" )
printStats( fruit_features )

fruit_labels = ( fruits.fruit_name == "apple" ).to_numpy().astype( int )

# %%
# Extract random entries for test, without double drawing
random.seed( 20 )
N_TEST = 10
test_index = [ 1, 1 ]
while len( test_index ) != len( set( test_index ) ):
    test_index = [ random.randint( 0, len( fruit_features ) - 1 ) for _ in range( N_TEST ) ]
train_index = [ x for x in range( len( fruit_features ) ) if x not in test_index ]

# Test-set
test_features = fruit_features[test_index]
test_labels = fruit_labels[test_index]

# Train-set
train_features = fruit_features[train_index]
train_labels = fruit_labels[train_index]

# %% [markdown]
# Create the single Neurons using the Layer Class.

# %%
# Create the neurons
hebb = Layer( 4, 1, learning=r_hebb )
hebb_decay = Layer( 4, 1, learning=r_decay )
ojas = Layer( 4, 1, learning=lambda W, x, y, eta: W + eta * y * ( x - W * y ) )

# %%
# Categorize before Training
# Expectation: All perform the same (as they have been initialized with the same weights)
print( "Hebb" )
quicktest( hebb, test_features, test_labels )
print( "Hebb Decay" )
# quicktest( hebb_decay, test_features, test_labels )
print( "Ojas" )
quicktest( ojas, test_features, test_labels )

# %% tags=[]
# Set sampling order
random.seed( 1 )
order = [ x for x in range( len( train_features ) ) ]
# Train Neurons for 5 iterations
for x in range( 5 ):
    # Train in random order
    random.shuffle( order )
    for i in range( len( order ) ):
        ireal = order[i]
        hebb.learn( train_features[ireal], train_labels[ireal], eta=0.2 )
        # hebb_decay.learn( train_features[ireal], train_labels[ireal], eta=0.2 )
        ojas.learn( train_features[ireal], train_labels[ireal], eta=0.2 )

# %%
# Print their final weights
print( f"Hebb: { hebb.weights }" )
# print( f"Hebb_decay: { hebb_decay.weights }" )
print( f"Ojas: { ojas.weights }" )

# %%
# Categorize after Training
# Expectation: All perform the same (as they have been initialized with the same weights)
print( "Hebb" )
quicktest( hebb, test_features, test_labels )
print( "Hebb Decay" )
# quicktest( hebb_decay, test_features, test_labels )
print( "Ojas" )
quicktest( ojas, test_features, test_labels )

# %% [markdown]
# As you can see, the classification accuracy is not that great. Multiple reasons:
# - limited classification power of one single neuron + local learning rule
# - data is difficult for this learning rule to learn
# - not much data to learn from
#
# BUT, it works!
# Now that we have made that sure, we can go on to building the architecture of both classifiers!
