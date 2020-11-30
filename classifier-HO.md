---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Todos

@todo:
- x/X y/Y consistency
- Speed optimization for train/compute/learn
- Fix Oja's name to Oja.
- Decaying Learning Rate
- Hebb Decay Rule should be included in title

This draft is littered with @todos. It would be helpful to me if you let me know whether you think they are unnecessary or whether I have some I have overlooked.


AS.200.313, Models of Mind and Brain, Prof. Honey

Project draft, Gabriel Kressin

# How does a neural-network learning with Hebbs rule compare to a neural-network learning with Oja's rule regarding accuracy, learning speed and other features in a digit classification task?

This project builds and compares three networks featuring biologically plausible learning rules to classify digits from the MNIST 10-digit dataset (LeCun, Y., & Cortes, C., & Burges, C.J.C.). These networks only differ in their learning rule:
    
    1. Plain Hebbian rule. 
    2. Hebbian-Decay rule.
    3. Oja's learning rule.

The networks are built to classify 28x28 pixel images of handwritten digits correctly to a digit from 0 to 9.

The project consists of 3 stages:

#### Stage 1: Definition
First, the network, learning rules and activation functions are explained and defined. Additionally the data is loaded in and taken a look at.

#### Stage 2: Architecture
Second, the best architecture is explored by examining different combinations of hyperparameters such as amount of layers, hidden layer size and activation function. 

#### Stage 3: Comparison
Lastly, the three models are compared on following three criterias:
- Classification accuracy
- Learning speed
- Emerging other factors



# Stage 1: Definition

In this stage the Neurons, Networks, learning Rules and activation Functions are defined and the Data is loaded in.

## The Neuron

A neuron with the input $\mathbf{x}$ and the output $\mathbf{y}$ can be defined as

\begin{equation}
    \mathbf{y}  = f(\mathbf{wx})
\end{equation}

Whereas $\mathbf{w}$ is a vector of the weights of the input and $f$ is the so called 'activation function' - a potentially nonlinear function.

To make the computations more efficient, multiple Neurons are stacked together in a 'Layer'. In that case, multiple weight Vectors $\mathbf{w}$ are 'stacked' on top of each other creating a weight matrix $\mathbf{W}$ and the output becomes a vector of outputs.

The 'Layer' class below implements the above mentioned framework without specifying any details on activation function and how the neuron learns. The 'Layer' class takes learning rule and activation function as initializing arguments and then provides following functions:
- compute: computes the outputs of neurons in the layer
- learn: updates the weights for given samples
- getWeights: returns the weights object
- train: trains the Layer on a dataset

For convenience, to build a multi-layer network the 'Network' class is defined. It is a Wrapper for multiple stacks of Layers and defines like a Layer multiple functions:
- compute: computes the outputs of neurons in the layer
- learn: updates the weights for given samples
- getWeights: returns the weights object
- train: trains the Layer on a dataset

Furthermore, the cell below features all of the functions used in the project.

To load the project properly you will need to have matplotlib, numpy and pandas installed.

```python
%matplotlib inline

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
        predcan = np.where( ( predvec == np.amax( predvec ) ) & ( predvec > 0 ) )[0] # Candidates
        preds[i] = None if predcan.shape[0] == 0 else predcan[0]   # Take first candidate

    # Compare
    comp = preds == y
    correct = sum( comp.astype( np.int ) )
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
        activationFunction: potentially nonlinear function for the activation of the neuron: standard: 0 Threshhold
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


    def train( self, X, y, epochs, eta, seed=None, plot=False ):
        """
        Trains the neural network on training set for given epochs
        """
        assert X.shape[0] == y.shape[0], "X shape does not match y shape!"

        # Set seed
        np.random.seed( seed )
        for x in range( epochs ):
            print( f"Epoch { x + 1 }: ", end='' )
            for i in np.random.permutation( X.shape[0] ):
                self.learn( X[i], y[i], eta )

            # Pick last 10% and compute the hit rate on them
            lindex = int( X.shape[0] * 0.9 )
            correct, _ = runTest( X[lindex:], y[lindex:], self )
            print( f"Val: { correct }/{ X[lindex:].shape[0] }" )

        # @todo: plot


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


    def train( self, X, y, epochs, eta, seed=None, plot=False ):
        """
        Trains the neural network on training set for given epochs
        """
        assert X.shape[0] == y.shape[0], "X shape does not match y shape!"

        # Set seed
        np.random.seed( seed )
        for x in range( epochs ):
            print( f"Epoch { x + 1 }: ", end='' )
            for i in np.random.permutation( X.shape[0] ):
                self.learn( X[i], y[i], eta )

            # Pick last 10% and compute the hit rate on them
            lindex = int( X.shape[0] * 0.9 )
            correct, _ = runTest( X[lindex:], y[lindex:], self )
            print( f"Val: { correct }/{ X[lindex:].shape[0] }" )

        # @todo: plot



# Other Functions

def runPrintTest( X, y, network, name="" ):
    """
    runs a test given X and y with network and prints the result,
    returns amount of correct classified elements and indices of wrong ones
    """
    correct, indicesWrong = runTest( X, y, network )
    print( f"{name} {correct}/{y.shape[0]} correct: { correct/y.shape[0] * 100 } %" )
    return correct, indicesWrong


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
```

## Learning rules

The learning rules define how exactly a neuron updates its weights given a specific input and output. All learning rules in this project are 'biologically plausible' in the sense of them being local learning rules.

### Plain Hebb rule

Hebbs Rule can be summarized as "What fires together, wires together".
The weights $\mathbf{W}$ are updated according to the given input, if the neuron was supposed to be activated. In other words, given a pair $(\mathbf{x}, \mathbf{y})$ the updated weights $\mathbf{\hat{W}}$ are computed:

\begin{equation}         
\mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}\mathbf{x}^{T}
\end{equation} 

### Hebb with decay rule

Hebbs plain Rule has a big drawback: the weights explode indefinitly to infinity. To stop that from happening a decay term is introduced (Amato et al. 2019, p.3). This leads to following equation:

\begin{equation}         
\mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}( \mathbf{x} - \mathbf{W} )
\end{equation}

### Oja's Rule

Another way to stop the weight explosion is by normalizing the weights of each neuron to 1. Additionally the 'forgetting' part is limited to the correct outputs. This gives rise to Oja's rule. This leads to other interesting effects, such as that after enough learning attempts the weights of a single neuron represent the first principal component towards the learnt activation. Besides normalizing the weights of each Neuron to 1 after each learning iteration, Oja's rule defines:

\begin{equation}         
\mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}( \mathbf{x} - \mathbf{y} \mathbf{W} )
\end{equation}

### Rules in Python

Finally, the implementation of the rules in Python! For this project the learning rules are implemented as lambda functions. It is important to keep in mind that they need to work for multiple Neurons stacked on top of each other.

```python
# Learning rules in Python
r_hebb = lambda W, x, y, eta: W + eta * np.outer( y, x.T )
r_decay = lambda W, x, y, eta: W + eta * ( ( x - W ) * y[ :, None ] )
r_ojas = lambda W, x, y, eta: W + eta * ( ( x - W * y[ :, None ] ) * y[ :, None ] )
# @todo: @question: does nonlinearity need to be taken into account for the learning rule?
```

## Activation functions

For correctness of the above mentioned learning rules a linear Neuron is assumed. But by abiding to certain rules, we can introduce nonlinear functions which may lead to interesting properties and potentially could improve the classification accuracy! (@todo: include citation of nonlinear Hebb networks with interesting properties (and check on them(?))

On another note, nonlinear activation functions are present in our human brain and thus a good argument to make our learning rules more biologically plausible (@todo: another citation needed).

Below this section a visualization of the activation functions can be found.

### Linear

This one is simple:

\begin{equation}
    f(x) = x
\end{equation}

### Simple threshhold

The 'simple threshhold' simply classifies things into 0 or 1. Firing or not firing. (be warned: it does not seem to work at the moment)

\begin{equation}
    f(x) =
    \begin{cases}
        1 &\text{if } x > 0.5 \\
        0 &\text{else}
    \end{cases}
\end{equation}

### Sigmoid

The 'sigmoid' function squashes all outputs between 0 and 1 in a nonlinear way:

\begin{equation}
    f(x) = \frac{1}{1 + e^{-x}}
\end{equation}

### ReLU

'ReLU' stands for 'Rectified Linear Unit'. For every $x$ below 0 the activation stays 0, above 0 it behaves like the linear activation function.

\begin{equation}
    f(x) = \max{( 0, x )}
\end{equation}

### Activation functions in Python

Also for the activation functions lambda functions are used. Importantly, the functions need to be able to handle vectors!

```python
# Definition of activation functions
linear = lambda x: x
threshhold = lambda x: ( x > 0.5 ).astype( np.int )
sigmoid = lambda x: ( 1 / ( 1 + np.exp( -x ) ) )
relu = lambda x: np.maximum( 0, x )

# Create an array of activation functions for later convenience
activationFunctions = [ linear, threshhold, sigmoid, relu ]
activationFunctionNames = [ "Linear", "Threshhold", "Sigmoid", "ReLU" ]
```

Visualizing the activation functions:

```python
test = np.linspace( -2, 2, 300 )
plt.plot( test, linear( test ) )
plt.plot( test, threshhold( test ) )
plt.plot( test, sigmoid( test ) )
plt.plot( test, relu( test ) )
plt.legend( activationFunctionNames );
```

## The Data

The MNIST Database provides 60.000 training examples and 10.000 test examples without needing to preprocess or format them.

First, the data needs to be loaded in, there is 2 things to keep in mind:
- The labels are converted into One-Hot-Encodings. ( e.g. 1 -> [0,1,0,0,...], 2 -> [0,0,1,0,...] )
- The images have pixel values from 0 to 255, so the data is divided by 255 to have all data between 0 and 1.

```python
print( "Train" )
X_train = readImages( "data/train-images-idx3-ubyte.gz" ) / 255
y_train = np.array( [ np.array( [ 1 if x == label else 0 for x in range(10) ] ) for label in readLabels( "data/train-labels-idx1-ubyte.gz" ) ] )

print( "\nTest" )
X_test = readImages( "data/t10k-images-idx3-ubyte.gz" ) / 255
y_test = np.array( [ np.array( [ 1 if x == label else 0 for x in range(10) ] ) for label in readLabels( "data/t10k-labels-idx1-ubyte.gz" ) ] )
```

Visualizing some of the train data:

```python
plotData( np.reshape( X_train, ( X_train.shape[0], 28, 28 ) ), y_train, 20 )
```

And the test data:

```python
plotData( np.reshape( X_test, ( X_test.shape[0] , 28, 28 ) ), y_test, 20 )
```

@todo: if time, graph for number distribution


# Stage 2: Architecture

In this section different architectures are systematically explored by altering parameters such as amount of layers, layer size and activation function.

(thought: maybe merge Stage 2 and 3 and discuss the results directly under each trial...)

@question: analyze errors more thoroughly with precision and recall and look which numbers cause the errors?


### Single Layer Networks

First, single Layer Network are trained on the data, they are easy and straightforward to teach.

Note that Oja's network needs to be initialized with at least one weight which is not 0 for every Neuron, as the normalization will produce invalid values if not. Nevertheless, the weights are initialized to 0, but be aware that there is a runtime error because of that (which does not affect the results). @question: initialize in a different way?

```python
N_INPUT = 28 * 28
N_OUTPUT = 10
nTest = y_test.shape[0]
epochs = 5  # How often the Training Set is iterated over, Set lower to save significant amount of time
trials = 3  # Amount of different testRuns, set lower for maximum time saving

# Function to create empty arrays - to save space below
eA = lambda: [ [] for _ in range( len( activationFunctions ) ) ]

# Create a dictionary with all the networks and activationFunctions
oneLNacc = { 'hebb': eA(), 'deca': eA(), 'ojas': eA() }
oneLNaccPre = { 'hebb': eA(), 'deca': eA(), 'ojas': eA() } # Accuracy before training
oneLNind = { 'hebb': eA(), 'deca': eA(), 'ojas': eA() }
oneLNindPre = { 'hebb': eA(), 'deca': eA(), 'ojas': eA() } # Wrong indices before training


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
        oneLNaccPre['hebb'][i].append( hebb_pre_acc / nTest )
        oneLNindPre['hebb'][i].append( hebb_pre_iWrong )
        oneLNacc['deca'][i].append( deca_post_acc / nTest )
        oneLNind['deca'][i].append( deca_post_iWrong )
        oneLNaccPre['deca'][i].append( deca_pre_acc / nTest )
        oneLNindPre['deca'][i].append( deca_pre_iWrong )
        oneLNacc['ojas'][i].append( ojas_post_acc / nTest )
        oneLNind['ojas'][i].append( ojas_post_iWrong )
        oneLNaccPre['ojas'][i].append( ojas_pre_acc / nTest )
        oneLNindPre['ojas'][i].append( ojas_pre_iWrong )


print( "Done" )
```

Now the trials are averaged and visualized

```python
# Average the Results
# @todo: fix the thing with the threshhold function or delete it
# @todo: own function for averaging and plotting
# @todo: plot learning over epochs
for network in oneLNacc.keys():
    resList = oneLNacc[network]
    for i in range( len( resList ) ):
        resList[i] = np.average( resList[i] )
```

```python
x = np.arange( 0, len( activationFunctions ) * 2, 2 )
bWidth = 0.6

# Matplotlib preparation
plt.figure( figsize=( 10, 7 ) )
plt.title( "One Layer Networks Accuracy" )
plt.ylim( [0, 1] )

# plot the bars
plt.bar( x, [ acc for acc in oneLNacc['hebb'] ], width=bWidth )
plt.bar( x + bWidth, [ acc for acc in oneLNacc['deca'] ], width=bWidth )
plt.bar( x + bWidth * 2, [ acc for acc in oneLNacc['ojas'] ], width=bWidth )

# Set x axis
plt.xticks( x + bWidth, activationFunctionNames )

# Set legend
plt.legend( ["Hebbian", "Decay", "Oja"] );
```

The Results show that in terms of the activation function the linear activation Function seems to work best. For the networks, the plain Hebbian rule outperforms both other rules, with the Oja and Decay rule competing for the second place depending on the amount of epochs and trial runs.

Note that 0.98% of the images in the testset are digits with label '1'. Due to that and the way the evaluation works, a classification accuracy of 0.098 means the network learnt nothing, or in other words is dysfunctional. This can be currently seen with the treshhold activation function. I will fix that for the final version.


### Multi Layer Networks

Now, a two layer network is created.

Even though it seems simple, training now becomes non-trivial as it is unclear which output intermediat neurons should try to learn. There seem to be two approaches:

1. Initialize weights randomly, let the network learn just what it is producing and hope it creates enough diversity in outputs for a good categorization.
2. Force the intermediate Neurons to learn specific patterns, which arguably is not as biologically plausible anymore. A version of this was done in Amato et al. (2019), which can be seen in the chapter "3.3 Supervised Hebbian Learning".

At the moment I am not getting the first approach to work properly with an ojas based network. Probably I will include both approaches in the end.
BIG@todo: get multilayer working!

```python
N_INPUT = 28 * 28
N_L1 = 1000
N_OUTPUT = 10

h_l1 = Layer( N_INPUT, N_L1, random=True, normalize=True, learning=r_ojas )
h_out = Layer( N_L1, N_OUTPUT, learning=r_ojas, normalize=True, random=True )

twoLayer = Network()
twoLayer.setCompute( lambda x: h_out.compute( h_l1.compute( x ) ) )
twoLayer.setLearn( lambda x, y, eta: h_out.learn( ( h_l1.compulearn( x, eta=eta ) ), y, eta=eta ) )
```

```python
runPrintTest( X_test, y_test, twoLayer,"Two Layer Ojas:" );
```

```python tags=[]
twoLayer.train( X_train[:20000], y_train[:20000], epochs=5, eta=0.1 )
```

```python
print( runTest( X_test, y_test, twoLayer )[0] / X_test.shape[0] * 100 )
```

# Stage 3: Comparison

Now that everything is defined and the architectures are explored, let's discuss the results regarding following three points:

1. Which model has a higher classification accuracy?
2. Which model learns faster?
3. Interesting other effects



## Single Layer Networks

### Accuracy

In this section the clear winner is the plain simple Hebbian learning rule paired with linear or relu as activation function. Still, the plain Hebbian learning rule is quite impractical in real world usage, due to the weight explosion.

Not taking the plain Hebbian learning rule into account, the Oja learning rule performs on average better then the Hebbian Decay rule.

@todo: include code cell from above showing results (or a table or something like that)


### Learning speed

Also here, the plain Hebbian learning rule is clearly the fastest. As it is independent from the ordering in training it does not matter how often it is trained, it's best classification accuracy is reached after one Epoch already

@todo: include code cell creating a plot of accuracy dependent on Epochs


### Emerging other effects

In this category, the Oja Rule clearly wins. Due to the way it works, the weights of the Neurons resemble the first principle component of the data for the digit they are tuned to (@todo: include citation). This could be very helpful for image further analysis, for instance finding, which pixels are the most important ones.

@todo: include code cell showing the principal component for each of the numbers - "which pixel is most important for a a digit"


## Multi-Layer Networks

( this will look like the section above, likely with more text and some graphs )

### Accuracy


### Learning Speed


### Emerging other effects

<!-- #region -->
## Conclusion: How does a neural-network-classifier learning with Hebbs rule compare to a neural-network-classifier learning with Oja's rule?

Whereas the network with the plain Hebbian learning rule has quite an impressive learning rate and accuracy, it is impractical due to it's weight explosion.

The interesting part is the comparison of Oja's rule and Hebb's Decay Rule. In terms of accuracy, both rules provide for a image classification around 60%, with neither being completely dominant over the other. Furthermore, both rules highly depend on the order of the presented training data, making their learning rate quite unpredictable und not very stable over time. The key difference is, that Oja's network also gives you acces to the first principal component of the input data, making it very valuable for input data analysis.
( I can only really write more with the other results in )


@todo: relate to state of the art
<!-- #endregion -->

## Future Work

There is many interesting things to have a longer look at:

- How would a spiking neural network with the same learning rules do, compared to the models shown here.
- Is there a better and actually effective way to train the intermediate layers?
- How would a convoutional architecture perform?
- How would recurrance affect the prediction accuracy?


# References

LeCun, Y., Cortes, C., & Burges, C.J.C., The MNIST Database of Handwritten Digits \[Accessed 26.11.2020 18:00 CET\]. http://yann.lecun.com/exdb/mnist/

Amato, G., Carrara, F., Falchi, F., Gennaro, C., & Lagani, G.(2019). Hebbian Learning Meets Deep Convolutional Neural Networks. In: Ricci E., Rota Bulò S., Snoek C., Lanz O., Messelodi S., Sebe N. (eds) Image Analysis and Processing – ICIAP 2019. ICIAP 2019. Lecture Notes in Computer Science, vol 11751. Springer, Cham. https://doi.org/10.1007/978-3-030-30642-7_29

@todo: Oja's paper


# Old Code Parking Lot

Do not regard anything after this section, I am just parking old code in case I need it again - it will not be present in the final project.

```python
N_INPUT = 28 * 28
N_OUTPUT = 10

np.random.seed( 1 )

oneLNhebb = Layer( N_INPUT, N_OUTPUT )
oneLNdeca = Layer( N_INPUT, N_OUTPUT, learning=r_decay )
oneLNojas = Layer( N_INPUT, N_OUTPUT, learning=r_ojas, normalize=True )
```

```python
print( "Before Training" )
runPrintTest( X_test, y_test, oneLNhebb, "Hebb: " )
runPrintTest( X_test, y_test, oneLNdeca, "Decay:" )
runPrintTest( X_test, y_test, oneLNojas, "Oja's:" );
```

```python
# Training
print( "Hebb" )
oneLNhebb.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
print( "\nDecay" )
oneLNdeca.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
print( "\nOjas" )
oneLNojas.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
# Note: When initializing the Oja's Network there will be a runtime warning for true_divide. This is handled.
```

```python
print( "After Training" )
runPrintTest( X_test, y_test, oneLNhebb, "Hebb: " )
runPrintTest( X_test, y_test, oneLNdeca, "Decay:" )
runPrintTest( X_test, y_test, oneLNojas, "Oja's:" );
```

These resulst are not as bad already. As expected the order of training samples does not matter to the Hebbian Network, but to the Decay and Oja Network!

Now we can have a look at how the activation functions change our networks prediction.

```python
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
```

## Simple Test

To make sure the learning rules work, a very simple test.

This is a dataset from a [blog](https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2) discussing simple classifiers: [data](https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt).

I use this data, extract only the oranges and apples and then do binary classification.

The data is split into train and test. Additionally it is scaled to be between -1 and 1.

```python
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
```

```python
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
```

Create the single Neurons using the Layer Class.

```python
# Create the neurons
hebb = Layer( 4, 1, learning=r_hebb )
hebb_decay = Layer( 4, 1, learning=r_decay )
ojas = Layer( 4, 1, learning=lambda W, x, y, eta: W + eta * y * ( x - W * y ) )
```

```python
# Categorize before Training
# Expectation: All perform the same (as they have been initialized with the same weights)
print( "Hebb" )
quicktest( hebb, test_features, test_labels )
print( "Hebb Decay" )
# quicktest( hebb_decay, test_features, test_labels )
print( "Ojas" )
quicktest( ojas, test_features, test_labels )
```

```python tags=[]
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
```

```python
# Print their final weights
print( f"Hebb: { hebb.weights }" )
# print( f"Hebb_decay: { hebb_decay.weights }" )
print( f"Ojas: { ojas.weights }" )
```

```python
# Categorize after Training
# Expectation: All perform the same (as they have been initialized with the same weights)
print( "Hebb" )
quicktest( hebb, test_features, test_labels )
print( "Hebb Decay" )
# quicktest( hebb_decay, test_features, test_labels )
print( "Ojas" )
quicktest( ojas, test_features, test_labels )
```

As you can see, the classification accuracy is not that great. Multiple reasons:
- limited classification power of one single neuron + local learning rule
- data is difficult for this learning rule to learn
- not much data to learn from

BUT, it works!
Now that we have made that sure, we can go on to building the architecture of both classifiers!
