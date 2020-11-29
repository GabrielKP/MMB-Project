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

- x/X y/Y consistency
- Activation function section
- Learning Rule section
- Function documentation
- Network Class?
- Multilayer
- I know the dude was called Oja I will fix it in the end.


AS.200.313, Models of Mind and Brain, Prof. Honey

Project draft, Gabriel Kressin

# How does a neural-network-classifier learning with Hebbs rule compare to a neural-network-classifier learning with Oja's rule?

This project builds and compares three networks for the MNIST 10-digit classification (LeCun, Y., & Cortes, C., & Burges, C.J.C.) only differing in their learning rule:
    
    1. Plain Hebbian rule. 
    2. Hebbian-Decay rule.
    3. Oja's learning rule.

The networks are built to classify 28x28 pixel images of handwritten digits correctly to a digit from 0 to 9.

The project consists of 3 stages:

#### Stage 1
First, the network, learning rules and activation functions are explained and defined. Additionally the data is loaded in and taken a look at.

#### Stage 2
Second, the best architecture is explored by examining different combinations of hyperparameters such as amount of layers, hidden layer size and activation function. 

#### Stage 3
Lastly, the three models are compared on following three questions:
- Classification accuracy
- Learning speed
- Emerging other factors



# Stage 1

In this stage the Neuron, Network, learning Rules and activation Functions are defined and the Data is loaded in.

## The Neuron

A neuron with the input $\mathbf{x}$ and the output $\mathbf{y}$ can be defined as

\begin{equation}
    \mathbf{y}  = f(\mathbf{wx})
\end{equation}

Whereas $\mathbf{w}$ is a vector of the weights of the input and $f$ is the so called 'activation function' - a potentially nonlinear function.

To make the computations more efficient, multiple Neurons are stacked together in a 'Layer'. In that case, multiple weight Vectors $\mathbf{w}$ are 'stacked' on top of each other creating a weight matrix $\mathbf{W}$ and the output becomes a vector of outputs.

The 'Layer' class below implements the above mentioned framework without specifying any details on activation function and how the neuron learns. The 'Layer' class takes learning rule and activation function as initializing arguments and then provides you with multiple functions:
- compute: computes the outputs of neurons in the layer
- learn: updates the weights for given samples
- getWeights: returns the weights object
- train: trains the Layer on a dataset

To build a multi-layer network, for convenience the 'Network' class is defined. It is a Wrapper for multiple stacks of Layers and defines like a Layer following functions:
- compute: computes the outputs of neurons in the layer
- learn: updates the weights for given samples
- getWeights: returns the weights object
- train: trains the Layer on a dataset

Furthermore, the cell below features most of the functions used in the project.

To load the project properly you will need to have matplotlib, numpy and pandas installed.

```python
%matplotlib inline

# Imports

import gzip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

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
                    random=False ):
        """
        nInputs: amount of input neurons to layer
        nNeurons: amount of neurons in layer (==outputs)
        activationFunction: Potentially nonlinear function for the activation of the neuron: standard: 0 Threshhold
        learning: Learning Rule for Layers, standard: is simple Hebbian
        random: initialize weights randomly - will normalize weights to 1, standard: False
        """
        self.layerShape = ( nNeurons, nInputs )
        self.aF = activationFunction
        self.learning = learning

        if random:
            self.weights = normalizeRows( np.random.uniform( low=-1, high=1, size=self.layerShape ) )
        else:
            self.weights = np.zeros( self.layerShape )
    
    def getWeights( self ):
        return self.weights

    def learn( self, x, y, eta=0.25 ):
        """
        X: input data
        y: output data
        eta: learning rate
        """
        self.weights = self.learning( self.weights, x, y, eta )

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
            print( f"Epoch {x}:", end='' )
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
            print( f"Epoch {x}:", end='' )
            for i in np.random.permutation( X.shape[0] ):
                self.learn( X[i], y[i], eta )

            # Pick last 10% and compute the hit rate on them
            lindex = int( X.shape[0] * 0.9 )
            correct, _ = runTest( X[lindex:], y[lindex:], self )
            print( f"Val: { correct }/{ X[lindex:].shape[0] }" )

        # @todo: plot


# Functions

def normalizeRows( x ):
    """
    Normalizes Rows
    """
    return x / np.linalg.norm( x, axis=1 )[ :, None ]

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

def printStats( xs, topFive=False ):
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
```

```python
print( np.linalg.norm( np.arange( 0, 10, 1 ).reshape( 5, 2 ), axis=1 ) )
n = normalizeRows( np.arange( 0, 10, 1 ).reshape( 5, 2 ) )
print( n )
print( np.linalg.norm( n, axis=1 ) )
```

```python tags=["outputPrepend"]
import math
t = Layer( 900, 1, learning=r_ojas )
t.getWeights()
t.weights[0][1] = 0.5
t.weights[0][2] = 0.5 ** 0.5
t.weights[0][3] = 0.5
print( t.weights )
i = np.random.uniform( 0, 1, 900 )
print( i )
print( t.compute( i ) )
t.learn( i, t.compute( i ), eta=0.1 )
print( t.weights )
```

```python
# print( t.weights )
# i = np.array( [0.8, 1, 0.3,0.5] )
# print( i )
print( t.compute( i ) )
t.learn( i, t.compute( i ), eta=0.1 )
print( t.weights )

```

```python
t.compute( i )
```

## Learning Rules

### Hebbs Rule

Hebbs Rule can be summarized as "What fires together, wires together".
The weights $\mathbf{W}$ are updated according to the given input, if the neuron was supposed to be activated. In other words, given a pair $(\mathbf{x}, \mathbf{y})$ the updated weights $\mathbf{\hat{W}}$ are computed:

\begin{equation}         
\mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}\mathbf{x}^{T}
\end{equation} 

### Hebbs Decay Rule

\begin{equation}         
\mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}( \mathbf{x} - \mathbf{W} )
\end{equation}

### Ojas Rule

\begin{equation}         
\mathbf{\hat{W}} = \mathbf{W} + \eta \mathbf{y}( \mathbf{x} - \mathbf{y} \mathbf{W} )
\end{equation}

### Rules in Python

Finally, let's implement the rules in Python:


```python
# Learning rules in Python
r_hebb = lambda W, x, y, eta: W + eta * np.outer( y, x.T )
r_hebb_decay = lambda W, x, y, eta: W + eta * y * ( x - W )
r_ojas = lambda W, x, y, eta: W + eta * ( ( x - W * y[ :, None ] ) * y[ :, None ] )
# Nonlinearity needs to be taken into account!
```

```python
w = np.arange( 1, 22, 1 ).reshape( 7, 3 )
print( w )
t = np.array( [ 0, 0, 0, 1, 0, 0, 0 ] )
# print( t[:, None] )
# print( w * t[:, None] )
x = np.array( [ 2, 3, 4 ] )
print( x )
print( x - w )
```

## Activation functions

### Linear

### Simple Threshhold

### Sigmoid

### ReLU

```python
# Define the Activation functions, need to work with vector inputs
linear = lambda x: x
threshhold = lambda x: ( x > 0 ).astype( np.int )
sigmoid = lambda x: ( 1 / ( 1 + np.exp( -x ) ) )
relu = lambda x: np.maximum( 0, x )
```

Visualizing the activation functions:

```python
test = np.linspace( -2, 2, 300 )
plt.plot( test, linear( test ) )
plt.plot( test, threshhold( test ) )
plt.plot( test, sigmoid( test ) )
plt.plot( test, relu( test ) )
plt.legend( ["Linear", "Threshhold", "Sigmoid", "ReLU"] )
```

## The Data

The MNIST Database provides 60.000 training examples and 10.000 test examples without needing to preprocess or format them.

First, let's load the data in, there is 2 things to keep in mind:
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

```python
print( X_train[0] )
```

We can have a look at the train data:

```python
plotData( np.reshape( X_train, ( X_train.shape[0], 28, 28 ) ), y_train, 20 )
```

And the test data:

```python
plotData( np.reshape( X_test, ( X_test.shape[0] , 28, 28 ) ), y_test, 20 )
```

## Simple Test

To make sure the learning rules work, a very simple test.

This is a dataset from a [blog](https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2) discussing simple classifiers: [data](https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt).

I use this data, extract only the oranges and apples and then do binary classification.


The data is split into train and test. Additionally it is scaled to be between -1 and 1.

```python
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
hebb_decay = Layer( 4, 1, learning=r_hebb_decay )
ojas = Layer( 4, 1, learning=lambda W, x, y, eta: W + eta * y * ( x - W * y ) )
```

```python
# Categorize before Training
# Expectation: All perform the same (as they have been initialized with the same weights)
print( "Hebb" )
quicktest( hebb, test_features, test_labels )
print( "Hebb Decay" )
quicktest( hebb_decay, test_features, test_labels )
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
        hebb_decay.learn( train_features[ireal], train_labels[ireal], eta=0.2 )
        ojas.learn( train_features[ireal], train_labels[ireal], eta=0.2 )
```

```python
# Print their final weights
print( f"Hebb: { hebb.weights }" )
print( f"Hebb_decay: { hebb_decay.weights }" )
print( f"Ojas: { ojas.weights }" )
```

```python
# Categorize after Training
# Expectation: All perform the same (as they have been initialized with the same weights)
print( "Hebb" )
quicktest( hebb, test_features, test_labels )
print( "Hebb Decay" )
quicktest( hebb_decay, test_features, test_labels )
print( "Ojas" )
quicktest( ojas, test_features, test_labels )
```

As you can see, the classification accuracy is not that great. Multiple reasons:
- limited classification power of one single neuron + local learning rule
- data is difficult for this learning rule to learn
- not much data to learn from

BUT, it works!
Now that we have made that sure, we can go on to building the architecture of both classifiers!


## The Architecture

In this section we systematically explore which combination of architecture, activation function and learning rate works the best


### Single Layer Networks

First, I try learning single Layer Networks, they are easy and straightforward to teach.

```python
N_INPUT = 28 * 28
N_OUTPUT = 10

np.random.seed( 1 )

oneLNhebb = Layer( N_INPUT, N_OUTPUT )
oneLNojas = Layer( N_INPUT, N_OUTPUT, learning=r_ojas, random=True )
```

```python
oneLNojas.learn( X_train[1], y_train[1], eta=0.1 )
oneLNojas.getWeights()
```

```python
# Before Training
correct, iWrong = runTest( X_test, y_test, oneLNhebb )
print( f"{correct}/{y_test.shape[0]} correct: { correct/y_test.shape[0] * 100 } %" )
correct, iWrong = runTest( X_test, y_test, oneLNojas )
print( f"Ojas: {correct}/{y_test.shape[0]} correct: { correct/y_test.shape[0] * 100 } %" )
```

```python
# Training
print( "Hebb" )
oneLNhebb.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
print( "Ojas" )
oneLNojas.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
```

```python
# After Training
correct, iWrong = runTest( X_test, y_test, oneLNhebb )
print( f"{correct}/{y_test.shape[0]} correct: { correct/y_test.shape[0] * 100 } %" )
correct, iWrong = runTest( X_test, y_test, oneLNojas )
print( f"Ojas: {correct}/{y_test.shape[0]} correct: { correct/y_test.shape[0] * 100 } %" )
```

We can see some not-so-bad results here already. As expected the order of training samples does not matter to the Hebbian Network, but to the Ojas Network!

Now we can have a look at how the activation functions change our networks prediction.

```python
N_INPUT = 28 * 28
N_OUTPUT = 10

oneLNhebb = Layer( N_INPUT, N_OUTPUT, activationFunction=relu )
oneLNojas = Layer( N_INPUT, N_OUTPUT, learning=r_ojas, activationFunction=relu )

np.random.seed( 1 )

# Before Training
print( "Before Training")
correct, iWrong = runTest( X_test, y_test, oneLNhebb )
print( f"Hebb: {correct}/{y_test.shape[0]} correct: { correct/y_test.shape[0] * 100 } %" )
correct, iWrong = runTest( X_test, y_test, oneLNojas )
print( f"Ojas: {correct}/{y_test.shape[0]} correct: { correct/y_test.shape[0] * 100 } %" )

# Training
print( "Training" )
print( "Hebb" )
oneLNhebb.train( X_train, y_train, epochs=5, eta=0.1, seed=None )
print( "Ojas")
oneLNojas.train( X_train, y_train, epochs=5, eta=0.1, seed=None )

# After Training
print( "After Training" )
correct, iWrong = runTest( X_test, y_test, oneLNhebb )
print( f"Hebb: {correct}/{y_test.shape[0]} correct: { correct/y_test.shape[0] * 100 } %" )
correct, iWrong = runTest( X_test, y_test, oneLNojas )
print( f"Ojas: {correct}/{y_test.shape[0]} correct: { correct/y_test.shape[0] * 100 } %" )
```

```python
oneLNojas.compute( X_test[0] )
```

### Multi Layer Networks

Training becomes non-trivial.
Which outputs should intermediate neurons learn?

3 Approaches:
- Create enough random diversity until it works
- Force them to learn certain patterns ( would not be biologically plausible )
- Create random diversity, and if they have very low weights to the neurons in the next layer, they get a reset on the weights

```python
N_INPUT = 28 * 28
N_L1 = 100
N_OUTPUT = 10

h_l1 = Layer( N_INPUT, N_L1, random=True, learning=r_ojas, activationFunction=sigmoid )
h_out = Layer( N_L1, N_OUTPUT, learning=r_ojas, activationFunction=sigmoid, random=True )

twoLayer = Network()
twoLayer.setCompute( lambda x: h_out.compute( h_l1.compute( x ) ) )
twoLayer.setLearn( lambda x, y, eta: h_out.learn( ( h_l1.compulearn( x, eta=eta ) ), y, eta=eta ) )
```

```python
h_l1.weights
```

```python
h_l1.learn( X_train[0], h_l1.compute( X_train[0] ) )
h_l1.getWeights()
```

```python
print( runTest( X_test, y_test, twoLayer )[0] / X_test.shape[0] * 100 )
```

```python tags=[]
twoLayer.train( X_train, y_train, epochs=1, eta=0.1 )
```

```python
print( runTest( X_test, y_test, twoLayer )[0] / X_test.shape[0] * 100 )
```

## Comparison

- Which model performs better?
- Which model learns faster?
- Interesting other effects?



## References

LeCun, Y., & Cortes, C., & Burges, C.J.C., The MNIST Database of Handwritten Digits \[Accessed 26.11.2020 18:00 CET\]. http://yann.lecun.com/exdb/mnist/


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
