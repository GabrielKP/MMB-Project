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

AS.200.313, Models of Mind and Brain, Prof. Honey

Project draft, Gabriel Kressin

# How does a neural-network-classifier learning with a Hebbian-competetive rule compare to a neural-network-classifier learning with Oja's rule?

This Project builds two classifiers which are initialized the same way and feature the same Architecture, but differ in their learning rule.
The goal is to classify images of handwritten digits from the MNIST dataset correctly.
The first classifier uses a Hebbian learning Rule with weight-decay, the second uses Ojas Rule.
First, both Rules are defined and explained, then a little test on a small dataset shows the basic capabilities of the code and then the road to the best architecture and parameters is shown. Lastly, both classifiers will be compared.
If there is time, I will implement a spiking neural network with a similar architecture and compare it's performance to both other classifiers


## The Neuron

A neuron with the input $\mathbf{X}$ and the output $\mathbf{y}$ can be defined as

\begin{equation}
    \mathbf{y}  = f(\mathbf{wx})
\end{equation}

Whereas $\mathbf{w}$ is a vector of the weights of the input and $f$ is the so called 'activation function' - a potentially nonlinear function which produces an output from the input.

To make the computations more efficient, multiple Neurons are stacked together in a 'Layer'. In that case, multiple weight Vectors $\mathbf{w}$ are 'stacked' on top of each other creating a weight matrix $\mathbf{W}$ and the output becomes a vector of outputs.

The 'Layer' class below implements the above mentioned framework without specifying any details on activation function and how the neuron learns. The 'Layer' class takes learning rule and activation function as initializing arguments and then provides you with multiple functions:
- compute: computes the outputs of neurons in the layer
- learn: updates the weights for given samples
- getWeights: returns the weights object


```python
# Imports
import numpy as np
import pandas as pd
import matplotlib as plt
import random

class Layer():
    """
    A Layer object includes all neurons in a layer.
    It saves the weights in Matrix form.
    The weights can be updated whilst computing the results.
    """

    def __init__( self,
                    nInputs,
                    nNeurons,
                    activationFunction=( lambda x: int( x > 0 ) ),
                    learning=( lambda w, X, y, eta: w + eta * y * X ),
                    random=False ):
        """
        nInputs: amount of input neurons to layer
        nNeurons: amount of neurons in layer (==outputs)
        activationFunction: Potentially nonlinear function for the activation of the neuron: standard: 0 Threshhold
        learning: Learning Rule for Layers, standard: is simple Hebbian
        random: initialize weights randomly, standard: False
        """
        self.layerShape = ( nNeurons, nInputs )
        self.aF = activationFunction
        self.learning = learning

        if random:
            self.weights = np.random.uniform( low=-1, high=1, size=self.layerShape )
        else:
            self.weights = np.zeros( self.layerShape )
    
    def getWeights( self ):
        return self.weights

    def learn( self, X, y, eta=0.25, verbose=False ):
        """
        X: input data
        y: output data
        eta: learning rate
        """
        # assert( y.shape[0] == self.layerShape[0] )
        # assert( X.shape[0] == self.layerShape[1] )

        self.weights = self.learning( self.weights, X, y, eta )
        if verbose:
            print( self.weights )

    def compute( self, X ):
        return self.aF( np.dot( self.weights, X ) )
```

## Learning Rules

### Hebbs Rule

Hebbs Rule can be summarized as "What fires together, wires together!".

\begin{equation}      
\mathbf{\hat{y}}  = \mathbf{Wx}
\end{equation}

We train the neural network by providing it with a set of input-output
pairs, $({\bf x},{\bf y})$. Hebbian learning adjusts the weights using
the following equation for each input-output pair:

\begin{equation}         
\Delta\mathbf{W} = \eta \mathbf{y}\mathbf{x}^{T}
\end{equation} 
        
In other words, the change in the weight matrix $\mathbf{W}$ is determined by
the outer product of the output and input vectors, multiplied by the
learning rate $\eta$. Then, the updated weight matrix equals the old
weight matrix plus $\Delta\mathbf{W}$.

\begin{equation}         
\mathbf{\hat{W}} = \mathbf{W} + \Delta \mathbf{W}
\end{equation} 

### Hebbs Decay Rule

### Ojas Rule


```python
# Learning rules in Python
r_hebb = lambda W, x, y, eta: W + eta * y * x
r_hebb_decay = lambda W, x, y, eta: W + eta * y * ( x - W )
r_ojas = lambda W, x, y, eta: W + eta * y * ( x - y * W )
```

## Activation functions

### Simple Threshhold

### Sigmoid

### ReLU


## The Data

- Using MNIST Database

```python
# Define the Learning Rules
r_hebb = lambda W, x, y, eta: W + eta * y * x
r_hebb_decay = lambda W, x, y, eta: W + eta * y * ( x - W )
r_ojas = lambda W, x, y, eta: W + eta * y * ( x - y * W )
```

## The Functions

Throughout the project following functions will be used to keep the code as readable as possible.


```python
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
```

### Simple Test

To make sure the learning rules work, a very simple test.

This is a dataset from a [blog](https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2) discussing simple classifiers: [data](https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt).

I use this data, extract only the oranges and apples and then do binary classification.

```python

```

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
random.seed( 2 )
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
ojas = Layer( 4, 1, learning=r_ojas )
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


## Comparison

- Which model performs better?
- Which model learns faster?
- Interesting other effects?


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
