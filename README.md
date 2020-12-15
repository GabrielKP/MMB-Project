AS.200.313, Models of Mind and Brain, Prof. Honey

Final Project, Gabriel Kressin Palacios

# How do Neural Networks learning with the plain Hebb rule, Hebb-Decay rule and Oja rule respectively compare to each other regarding accuracy, learning speed and other features in a digit classification task?

This project builds and compares three networks featuring biologically plausible learning rules to classify digits from the MNIST 10-digit dataset (LeCun, Y., & Cortes, C., & Burges, C.J.C.). How do these networks compare to each other regarding accuracy, learning speed and other features?

Neural Networks are very powerful machine learning tools, which currently are one of the most successfull models at predicting human brain activity (Yamins et al., 2014), whilst also achieving human-like performance and performance patterns in specific tasks (Serre, Oliva, & Poggio, 2007). Despite being biologically inspired, the most powerful methods to train Neural Networks rely on "backpropagation" - a process which is difficult to imagine taking place in the human brain (although that seems to be under dispute currently). Besides backpropagation, other learning rules exist which are insipired by biological processes such as Long Term Potentiation and thus seem to be more biologically plausible. This project builds three networks which only differ in those rules and compares them based on a 10-way 28x28 handwritten digit classification task. Following rules are used:

    1. Plain Hebbian rule.
    2. Hebbian-Decay rule.
    3. Oja's learning rule.

How do those learning rules compare to each other in classification accuracy? What could be one of the main differences making one effective and the other not? Furthermore, biological systems tend to be very efficient and effective in their learning capabilities, thus learning speed is taken into account into the comparison. Additionally, other emerging factors of the learning rules could lead to interesting properties in a biological system.

The project consists of 3 stages:

#### Stage 1: Definition
First, the network and learning rules are explained and defined. Additionally the data is loaded in and taken a look at.

#### Stage 2: Training & Exploration
Second, the networks are trained on the data and results are plotted. Based on the results additional investigations into learning speed and specific effects are made.

#### Stage 3: Conclusion
Finally, a conclusion is drawn based on the results and following three criteria:
- Classification accuracy
- Learning speed
- Emerging other factors

### Dependencis
- python 3.8.6
- matplotlib
- numpy
