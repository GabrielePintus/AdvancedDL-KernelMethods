# Topics

This document contains the list of topics that, in my opinion, are important for the course. In addition to the professor's slides, I have included some references that I found useful.

## Prerequisites

- Group [Wikipedia](https://en.wikipedia.org/wiki/Group_(mathematics))
- Lagrange multipliers [Wikipedia](https://en.wikipedia.org/wiki/Lagrange_multiplier)
- Hilbert space [Wikipedia](https://en.wikipedia.org/wiki/Hilbert_space)
- Spectral theorem [Wikipedia](https://en.wikipedia.org/wiki/Spectral_theorem)
- Eigendecomposition [Wikipedia](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)
- Singular value decomposition [Wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- Convolution [Wikipedia](https://en.wikipedia.org/wiki/Convolution)
- Pooling [Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer), [Review paper](References/Pooling%20Methods%20in%20Deep%20Neural%20Networks,%20a%20Review.pdf)
- Fourier transform [Wikipedia](https://en.wikipedia.org/wiki/Fourier_transform)
- Wavelet transform [Wikipedia](https://en.wikipedia.org/wiki/Wavelet_transform)
- Curse of dimensionality [Wikipedia](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

## Preliminaries

- Linear separability [Wikipedia](https://en.wikipedia.org/wiki/Linear_separability)
- Perceptron
- Maximum margin classifier
- Support vector machine
- Cover's theorem [Wikipedia](https://en.wikipedia.org/wiki/Cover%27s_theorem), [NYU Notes](References/Cover’s%20Function%20Counting%20Theorem%20(1965).pdf)

## Kernel methods

The outline of this section is taken from the course "Machine Learning
with Kernel Methods" of J. Mairal and J.P. Vert. [slides](References/Machine%20Learning%20with%20Kernel%20Methods.pdf).

### Kernel functions and RKHS
- Positive definite kernel, *page 18*. [Wikipedia](https://en.wikipedia.org/wiki/Positive-definite_kernel).
- Aronszajn’s theorem. [Wikipedia](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space#Aronszajn's_theorem), [JP Vert's notes](References/Aronszajn’s%20theorem.pdf).
- Reproducing kernel Hilbert space (RKHS), *page 25*. [Wikipedia](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space).
- Mercer's theorem, *page 232*. [Wikipedia](https://en.wikipedia.org/wiki/Mercer%27s_theorem).
- Popular kernels
- Smoothness functional, *page 54*.

### Kernel trick
- Kernel trick, *page 63*. [Wikipedia](https://en.wikipedia.org/wiki/Kernel_method)
- Representer theorem, *page 76*. [Wikipedia](https://en.wikipedia.org/wiki/Representer_theorem)


### Kernel methods

- Kernel methods [Wikipedia](https://en.wikipedia.org/wiki/Kernel_method)

#### Unsupervised learning

- Kernel PCA, *page 172*. [Wikipedia](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
- Kernel k-means, *page 183*. [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering#Kernel_k-means), [Paper](References/Kernel%20k-Means,%20By%20All%20Means%20-%20Algorithms%20and%20Strong%20Consistency.pdf)

#### Supervised learning

- Kernel ridge regression. [M Welling's notes](References/Kernel%20ridge%20Regression.pdf)
- Kernel logistic regression
- Large margin classifiers
- Kernel SVM

### Further topics
- Kernel Jungle, *page 203*.
- Multiple Kernel Learning, *page 209*. [Wikipedia](https://en.wikipedia.org/wiki/Multiple_kernel_learning)
- Deep kernel machines, *page 645*.
- Neural Tangent Kernel, *page 649*. [Wikipedia](https://en.wikipedia.org/wiki/Neural_tangent_kernel), [Paper](References/Neural%20Tangent%20Kernel%20-%20Convergence%20and%20Generalization%20in%20Neural%20Networks.pdf)



## Neural networks

- Shallow vs deep networks
- Large Width vs Large Depth
- Double descent phenomenon. [Wikipedia](https://en.wikipedia.org/wiki/Double_descent)
- Data manifold hypothesis. [Wikipedia](https://en.wikipedia.org/wiki/Manifold_hypothesis)

### Regularization

#### Explicit regularization

- Norm-based regularization


#### Implicit regularization

- Explicit vs implicit regularization
- Neural Tangent Kernel, *page 649*. [Wikipedia](https://en.wikipedia.org/wiki/Neural_tangent_kernel), [Paper](References/Neural%20Tangent%20Kernel%20-%20Convergence%20and%20Generalization%20in%20Neural%20Networks.pdf)
- Early stopping. [Wikipedia](https://en.wikipedia.org/wiki/Early_stopping), [Paper](References/On%20Regularization%20via%20Early%20Stopping%20for%20Least%20Squares%20Regression.pdf)
- Gradient descent. [Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent), [Paper](References/Implicit%20Gradient%20Regularization.pdf)
- Stochastic gradient descent. [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), [Paper](References/On%20the%20Origin%20of%20Implicit%20Regularization%20in%20Stochastic%20Gradient%20Descent.pdf)
- Noisy inputs. [Paper](References/Training%20with%20Noise%20is%20Equivalent%20to%20Tikhonov%20Regularization.pdf)
- Dropout. [Paper](References/Improving%20neural%20networks%20by%20preventing%20co-adaptation%20of%20feature%20detectors.pdf)
- Matrix completion via convex optimization. [Paper](References/Exact%20Matrix%20Completion%20via%20Convex%20Optimization.pdf)
- Weights initialization. [Paper](References/On%20weight%20initialization%20in%20deep%20neural%20networks.pdf)
- Xavier initialization. [Paper](References/Understanding%20the%20difficulty%20of%20training%20deep%20feedforward%20neural%20networks.pdf)
- Batch normalization. [Paper](References/Towards%20understanding%20regularization%20in%20Batch%20Normalization.pdf)


#### Implicit bias

- Gradient descent on Linear Convolutional Networks [Paper](References/Implicit%20Bias%20of%20Gradient%20Descent%20on%20Linear%20Convolutional%20Networks.pdf)
- Data augmentation and symmetries [Paper](References/Data%20Symmetries%20and%20Learning%20in%20Fully%20Connected%20Neural%20Networks.pdf)
- Generalization of Equivariance and Convolution in Neural Networks
to the Action of Compact Groups. [Paper](References/On%20the%20Generalization%20of%20Equivariance%20and%20Convolution%20in%20Neural%20Networks%20to%20the%20Action%20of%20Compact%20Groups.pdf)
- Permuation equivariant layer. [Paper](References/Deep%20Sets.pdf)
- Data Augmentation. [Wikipedia](https://en.wikipedia.org/wiki/Data_augmentation).
- Compositionality prior and weight sharing
- Fourier important features
- Softmax function. [Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)
- Exponential tilting. [Wikipedia](https://en.wikipedia.org/wiki/Exponential_tilting)

### Other topics

- Generative adversarial networks. 
- Networks:
    - Perceptrons: function regularity
    - CNNs: Translation invariance
    - Group CNNs: Translation+Rotation invariance, global groups
    - LSTMs: Time warping invariance
    - DeepSets/Transformer: Permutation invariance
    - Graph CNNs: Permutation invariance


## Biologically inspired learning

- Interplay between neuroscience and machine learning. [Paper](References/Natural%20and%20Artificial%20Intelligence%20-%20A%20brief%20introduction%20to%20the%20interplay%20between%20AI%20and%20neuroscience%20research.pdf)
- Invariance and selectivity


## Geometric Deep Learning

One important direction is the understanding of the geometry of the data and the design of meaningful priors to decrease the sample complexity of the learning.


## Advarsarial Attacks


