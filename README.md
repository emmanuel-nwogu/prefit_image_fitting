# Pretraining for image fitting

A while ago, I read [Implicit Neural Representations for Image Compression](https://arxiv.org/abs/2112.04267), and I was
very drawn to the idea of learning some rich initialization for image fitting neural networks. After forking the
paper's [implementation](https://github.com/YannickStruempler/inr_based_compression) and playing around with the
different compression stages, I found the meta-learning stage to be very compute-hungry and time-consuming for my
laptop. This got me curious about whether simply using a model already fit to another image would prove to be
beneficial. This method is cheaper and faster, but I don't expect it to be better than meta-learning; I don't even try
to compare them (mostly due to training costs).

I tried to find papers that tackle this problem, and I couldn't. I didn't try too hard though since I was more
interested in just doing a project myself. If you know of any such papers, do let me know :)

## Problem Statement

Given an image _A_ to be fit using a neural network, which of the following fitting modes would be more efficient:

- randomly initialise the neural network, then fit image _A_
- fit the neural network on an image _B_ from the same class as _A_, then fit image _A_
- fit the neural network on an image _C_ from a random different class as _A_, then fit image _A_

Note that images _A_, _B_, and _C_ are distinct. Also, "fitting" an image means predicting its pixel values from its
coordinates. This could be represented by `f: (px, py) → (r, g, b)` where px, py are the pixel coordinates, (r, g, b)
the RGB color channels, and f the neural network.

## Experiments

Our image dataset is the deduplicated version of the famous [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html):
[ciFAIR-10](https://cvjena.github.io/cifair/). Where a "triple" represents three images (A, B, C) as defined in the
problem statement, we randomly select n=20 triples from CiFAIR-10 such that all images across all triples are distinct.
Then, we conduct all three fitting schemes of concern for each triple and record the peak signal-to-noise ratio (PSNR)
every epoch throughout the actual fitting on image A. Fitting, regardless of the image being fit, is done for 2000
epochs using mean-squared loss (MSE) and Adam optimization. We use sinusoidal representation networks (SIREN) as our
neural network and vary the number of hidden layers and hidden features to investigate scaling: `hls` x `hfs` where
hidden layers, `hls` = [2, 3, 4, 5, 6, 7, 8] and hidden features, `hfs` = [32, 64, 128, 256, 512, 1024].

[//]: # (TODO: Add colab link)

## Results

Here are a few PSNR vs Epochs graphs from the experiments. The solid lines represent the mean PSNR of 20 triple runs
while the shaded area represents the standard deviation.

![Alt text](plots/3hls_128hfs.png?raw=true "3 hidden layers, 128 hidden features")

![Alt text](plots/5hls_32hfs.png?raw=true "3 hidden layers, 128 hidden features")

![Alt text](plots/6hls_64hfs.png?raw=true "3 hidden layers, 128 hidden features")

![Alt text](plots/7hls_512hfs.png?raw=true "3 hidden layers, 128 hidden features")

![Alt text](plots/8hls_256hfs.png?raw=true "3 hidden layers, 128 hidden features")

There are 42 graphs in total and they are located in the [plots](plots) directory.

To get a big picture of the effects of the different fitting schemes as we scale model size, we calculate and plot the
Area Under the Curve (AUC) for the PSNR vs Epochs curves from the 42 experiments. A higher AUC points to a more stable
and faster-converging training run. As seen in the box plot below, we sort the x-axis by the product of the hidden
layers and hidden features in ascending order.
![Alt text](auc.png?raw=true "AUCs graph")

My main takeaway from these experiments is that pre-fitting on a different image, especially one from a different class,
before fitting a target image may be very useful for smaller models. On the other hand, this advantage quickly turns to
a strong disadvantage for larger models. However, since CiFAIR images are 32 x 32 and SIRENS are notoriously difficult
to train without a meticulous initialisation strategy [1], there may be more to it than pre-fitting being hurtful. If
you have any ideas. please let me know :)

[1] https://arxiv.org/abs/2006.09661
