# Vanishing_Gradients_paper_study
Exploring the well-known issue of "Vanishing Gradients" in Deep Learning and the influence the activation functions and the weights initialization have over it. This was done using different NN architectures (FFNN, CNN) with different sizes and weights' initialization functions, and two different datasets (mnist, cifar10).

Inspired by the paper "_[Regularization and Reparameterization Avoid Vanishing Gradients in Sigmoid-Type Networks]([url](https://arxiv.org/abs/2106.02260#))"_ by Ven and Lederer [2021].

## Goal
The goal of this project was to replicate the results of the paper by Ven and Lederer [2021] with slight models architectural variations, as an optional graduate project with a theoretical flavour. The focus is on analyzing how gradients of the models change by varying different experimental variables and/or setups (initialization functions, activation functions, depth of the network, etc.) 

# Repository Structure
* _VanishingGradients_report.pdf_ contains a final report of the work, with both an introduction to the problem, the goal of the project and the main results.
* _Notebooks/_ contains four notebooks, to account for
  * 2 experiment setups (large initialization parameters, deep network)
  * 2 different datasets for training (mnist, cifar10)
* _environment.yml_ contains the libraries necessary to run the notebooks
  
Most of the experiments are the same. To have a clear view of the whole ordered pipeline (step-by-step), consult _mnist_LargeParameters.ipynb_: this notebook has been consistently annotated with explanations and explicit code. The other notebooks, to avoid verbosity, use the same functions but imported from _/src/*.py_ source files, however less descriptive.

