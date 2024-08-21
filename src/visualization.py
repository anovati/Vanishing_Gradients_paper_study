import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

def plot_accuracy_loss(train_accuracy, val_accuracy, train_loss, val_loss):
    """Plot the training/validation accuracy and loss"""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(15,5))
    ax_acc.xlabel = 'Epoch'
    ax_acc.ylabel = 'Accuracy'
    ax_acc.set_title("Training Accuracy vs Validation Accuracy")
    ax_acc.plot(np.arange(len(train_accuracy)), train_accuracy, label='Training Accuracy')
    ax_acc.plot(np.arange(len(train_accuracy)), val_accuracy, label='Validation Accuracy')  
    ax_acc.legend()
    
    ax_loss.xlabel = 'Epoch'
    ax_loss.ylabel = 'Loss'
    ax_loss.set_title("Training Loss vs Validation Loss")
    ax_loss.plot(np.arange(len(train_loss)), train_loss, label='Training Loss')
    ax_loss.plot(np.arange(len(train_loss)), val_loss, label='Validation Loss')  
    ax_loss.legend()
    
    plt.show()

            
    
    
def plot_gradient(gradhistory):
    """Plot gradient mean and sd across epochs"""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    ax[0].set_title("Mean gradient")
    for key in gradhistory[0]:
        ax[0].plot(range(len(gradhistory)), [w[key].mean() for w in gradhistory], label=key)
    ax[0].legend()
    
    ax[1].set_title("S.D.")
    for key in gradhistory[0]:
        ax[1].semilogy(range(len(gradhistory)), [w[key].std() for w in gradhistory], label=key)
    ax[1].legend()
    
    
    
def plot_gradient_first_last(gradhistory, paper=False, deep=False):
    """Plot the distribution of the first layer's and last layer's gradients
    
    Parameters
    ----------
    gradhistory : np.ndarray
        array with gradients' records.
    paper : bool
        change naming of the first layer in the plot to match the type of layers use in the paper model
    deep  : bool
        change naming of last layer in the plot to match the depth of the model
    """
    first_layer_name = 'Convo1' if paper else 'Layer1'
    
    last_layer_name  = ''
    if paper and deep:
        last_layer_name = 'Layer5'
    elif paper and not deep:
        last_layer_name = 'Layer2'
    elif deep and not paper:
        last_layer_name = 'Layer20'
    else:
        last_layer_name = 'Layer2'


    mu1 = '$\mu_{first}$ = '
    mu2 = '$\mu_{last}$ = '
    sigma1 = '$\sigma_{first}$ = '
    sigma2 = '$\sigma_{last}$ = '
    
    mean1 = str(np.mean(gradhistory[-1][first_layer_name]))
    stdev1 = str(np.std(gradhistory[-1][first_layer_name]))
    mean2 = str(np.mean(gradhistory[-1][last_layer_name]))
    stdev2 = str(np.std(gradhistory[-1][last_layer_name]))
    
    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    
    plt.subplot(1,3,1)

    ax = sns.distplot(gradhistory[-1][first_layer_name], color = 'brown', axlabel = 'Gradients', label = 'First Layer Gradients')
    sns.distplot(gradhistory[-1][last_layer_name], color = 'blue', axlabel = 'Gradients', label = 'Last Layer Gradients')
    ax.set_title("First&Last Layer Gradients")
    plt.text(0.9, 0.95, mu1+mean1, horizontalalignment='right', size='medium', color='black', transform = ax.transAxes)
    plt.text(0.9, 0.90, mu2+mean2, horizontalalignment='right', size='medium', color='black', transform = ax.transAxes)
    plt.text(0.9, 0.85, sigma1+stdev1, horizontalalignment='right', size='medium', color='black', transform = ax.transAxes)
    plt.text(0.9, 0.80, sigma2+stdev2, horizontalalignment='right', size='medium', color='black', transform = ax.transAxes)
    ax.legend(loc = "upper left")
    
    plt.subplot(1,3,2)
    ax = sns.distplot(gradhistory[-1][first_layer_name], color = 'brown', axlabel = 'Gradients', label = 'First Layer Gradients')    
    ax.set_title("Last Layer Gradients")
    plt.text(0.9, 0.95, mu1+mean1, horizontalalignment='right', size='medium', color='black', transform = ax.transAxes)
    plt.text(0.9, 0.90, sigma1+stdev1, horizontalalignment='right', size='medium', color='black', transform = ax.transAxes)
    ax.legend(loc = "upper left")
    
    plt.subplot(1,3,3)
    ax = sns.distplot(gradhistory[-1][last_layer_name], color = 'blue', axlabel = 'Gradients', label = 'Last Layer Gradients')
    ax.set_title("Last Layer Gradients")
    plt.text(0.9, 0.95, mu2+mean2, horizontalalignment='right', size='medium', color='black', transform = ax.transAxes)
    plt.text(0.9, 0.90, sigma2+stdev2, horizontalalignment='right', size='medium', color='black', transform = ax.transAxes)
    ax.legend(loc = "upper left")