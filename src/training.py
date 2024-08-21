import tensorflow as tf
import numpy as np
from time import time

def train_model(X_train, y_train, X_val, y_val, model, n_epochs, optimizer, loss, metric, batch_size):
    """
    Trains a TensorFlow model using a manual training loop, with support for gradient recording, loss tracking, 
    and accuracy evaluation on both training and validation datasets.   
    Parameters
    ----------
    X_train : tf.Tensor
        The input training data.
    y_train : tf.Tensor
        The labels for the training data.
    X_val : tf.Tensor
        The input validation data.
    y_val : tf.Tensor
        The labels for the validation data.
    model : tf.keras.Model
        The TensorFlow Keras model to be trained.
    n_epochs : int
        The number of epochs to train the model.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use for training.
    loss : tf.keras.losses.Loss
        The loss function to use during training.
    metric : tf.keras.metrics.Metric
        The metric to evaluate model performance (e.g., accuracy).
    batch_size : int
        The size of the batches used during training and validation.    
    Returns
    -------
    train_gradhistory : list of dict
        A history of gradients for each layer, recorded at the start and end of the training.
    train_losshistory : list of float
        A history of training loss values recorded at the start and end of the training.
    train_acchistory : list of float
        A history of training accuracy values recorded at the end of each epoch.
    val_losshistory : list of float
        A history of validation loss values recorded at the end of each epoch.
    val_acchistory : list of float
        A history of validation accuracy values recorded at the end of each epoch.
    timing : float
        The total time taken to complete the training process.  
    """  

    
    #Define a secondary function to record weights
    def record_train(): 
        data = {}
        for g,w in zip(grads, model.trainable_weights):
            if '/kernel:' not in w.name:
                continue # skip bias
            name = w.name.split("/")[0]
            data[name] = g.numpy()
        train_gradhistory.append(data)
        train_losshistory.append(train_loss_value.numpy())
        
        
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            train_loss_value = loss(y, y_pred)
            train_loss_value += sum(model.losses)
        grads = tape.gradient(train_loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        metric.update_state(y, y_pred)
        return train_loss_value, grads
    

    @tf.function
    def val_step(x, y):
        y_pred = model(x, training=False)
        val_loss_value = loss(y, y_pred)
        metric.update_state(y, y_pred)
        return val_loss_value
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) #Transform tensor (X,y) into a dataset made up of many rows (image + label)
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)  #Shuffle dataset and batch consecutive rows together
    
    #Do the same thing for the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) #Transform tensor (X,y) into a dataset made up of many rows (image + label)
    val_dataset = val_dataset.shuffle(buffer_size=1000).batch(batch_size) #Shuffle dataset and batch consecutive rows together
    
    
    train_gradhistory = []
    train_losshistory = []
    train_acchistory  = []
    
    val_losshistory = []
    val_acchistory = []
    
    start_time = time()
    for epoch in range(n_epochs):
        
        #Some logging
        print("\nStart of epoch %d" % (epoch,))
        
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset): #For each step, corresponding to a single batc
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            train_loss_value, grads = train_step(x_batch_train, y_batch_train)
            
            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(train_loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
#
            if step == 0:
                record_train() #Record gradient, loss at the start of each epoch
    
        print("\nEnd of epoch %d" % (epoch,))
        
        # Display metrics at the end of each epoch.
        train_acc = metric.result()
        train_acchistory.append(train_acc)
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        metric.reset_states() #Reset the metric value at the start of the epoch
        
        ##############VALIDATION EPOCH#############################
        print("\nStarting Validation of epoch %d" % (epoch,))
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = val_step(x_batch_val, y_batch_val)
            
            if step == 0:
                val_losshistory.append(val_loss_value.numpy())
                
        val_acc = metric.result()
        val_acchistory.append(val_acc)
        metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        
    # After all epochs, record again
    record_train()
    val_losshistory.append(val_loss_value.numpy())
    
    end_time = time()
    timing = end_time - start_time    
        
    
    return train_gradhistory, train_losshistory, train_acchistory, val_losshistory, val_acchistory, timing