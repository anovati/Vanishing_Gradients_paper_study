import tensorflow as tf
import numpy as np

# RECREATION OF NEURAL NETWORK USED IN THE PAPER
def paper_net(input_shape, issue, activation, predicter, initializer, name, regularization_type=None, regularization_value=0.0):
    """
    Builds a convolutional neural network model based on the provided configuration. 

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data.
    issue : str
        Specifies the configuration type: 'parameters' for parameter-heavy layers or 'layers' for layer-heavy configuration.
    activation : str
        The activation function to use for the convolutional and dense layers (e.g., 'relu', 'tanh').
    predicter : str
        The activation function to use for the output layer.
    initializer : str
        The initializer to use for kernel and bias (e.g., 'he_normal', 'glorot_uniform').
    name : str
        The base name for the layers in the model.
    regularization_type : str, optional
        The type of regularization to apply ('l1', 'l2', etc.). Defaults to None.
    regularization_value : float, optional
        The regularization value to apply if regularization is used. Defaults to 0.0.

    Returns
    -------
    tf.keras.Model
        A TensorFlow Keras Sequential model configured based on the provided parameters.

    """
    regularizer = get_regularizer(regularization_type, regularization_value)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_shape, name = 'Input')) 

    if issue == 'parameters':
        for i in range(1, 7):
          model.add(tf.keras.layers.Convolution2D(
              36, (3, 3), activation=activation, kernel_initializer=initializer,
              kernel_regularizer=regularizer, bias_initializer=initializer,
              bias_regularizer=regularizer, name=f'Convo{i}', padding='same'
              ))
          if i % 2 == 0:
              model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f'Pool{i//2}'))

        model.add(tf.keras.layers.Flatten(name = name+str(0))) #From here on is FFNN    
        model.add(create_dense_layer(8, activation, initializer, regularizer, f"{name}1"))
        model.add(create_dense_layer(10, predicter, initializer, regularizer, f"{name}2"))

    if issue == 'layers':
        for i in range(1, 7):
           model.add(tf.keras.layers.Convolution2D(
               3, (3, 3), activation=activation, kernel_initializer=initializer,
               kernel_regularizer=regularizer, bias_initializer=initializer,
               bias_regularizer=regularizer, name=f'Convo{i}', padding='same'
               ))
           if i % 3 == 0:
             model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f'Pool{i//2}'))
        
        model.add(tf.keras.layers.Flatten(name = name+str(0))) #Da qui è una FFNN
        
        for i in range(4):
            model.add(create_dense_layer(8, activation, initializer, regularizer, f"{name}{i+1}"))
        model.add(create_dense_layer(10, activation=predicter, initializer=initializer, regularizer=regularizer, name=f"{name}5"))
    
    return model



# CREATING MY MLP
def make_MLPnet(input_shape, activation, predicter, initializer, layers_name, n_hidden=1, units=1, regularization_type=None, regularization_value=0.0):
    """
    Builds a Multi-Layer Perceptron (MLP) model based on the provided configuration.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data.
    activation : str
        The activation function to use for the hidden layers (e.g., 'relu', 'tanh').
    predicter : str
        The activation function to use for the output layer.
    initializer : str
        The initializer to use for kernel and bias (e.g., 'he_normal', 'glorot_uniform').
    layers_name : str
        The base name for the layers in the model.
    n_hidden : int, optional
        The number of hidden layers in the model. Defaults to 1.
    units : int, optional
        The number of units in each hidden layer. Defaults to 1.
    regularization_type : str, optional
        The type of regularization to apply ('l1', 'l2', etc.). Defaults to None.
    regularization_value : float, optional
        The regularization value to apply if regularization is used. Defaults to 0.0.

    Returns
    -------
    tf.keras.Model
        A TensorFlow Keras Sequential model configured as a MLP based on the provided parameters.

    """    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(input_shape, name='Input'))
    model.add(tf.keras.layers.Flatten(name=layers_name))

    
    regularizer = get_regularizer(regularization_type, regularization_value)
    for i in range(n_hidden):
        model.add(create_dense_layer(units, activation, initializer, regularizer, f"{layers_name}{i+1}"))
    
    model.add(create_dense_layer(10, predicter, initializer, regularizer, f"{layers_name}{n_hidden+1}"))

    return model

    
# CREATING MY CNN + MLP
def make_CNNnet(input_shape, activation, predicter, initializer, layers_name, n_hidden=1, units=1, regularization_type=None, regularization_value=0.0):
    """
    Builds a Convolutional Neural Network (CNN) model followed by a Multi-Layer Perceptron (MLP) based on the provided configuration.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data.
    activation : str
        The activation function to use for the convolutional and hidden layers (e.g., 'relu', 'tanh').
    predicter : str
        The activation function to use for the output layer.
    initializer : str
        The initializer to use for kernel and bias (e.g., 'he_normal', 'glorot_uniform').
    layers_name : str
        The base name for the layers in the model.
    n_hidden : int, optional
        The number of hidden layers after the convolutional layers. Defaults to 1.
    units : int, optional
        The number of units in each hidden layer. Defaults to 1.
    regularization_type : str, optional
        The type of regularization to apply ('l1', 'l2', etc.). Defaults to None.
    regularization_value : float, optional
        The regularization value to apply if regularization is used. Defaults to 0.0.

    Returns
    -------
    tf.keras.Model
        A TensorFlow Keras Sequential model configured as a CNN followed by an MLP based on the provided parameters.

    """    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(input_shape, name='Input'))

    regularizer = get_regularizer(regularization_type, regularization_value)
    for i in range(1, 5):
        model.add(tf.keras.layers.Convolution2D(
            36, (3, 3), activation=activation, kernel_initializer=initializer,
            kernel_regularizer=regularizer, bias_initializer=initializer,
            bias_regularizer=regularizer, name=f'Convo{i}', padding='same'
        ))
        if i % 2 == 0:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f'Pool{i//2}'))
    
    for i in range(n_hidden):
        model.add(create_dense_layer(units, activation, initializer, regularizer, f"{layers_name}{i+1}"))
    
    model.add(create_dense_layer(10, predicter, initializer, regularizer, f"{layers_name}{n_hidden+1}"))

    return model


######################## Building Blocks ######################################
def create_dense_layer(units, activation, initializer, regularizer, layers_name):
    return tf.keras.layers.Dense(
        units=units, activation=activation, kernel_initializer=initializer,
        kernel_regularizer=regularizer, bias_initializer=initializer,
        bias_regularizer=regularizer, name=layers_name
    )

def get_regularizer(regularization_type, regularization_value):
    if regularization_type == 'l2':
        return tf.keras.regularizers.l2(regularization_value)
    elif regularization_type == 'l1':
        return tf.keras.regularizers.l1(regularization_value)
    return None

