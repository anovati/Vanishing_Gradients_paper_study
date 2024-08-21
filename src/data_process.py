import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

from typing import Literal

# Full Pipeline

def data_process_pipeline(dataset_name: Literal['mnist', 'cifar'], model_type : Literal['mlp', 'cnn'], pca_ev : float = 0.95):
    
    print(f"Processing {dataset_name} for {model_type} model...")
    # Download data
    print(f'Step 1: Downloading {dataset_name} dataset')
    (x_train, y_train), (x_test, y_test) = download_dataset(dataset_name)
    
    # Normalize data
    print('Step 2: Normalizing data')
    x_train, x_test = normalize_dataset(x_train), normalize_dataset(x_test)

    # Process differently based on model and dataset
    if model_type == 'cnn' and dataset_name == 'mnist':
        x_train = tf.expand_dims(x_train, axis=-1)
        x_test  = tf.expand_dims(x_test, axis=-1)

    if model_type == 'mlp':
        x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32)
        x_test  = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)

        # Apply PCA for MLP
        print('Step 3: Applying PCA')
        x_train, x_test = apply_pca(pca_ev, x_train, x_test)

    print('Data is ready to run the notebook')
    return (x_train, y_train), (x_test, y_test)


# Modular functions 

def download_dataset(dataset_name: Literal['mnist', 'cifar']) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    if dataset_name not in ['mnist', 'cifar']:
        raise ValueError('For this project, only "mnist" and "cifar" datasets are considered')
    if dataset_name == 'mnist':
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name == 'cifar':
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    return (x_train, y_train), (x_test, y_test)


def normalize_dataset(dataset: tf.Tensor):
    dataset_norm = dataset /255.0*2 - 1
    return dataset_norm 


def apply_pca(explained_variance : float, train_dataset: tf.Tensor, test_dataset: tf.Tensor):
    pca = PCA()
    pca.fit(train_dataset)

    ev_cumsum = np.cumsum(pca.explained_variance_)/(pca.explained_variance_).sum() 
    ev_at95 = ev_cumsum[ev_cumsum<explained_variance].shape[0]

    pca = PCA(ev_at95)
    pca.fit(train_dataset)

    return pca.transform(train_dataset), pca.transform(test_dataset)







