'''
Assignment 2 functions to help with analysis
--------------------------------------------
Jericho O'Connell 2022
'''
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from joblib import parallel_backend
import time
import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Taken from one of the notebooks from class
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Define a function to add noise
def add_noise_to_MNIST(inp_tr, inp_va, c_noise):
    '''
    Add noise to MNIST data
    -----------------------
    
    Inputs
    ------
    inp_tr: numpy arrays
        input and target arrays for training
    inp_va: numpy arrays
        input and target arrays for validation
    Returns
    -------
    inp_tr, inp_va: numpy arrrays
    
    '''
    
    # So I can specify the amount of the dataset to use
    tr_x = inp_tr.shape[0]
    va_x = inp_va.shape[0]
    
    # Just reshape and normalize if zero
    if c_noise == 0:
        
        return np.reshape(inp_tr, (-1, 784)) / 255., np.reshape(inp_va, (-1, 784)) / 255.
    
    # Make some arrays of Noise
    noise_tr=np.random.rand(tr_x,784)*c_noise
    noise_va=np.random.rand(va_x,784)*c_noise

    # Add an empty dimension just in case sklearn wants
    # Then normalize by 255
    inp_tr = np.reshape(inp_tr, (-1, 784)) / 255.
    inp_va = np.reshape(inp_va, (-1, 784)) / 255.
    
    # Return the noisy data renormalized
    return (inp_tr+noise_tr)/(1+c_noise), (inp_va+noise_va)/(1+c_noise)

# classification accuracy function
def return_classification_accuracy(inp_tr, tar_tr, inp_va, tar_va, c_noise=0 , PCA_comp = 0,
                                   model=LogisticRegression, conf_matrix=False, **kwargs):
    '''
    Return the classification accuracy for a given noise and pca
    -----------------
    
    Inputs
    ------
    inp_tr, tar_tr: numpy arrays
        input and target arrays for training
    inp_va, tar_va: numpy arrays
        input and target arrays for validation
    c_noise: float
        The noise relative to the normalized value of the
        data, 0 for no noise
    PCA_comp: int
        number of PCs to compress the data to 0 for no
        PCA
    model: sklearn model object
        model to train
    ** kwargs: keyword arguments to pass to the model
    
    Returns
    -------
    training_score: float
        model performance on training data
    validation_score: float
        model performance on training data
    time_taken: float
        time taken to train
    
    '''
    
    # I'll use all of my cores because this can be slow
    with parallel_backend('threading', n_jobs=8):
        
        # Add noise
        # Use the function defined above
        inp_tr, inp_va = add_noise_to_MNIST(inp_tr, inp_va, c_noise)
        
        # Initialize and transform data using PCA
        # If there is a number of components to use then fit to the
        # number of components
        if PCA_comp > 0:
            pca = PCA(n_components=PCA_comp)
            pca.fit(inp_tr)
            inp_tr = pca.transform(inp_tr)
            inp_va = pca.transform(inp_va)
        
        print(f'Noise is {c_noise}, number of PCs is {PCA_comp}')
        print(f'Shape of the training, validation data {inp_tr.shape, inp_va.shape}')
        # Initialize the model
        model_instance = model(**kwargs)
        
        # Look at the time started
        start = time.time()
        
        # Fit the model
        model_instance.fit(inp_tr, tar_tr)
        pred_tr= model_instance.predict(inp_tr)
        pred_va= model_instance.predict(inp_va)
        
        # Look at the time at the end
        end = time.time()
        time_taken = end - start
        print(f'The time taken was {time_taken} seconds\n')
    
    if conf_matrix:
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(pred_va, tar_va)
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['0','1', '2','3','4','5','6','7','8','9'], normalize=True,
                              title='Normalized confusion matrix')
    
    # Return the performance of the model
    return model_instance.score(inp_tr, tar_tr), model_instance.score(inp_va,tar_va), time_taken