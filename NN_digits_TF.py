## NN_digits_TF.py
## Neural Network - Digit Image Classification using MNIST data.
## Using TensorFlow Keras to model/prototype.

from NN_Digits_Model import NN_Digits_Model;
from random import randint;

import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;    #   To display digit image.

def plot_image(x):
    ##  Using matplotlib to display the gray-scaled digit image.
    ##  Input:  2D np.darray representing (28x28 pixels).

    image = np.asarray(x).squeeze();
    plt.imshow(image);
    plt.show();
    return;

def main():


    ## Import training and test data from mnist.
    mnist = tf.keras.datasets.mnist;
    (x_train, y_train), (x_test, y_test) = mnist.load_data();


    ## Normalize data to avoid overflow/over-rounding numbers.
    avg, std    =   np.mean(x_train), np.std(x_train);
    x_train     =   (x_train - avg)/std;
    x_test      =   (x_test  - avg)/std;    


    ## Randomly pick 10k data in 32 batch sizes, if using Conv2D.
    #train_ds    =   tf.data.Dataset.from_tensor_slices( (x_train, y_train)).shuffle(10000).batch(32);
    #test_ds     =   tf.data.Dataset.from_tensor_slices( (x_test,  y_test) ).batch(32);


    ## Generate NN model: 28*28, 128, 10 units. 
    model   =   NN_Digits_Model();

    ## Training the model weight parameters using train dataset.
    model.train_NN( x_train, y_train);    


    # Test the model on the test (validation) samples,
    # and prints out loss and accuracy result.
    print("\nValidation test result on 10k samples: ");
    eval_loss, eval_accuracy  =   model.test_accuracy(x_test, y_test);  # ~ 97% accuracy.


    ## Testing on N random samples from test dataset.
    N = 20;
    err_cnt =   0;

    for _ in range(N)  :

        ## pick a random sample from test dataset (x_test, y_test).
        i           =   randint(0, 10000 - 1);      
        x_sample    =   x_test[i:i+1];
        
        y_predict   =   model.predict_funct(    x_sample)   [0];

        ## If mismatch found between y_output and predicted_output, then
        ##   prints and display the image.
        if y_predict != y_test[i]:

            err_cnt     +=  1;
            print("Prediction = ", y_predict, ", ", y_test[i], "= Output data.");  
            plot_image( x_sample);     

    print("Error rate: ", err_cnt/N);

    return 0;


if __name__ == '__main__':      main();