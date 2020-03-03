## NN_Digits_Model.py

from __future__ import division, absolute_import, print_function, unicode_literals;

import os;
import tensorflow as tf;

from tensorflow.keras.layers import Dense, Flatten, Conv2D;
from tensorflow.keras import Model;

import numpy as np;

BATCH_SIZE = 32;


## Inherit class from tf.keras.Model
class   NN_Digits_Model(Model):

    def __init__(self):

        #super(self).__init__(self);
        super(NN_Digits_Model, self).__init__();

        self.epoch  =   5;


        ## Initializing the layers.
        self.L1_input           =   Flatten(input_shape=(28, 28));
        self.L1_conv            =   Conv2D( BATCH_SIZE, 3, activation = 'relu');

        self.L2_hidden          =   Dense(  128, activation = 'relu');
        self.L2_regulation      =   tf.keras.layers.Dropout(    0.2);

        self.L3_output_classify =   Dense(  10);

        self.L3_output_softmax  =   tf.keras.layers.Softmax();


        ## Stacking the NN layers to build a model.
        self.model = tf.keras.models.Sequential([
                                                    self.L1_input,
                                                    #self.L1_conv,
                                                    self.L2_hidden,
                                                    self.L2_regulation,
                                                    self.L3_output_classify,
                                                    #self.L3_output_softmax,
                                                    ]);
        self.compile_model();        

        return;
        
    def compile_model(self):
        ## Loss function: Use cross-entropy (log).
        loss_fn     =   tf.keras.losses.SparseCategoricalCrossentropy(    from_logits = True);

        ## Optimizer.
        #opt    =  tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False);
        #opt         =   tf.keras.optimizers.SGD(learning_rate=0.01);
        opt         =  tf.keras.optimizers.Adam();

        self.model.compile(     optimizer   =   opt,
                                loss        =   loss_fn,
                                metrics     =   ['accuracy']);
        
        return;

    def train_NN(self, X_input, y_output):
        # Add a channels dimension.
        #X_input =   X_input[..., tf.newaxis];

        self.model.fit(     X_input, y_output, 
                            epochs  =   self.epoch, 
                            callbacks = None, 
                            verbose = 2);
        return;

    def predict_funct(self, X_input):
        X_input     =   X_input.astype(np.float32);
        #X_input =   X_input[..., tf.newaxis];      # Adding another dimension if using Conv2D input layer.

        prob_output =   self.model.predict(X_input);
        prob_output =   self.L3_output_softmax(prob_output)[0].numpy();

        predicted_digit =   np.argmax(prob_output);

        return  predicted_digit, prob_output;

    def test_accuracy(self, X_input, y_output):
        
        eval_loss, eval_accuracy  =   self.model.evaluate(     X_input,  y_output, verbose=2);
        return eval_loss, eval_accuracy ;

