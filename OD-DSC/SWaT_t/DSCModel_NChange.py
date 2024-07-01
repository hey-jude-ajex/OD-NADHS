import numpy as np
#from numpy.random import seed
#seed(3)
import tensorflow as tf

tf.random.set_seed(7)
from tensorflow import keras
from tensorflow.keras import layers
#import utilss
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


class DSC(object):

    def __init__(self,  network_architecture, learning_rate, batch_size, alpha,  batch_num, C_point_index, model_path=None):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        #self.dim = dim
        self.model_path = model_path
        self.batch_num = batch_num
        self.C_point_index = C_point_index

        self.enc_base_model = tf.keras.models.load_model(
            self.model_path + "enc/Change_point_model/enc_CP_" + self.C_point_index + ".h5", compile=False)
        self.se_base_model = tf.keras.models.load_model(
            self.model_path + "se/Change_point_model/se_CP_" + self.C_point_index + ".h5", compile=False)
        self.dec_base_model = tf.keras.models.load_model(
            self.model_path + "dec/Change_point_model/dec_CP_" + self.C_point_index + ".h5", compile=False)

        self.enc = self.encoder_model()
        self.se = self.se_model()
        self.dec = self.decoder_model()

        self.enc.set_weights(self.enc_base_model.get_weights())
        self.se.set_weights(self.se_base_model.get_weights())
        self.dec.set_weights(self.dec_base_model.get_weights())

        #print(type(self.enc.trainable_variables), len(self.enc.trainable_variables), self.enc.trainable_variables[0])
        self.var_list = []
        for var in self.enc.trainable_variables:
            self.var_list.append(var)
        for var in self.dec.trainable_variables:
            self.var_list.append(var)
        for var in self.se.trainable_variables:
            self.var_list.append(var)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)



    # Encoder
    def encoder_model(self):
        # Functional model
        input = tf.keras.Input(self.network_architecture['n_input'], batch_size=self.batch_size)
        E_h1 = layers.Dense(self.network_architecture['n_hidden_enc_1'], activation='relu')(input)
        E_h3 = layers.Dense(self.network_architecture['n_hidden_enc_2'], activation='relu')(E_h1)
        model = tf.keras.Model(inputs=input, outputs=E_h3, name="encoder")
        #model.summary()
        return model

        # Self-Expression layer
    def se_model(self):
        # Functional model):
        inputs = tf.keras.Input(self.batch_size, batch_size=self.network_architecture['n_hidden_enc_2'])
        outputs = layers.Dense(self.batch_size, activation=None, use_bias=False)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="se")
        #model.summary()
        return model
    # Decoder
    def decoder_model(self):
        # Functional model):
        # Functional model
        #inputs = tf.keras.Input(self.network_architecture['n_hidden_dec_1'], batch_size=self.batch_size)
        inputs = tf.keras.Input(self.network_architecture['n_hidden_dec_1'], batch_size=self.batch_size)
        D_h1 = layers.Dense(self.network_architecture['n_hidden_dec_2'], activation='relu')(inputs)
        #D_h2 = layers.UpSampling2D(size=(2, 2))(D_h1)
        D_h2 = layers.Dense(self.network_architecture['n_input'], activation='relu')(D_h1)

        model = tf.keras.Model(inputs=inputs, outputs=D_h2, name="decoder")
        #model.summary()
        return model


    # dsc network
    def dsc_network(self, X, is_training=False):

        # use the encode network to get the self_expression Z
        self.z = self.enc(X, training=is_training)  # [batch_size, dim]
        self.z_c = self.se(tf.transpose(self.z), training=is_training)  # [dim, batch_size]
        self.z_c = tf.transpose(self.z_c)
        #d_In = tf.reshape(self.z_c, [self.batch_size, self.en_dim_1, self.en_dim_1, 8])
        self.rec_X = self.dec(self.z_c, training=is_training)



    def loss_optimizer_NChange(self, X):
        # loss function of Generator
        self.loss_MSE = tf.reduce_sum((self.rec_X - X) ** 2)
        self.Coef = self.se.trainable_variables[0].numpy()
        self.loss_se_Coef = tf.reduce_sum(self.Coef ** 2)
        self.loss_se = tf.reduce_sum((self.z_c - self.z) ** 2)
        self.loss = self.loss_MSE + self.alpha['a1'] * self.loss_se_Coef + self.alpha['a2'] * self.loss_se

    def train_step_NChange(self, X, is_training):
        with tf.GradientTape() as tape:
            self.dsc_network(X, is_training)
            self.loss_optimizer_NChange(X)
            #print("95", self.var_list)
            self.gradients = tape.gradient(self.loss, self.var_list)
            #print("96", self.var_list)
            self.optimizer.apply_gradients(zip(self.gradients, self.var_list))

    def save_model_NChange(self):
        self.enc.save(self.model_path + 'enc/enc_NChange'+ self.batch_num + '.h5', overwrite=True,save_format='tf')
        self.dec.save(self.model_path + 'dec/dec_NChange'+ self.batch_num + '.h5', overwrite=True,save_format='tf')
        self.se.save(self.model_path + 'se/se_NChange/se_NChange' + self.batch_num + '.h5', overwrite=True,save_format='tf')

from shutil import copyfile
def train_model_NChange(DSC, norm_data, training_epochs, C_point_index):

    display_step = 1
    save_step = 5
    # train the network
    print("Start training...")
    #model_copy

    # DACIN.trainable_params()


    for epoch in range(training_epochs):

        # Fit training using batch data
        DSC.train_step_NChange(norm_data, is_training=True)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(DSC.loss),
                  "loss_se_Coef=", "{:.9f}".format(DSC.loss_se_Coef), "loss_se=", "{:.9f}".format(DSC.loss_se))
        if epoch % save_step == 0:
            DSC.save_model_NChange()

def get_RMSE(y_int, y_out):
        R = np.sqrt(((y_int - y_out) ** 2).mean())
        return R

import time
def coef_get_NChange(model_path, norm_data, C_point_index, batch_num):
    encoder = keras.models.load_model(model_path + 'enc/Change_point_model/enc_CP_' + C_point_index + '.h5', compile=False)
    decoder = keras.models.load_model(model_path + 'dec/Change_point_model/dec_CP_' + C_point_index + '.h5', compile=False)
    se = keras.models.load_model(model_path + 'se/se_NChange/se_NChange'+ batch_num + '.h5', compile=False)

    z = encoder(norm_data)
    z_c = se(tf.transpose(z))
    z_c = tf.transpose(z_c)
    y_intput = decoder(z_c)
    coef = se.trainable_variables[0].numpy()

    #print("test_RMSE", get_RMSE(np.array(norm_data[676:717,:]), np.array(y_intput[676:717,:])))


    return y_intput, coef
