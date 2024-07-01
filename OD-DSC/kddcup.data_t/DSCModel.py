import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
#import utilss
import tensorflow as tf
tf.random.set_seed(7)
import matplotlib.pyplot as plt

class DSC(object):

    def __init__(self,  network_architecture, learning_rate, batch_size, alpha, C_point_index, model_path=None):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        #self.dim = dim
        self.C_point_index = C_point_index
        self.model_path = model_path

        self.enc = self.encoder_model()
        self.se = self.se_model()
        self.dec = self.decoder_model()


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

        E_h2 = layers.Dense(self.network_architecture['n_hidden_enc_2'], activation='relu')(E_h1)
        #E_h3 = layers.BatchNormalization()(E_h2)



        #数据拉平
        self.en_dim_1 = E_h2.shape[1]
        model = tf.keras.Model(inputs=input, outputs=E_h2, name="encoder")
        model.summary()
        return model

        # Self-Expression layer
    def se_model(self):
        inputs = tf.keras.Input(self.batch_size, batch_size=self.network_architecture['n_hidden_enc_2'])
        outputs = layers.Dense(self.batch_size, activation=None, use_bias=False)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
    # Decoder
    def decoder_model(self):
        # Functional model
        #inputs = tf.keras.Input(self.network_architecture['n_hidden_dec_1'], batch_size=self.batch_size)
        inputs = tf.keras.Input(self.network_architecture['n_hidden_dec_1'], batch_size=self.batch_size)
        D_h1 = layers.Dense(self.network_architecture['n_hidden_dec_2'], activation='relu')(inputs)
        #D_h2 = layers.UpSampling2D(size=(2, 2))(D_h1
        D_h2 = layers.Dense(self.network_architecture['n_input'], activation='relu')(D_h1)
        #D_h3 = layers.BatchNormalization()(D_h2)


        model = tf.keras.Model(inputs=inputs, outputs=D_h2, name="decoder")
        model.summary()
        return model


    # dsc network
    def dsc_network(self, X, is_training=False):

        # use the encode network to get the self_expression Z
        self.z = self.enc(X, training=is_training)  # [batch_size, dim]
        self.z_c = self.se(tf.transpose(self.z), training=is_training)  # [dim, batch_size]
        self.z_c = tf.transpose(self.z_c)
        #d_In = tf.reshape(self.z_c, [self.batch_size, self.en_dim_1, self.en_dim_1, 8])
        self.rec_X = self.dec(self.z_c, training=is_training)
        #pit_1 = tf.reshape(self.rec_X, [3200, 28, 28])
        #plt.matshow(pit_1[1])
        #plt.show()

    
    
    def loss_optimizer(self, X):
        # loss function of Generator
        self.loss_MSE = tf.reduce_sum((self.rec_X - X) ** 2)
        self.Coef = self.se.trainable_variables[0].numpy()
        self.loss_se_Coef = tf.reduce_sum(self.Coef ** 2)
        self.loss_se = tf.reduce_sum((self.z_c - self.z) ** 2)
        self.loss = self.loss_MSE+ self.alpha['a1'] * self.loss_se_Coef + self.alpha['a2'] * self.loss_se
        #self.loss = self.alpha['a1'] * self.loss_se_Coef + self.alpha['a2'] / 2 * self.loss_se


    def train_step(self, X, is_training):
        with tf.GradientTape() as tape:
            self.dsc_network(X, is_training)
            self.loss_optimizer(X)
            self.gradients = tape.gradient(self.loss, self.var_list)
            self.optimizer.apply_gradients(zip(self.gradients, self.var_list))



    def save_model(self):
        self.enc.save(self.model_path + "enc/Change_point_model/enc_CP_" + self.C_point_index + ".h5", save_format='tf')
        self.dec.save(self.model_path + "dec/Change_point_model/dec_CP_" + self.C_point_index + ".h5", save_format='tf')
        self.se.save(self.model_path + "se/Change_point_model/se_CP_"+ self.C_point_index + ".h5", save_format='tf')
        #self.se.save_weights(self.model_path + "/se/coef.h5")




def train_model(DSC, norm_data, training_epochs):
    display_step = 1
    save_step = 10
    # train the network
    print("Start training...")
    # DACIN.trainable_params()
    for epoch in range(training_epochs):
        # Fit training using batch data
        DSC.train_step(norm_data, is_training=True)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(DSC.loss), "loss_MSE=", "{:.9f}".format(DSC.loss_MSE),
                  "loss_se_Coef=", "{:.9f}".format(DSC.loss_se_Coef), "loss_se=", "{:.9f}".format(DSC.loss_se))
        if epoch % save_step == 0:
            DSC.save_model()




def coef_get(model_path, norm_data, C_point_index):
    encoder = keras.models.load_model(model_path + 'enc/Change_point_model/enc_CP_'+ C_point_index +'.h5', compile=False)
    decoder = keras.models.load_model(model_path + 'dec/Change_point_model/dec_CP_' + C_point_index + '.h5', compile=False)
    se = keras.models.load_model(model_path + 'se/Change_point_model/se_CP_' + C_point_index + '.h5', compile=False)
    z = encoder(norm_data)
    z_c = se(tf.transpose(z))
    z_c = tf.transpose(z_c)
    y_intput = decoder(z_c)
    coef = se.trainable_variables[0].numpy()
    print("z: ", z.shape, "z_c: ", z_c.shape, "coef: ", coef.shape)
    return y_intput, coef

