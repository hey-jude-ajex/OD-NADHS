import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
#import utilss
import tensorflow as tf
import matplotlib.pyplot as plt
import SpectralCluster
import tensorflow as tf
tf.random.set_seed(7)



class DSC(object):

    def __init__(self, network_architecture, learning_rate, batch_size, alpha, model_path=None):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        # self.dim = dim
        self.model_path = model_path

        self.enc = self.encoder_model()
        self.dec = self.decoder_model()

        self.var_list = []
        for var in self.enc.trainable_variables:
            self.var_list.append(var)
        for var in self.dec.trainable_variables:
            self.var_list.append(var)
        #for var in self.se.trainable_variables:
            #self.var_list.append(var)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    # Encoder
    def encoder_model(self):
        # Functional model
        input = tf.keras.Input(self.network_architecture['n_input'], batch_size=self.batch_size)
        E_h1 = layers.Dense(self.network_architecture['n_hidden_enc_1'], activation='relu')(input)
        # E_h2 = layers.MaxPooling2D(pool_size=(2, 2))(E_h1)
        E_h3 = layers.Dense(self.network_architecture['n_hidden_enc_2'], activation='relu')(E_h1)

        # iter_i= 1
        # while iter_i < n_layers:
        #    E_h1 = layers.Conv2D(50, 6, strides=[2, 2], padding='VALID', activation='relu')(E_h1)
        #    iter_i += 1
        # self.In_X = E_h1
        # print("E_h1", E_h1.shape)
        # E_h3 = layers.Conv2D(16, 5, strides=[1,2,2,1], padding='VALID', activation='relu')(E_h2)
        # print("E_h3", E_h3)
        # 数据拉平
        self.en_dim_1 = E_h3.shape[1]
        model = tf.keras.Model(inputs=input, outputs=E_h3, name="encoder")
        model.summary()
        return model

        # Self-Expression layer


    # Decoder
    def decoder_model(self):
        # Functional model
        # inputs = tf.keras.Input(self.network_architecture['n_hidden_dec_1'], batch_size=self.batch_size)
        inputs = tf.keras.Input(self.network_architecture['n_hidden_dec_1'], batch_size=self.batch_size)
        D_h1 = layers.Dense(self.network_architecture['n_hidden_dec_2'], activation='relu')(inputs)
        # D_h2 = layers.UpSampling2D(size=(2, 2))(D_h1)
        D_h2 = layers.Dense(self.network_architecture['n_input'], activation='relu')(D_h1)

        model = tf.keras.Model(inputs=inputs, outputs=D_h2, name="decoder")
        model.summary()
        return model

    # dsc network
    def dsc_network(self, X, is_training=False):

        # use the encode network to get the self_expression Z
        self.z = self.enc(X, training=is_training)  # [batch_size, dim]
        #self.z_c = self.se(tf.transpose(self.z), training=is_training)  # [dim, batch_size]
        #self.z_c = tf.transpose(self.z_c)
        # d_In = tf.reshape(self.z_c, [self.batch_size, self.en_dim_1, self.en_dim_1, 8])
        self.rec_X = self.dec(self.z, training=is_training)
        # pit_1 = tf.reshape(self.rec_X, [3200, 28, 28])
        # plt.matshow(pit_1[1])
        # plt.show()

    def loss_optimizer(self, X):
        # loss function of Generator
        self.loss_MSE = tf.reduce_sum((self.rec_X - X) ** 2)
        self.loss = self.loss_MSE
        # self.loss = self.alpha['a1'] * self.loss_se_Coef + self.alpha['a2'] / 2 * self.loss_se

    def train_step(self, X, is_training):
        with tf.GradientTape() as tape:
            self.dsc_network(X, is_training)
            self.loss_optimizer(X)
            self.gradients = tape.gradient(self.loss, self.var_list)
            self.optimizer.apply_gradients(zip(self.gradients, self.var_list))

    def save_model(self):
        self.enc.save(self.model_path + "/enc/enc.h5", save_format='tf')
        self.dec.save(self.model_path + "/dec/dec.h5", save_format='tf')
        #  self.se.save(self.model_path + "/se/se.h5", save_format='tf')
        # self.se.save_weights(self.model_path + "/se/coef.h5")


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
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(DSC.loss))
        if epoch % save_step == 0:
            DSC.save_model()


def coef_get(model_path, norm_data):
    encoder = keras.models.load_model(model_path + './enc/enc_t.h5', compile=False)
    decoder = keras.models.load_model(model_path + './dec/dec_t.h5', compile=False)
    #se = keras.models.load_model(model_path + './se/se.h5', compile=False)
    z = encoder(norm_data)
    #z_c = se(tf.transpose(z))
    #z_c = tf.transpose(z_c)
    y_intput = decoder(z)
    #coef = se.trainable_variables[0].numpy()
    print("z: ", z.shape)
    return y_intput


import pandas as pd

if __name__ == "__main__":
    data_dir = '../data/'
    # data_dir_e = '../data/test_data/test_normal_result_minmax_1.csv'

    data_dir_e = '../../data/stock/time-series-toNorm/a_0_normal.csv'
    print("data_dir: ", data_dir)
    print(data_dir_e)
    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)

    data_test = np.array(data_test)
    print("source_data_shape", data_test.shape)
    norm_data = data_test[:, 0:27]

    # batch_size = 3200
    learning_rate = 0.05
    training_epochs = 150

    batch_size, dim = norm_data.shape

    network_architecture = dict(n_hidden_enc_1=dim,  # 1nd layer encode neurons
                                n_hidden_enc_2=dim,  # 2nd layer encode neurons
                                n_hidden_dec_1=dim,  # 1nd layer decode neurons
                                n_hidden_dec_2=dim,  # 2nd layer decode neurons
                                n_input=dim)  # dimension of data input
    alpha_dsc = {
        'a1': 1,
        'a2': 18,
    }
    # ro = 0.7
    # ro = max(0.4 - (5 - 1) / 10 * 0.1, 0.1)

    model_path = './model/'

    dsc =DSC(network_architecture=network_architecture, learning_rate=learning_rate, batch_size=batch_size,
                       alpha=alpha_dsc,
                       model_path=model_path)
    # train model
    train_model(dsc, norm_data, training_epochs=training_epochs)


    # 不需训练模型

