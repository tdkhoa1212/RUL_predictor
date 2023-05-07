from model.autoencoder import autoencoder_model
from os.path import exists
from os.path import join
import tensorflow as tf

def train_EC(train_data, type_, weights, opt):
    print('\n' + 10*'-' + f' TRAIN AUTOENCODER MODEL WITH {type_} data ' + 10*'-')
    model = autoencoder_model(type_)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE))
    if exists(weights):
      model.load_weights(weights)
    model.fit(train_data, train_data,
                epochs=opt.EC_epochs,
                shuffle=True,
                batch_size=opt.batch_size)
    model.save(join(opt.save_dir, f'{type_}.h5'))