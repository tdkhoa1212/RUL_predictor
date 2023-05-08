from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, TimeDistributed, RepeatVector
import tensorflow as  tf
from keras.models import Model

def autoencoder_model(type_):
    if type_ == 'PHM':
      inputs = Input(shape=(2, 2560))
      x1 = 2
      x2 = 2560
    else:
      inputs = Input(shape=(2, 32768))
      x1 = 2
      x2 = 32768
    # L1 = LSTM(512, return_sequences=True)(inputs)
    # L2 = LSTM(256, return_sequences=False)(L1)
    # L3 = RepeatVector(x1)(L2)
    # L4 = LSTM(256, return_sequences=True)(L3)
    # L5 = LSTM(512, return_sequences=True)(L4)
    # # L5 = Dropout(0.2)(L5)
    # output = TimeDistributed(Dense(x2))(L5)    
    x = Dense(512)(inputs)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = Dropout(0.2)(x)
    output = Dense(x2)(x)
    model = Model(inputs=inputs, outputs=output)
    return model
