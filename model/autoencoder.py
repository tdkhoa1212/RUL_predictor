from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
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
    L1 = LSTM(128, return_sequences=True, activation='tanh')(inputs)
    L2 = LSTM(32, return_sequences=False, activation='tanh')(L1)
    L3 = RepeatVector(x1)(L2)
    L4 = LSTM(32, return_sequences=True, activation='tanh')(L3)
    L5 = LSTM(128, return_sequences=True, activation='tanh')(L4)
    output = TimeDistributed(Dense(x2))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model
