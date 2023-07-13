from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, TimeDistributed, RepeatVector, Conv1D, Conv1DTranspose
from keras.models import Model

# https://keras.io/examples/timeseries/timeseries_anomaly_detection/
def autoencoder_model(type_):
    if type_ == 'PHM':
      inputs = Input(shape=(2, 2560))
      x1 = 2
      x2 = 2560
    else:
      inputs = Input(shape=(2, 32768))
      x1 = 2
      x2 = 32768
    L1 = LSTM(512, return_sequences=True, activation="relu")(inputs)
    L1 = Dropout(0.1)(L1)
    L2 = LSTM(64, return_sequences=False, activation="relu")(L1)
    # L2 = Dropout(0.1)(L2)
    L3 = RepeatVector(x1)(L2)
    L4 = LSTM(64, return_sequences=True, activation="relu")(L3)
    # L4 = Dropout(0.1)(L4)
    L5 = LSTM(512, return_sequences=True, activation="relu")(L4)
    L5 = Dropout(0.1)(L5)
    output = TimeDistributed(Dense(x2, activation="relu"))(L5)    

    # x = Conv1D(filters=128, kernel_size=7, padding="same", strides=2, activation="relu")(inputs)
    # x = Dropout(rate=0.2)(x)
    # x = Conv1D(filters=64, kernel_size=7, padding="same", strides=2, activation="relu")(x)
    # x = Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=1, activation="relu")(x)
    # x = Dropout(rate=0.2)(x)
    # x = Conv1DTranspose(filters=128, kernel_size=7, padding="same", strides=2, activation="relu")(x)
    # output = TimeDistributed(Dense(x2))(x)

    model = Model(inputs=inputs, outputs=output)
    return model
