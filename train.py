from model.MIX_1D_2D import mix_model_PHM, mix_model_XJTU, mix_model_XJTU_SHAP
from model.resnet import resnet_101, resnet_34
from model.LSTM import lstm_extracted_model, lstm_model
from utils.tools import back_onehot, to_onehot, scaler_transform, all_matric_XJTU, all_matric_PHM
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
import tensorflow_addons as tfa
from tensorflow.keras.layers import  Dense
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import shap
import tensorflow as tf
import warnings
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--save_dir',       default='/content/drive/MyDrive/Khoa/results/RUL', type=str)
    parser.add_argument('--data_type',      default=['1d', '2d', 'extract'], type=list, help='shape of data. They can be 1d, 2d, extract')
    parser.add_argument('--train_bearing',  default=['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4'], type=str, nargs='+')   
    parser.add_argument('--test_bearing',   default=['Bearing2_5'], type=str, nargs='+')
    parser.add_argument('--condition',      default='c_all', type=str, help='c_1, c_2, c_3, c_all')
    parser.add_argument('--type',           default='XJTU', type=str, help='PHM, XJTU')
    parser.add_argument('--case',           default='case3', type=str, help='case1: OC independent rule, case2: OC dependent rule, case3: A Remaining Useful Life Prediction Method of Rolling Bearings Based on Deep Reinforcement Learning')
    parser.add_argument('--scaler',         default=None, type=str, help='PowerTransformer')
    parser.add_argument('--main_dir_colab', default='/content/drive/MyDrive/Khoa/data', type=str)

    parser.add_argument('--epochs',         default=300, type=int)
    parser.add_argument('--EC_epochs',      default=200, type=int)
    parser.add_argument('--batch_size',     default=16, type=int)
    parser.add_argument('--input_shape',    default=32768, type=int, help='1279 for using fft, 2560 for raw data in PHM, 32768 for raw data in XJTU')
    parser.add_argument('--noise_amplitude',default=0.1, type=float)

    parser.add_argument('--predict',      default=True, type=bool)
    parser.add_argument('--mix_model',    default=True,  type=bool)
    parser.add_argument('--encoder_train',default=False, type=bool)
    parser.add_argument('--PCAlabel',     default=False, type=bool)
    parser.add_argument('--load_weight',  default=False, type=bool)  
    parser.add_argument('--only_test',    default=True, type=bool) 
    parser.add_argument('--only_RUL',     default=True, type=bool) 

    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

# Train and test for PHM data ############################################################################################
def main_PHM(opt, train_1D, train_2D, train_extract, train_label_RUL, test_1D, test_2D, test_extract, test_label_RUL, only_test=False):  
  if opt.scaler != None:
    print(f'\nUse scaler: {opt.scaler}--------------\n')
    if opt.scaler == 'MinMaxScaler':
      scaler = MinMaxScaler
    if opt.scaler == 'MaxAbsScaler':
      scaler = MaxAbsScaler
    if opt.scaler == 'StandardScaler':
      scaler = StandardScaler
    if opt.scaler == 'RobustScaler':
      scaler = RobustScaler
    if opt.scaler == 'Normalizer':
      scaler = Normalizer
    if opt.scaler == 'QuantileTransformer':
      scaler = QuantileTransformer
    if opt.scaler == 'PowerTransformer':
      scaler = PowerTransformer
    train_1D = scaler_transform(train_1D, scaler)
    train_extract = scaler_transform(train_extract, scaler)
    test_1D = scaler_transform(test_1D, scaler)
    test_extract = scaler_transform(test_extract, scaler)
  
  val_2D, val_1D, val_extract, val_label_RUL = test_2D, test_1D, test_extract, test_label_RUL
  val_data = [val_1D, val_2D, val_extract]

  input_extracted = Input((14, 2), name='Extracted_LSTM_input')
  input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
  input_2D = Input((128, 128, 2), name='CNN_input')
  RUL = mix_model_PHM(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, True)
  network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)

  # get three types of different forms from original data-------------------------------
  train_data = [train_1D, train_2D, train_extract]
  test_data  = [test_1D, test_2D, test_extract]
  
  weight_path = os.path.join(opt.save_dir, f'model_{opt.type}')

  if only_test==False:
    if opt.load_weight:
      if os.path.exists(weight_path):
        print(f'\nLoad weight: {weight_path}\n')
        network.load_weights(weight_path)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=1)
    network.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
                    loss=tf.keras.losses.MeanSquaredLogarithmicError(), 
                    metrics=['mae', tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()]
  #                   run_eagerly=True
                      ) # https://keras.io/api/losses/ 
    network.summary()

    tf.debugging.set_log_device_placement(True)

    history = network.fit(train_data , train_label_RUL,
                          epochs     = opt.epochs,
                          batch_size = opt.batch_size,
                          validation_data = (val_data, val_label_RUL))
    network.save(weight_path)

  # ------------------------- PREDICT -------------------------------------
  RUL = mix_model_PHM(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, False)
  network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)
  network.load_weights(weight_path)
  RUL = network.predict(test_data)
  r2, mae_, mse_ = all_matric_PHM(test_label_RUL, RUL)
  RUL_mae = round(mae_, 4)
  RUL_r_square = round(r2, 4)
  RUL_mean_squared_error = round(mse_, 4)
  print(f'\n----------Score in test set: \n mae: {RUL_mae}, r2: {RUL_r_square}, rmse: {RUL_mean_squared_error}\n' )

# Train and test for XJTU data
def main_XJTU(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con):
    if opt.scaler is not None:
        print(f'\nUsing scaler: {opt.scaler}--------------\n')
        scaler = get_scaler(opt.scaler)  # Helper function to get scaler instance
        train_1D = scaler_transform(train_1D, scaler)
        train_extract = scaler_transform(train_extract, scaler)
        test_1D = scaler_transform(test_1D, scaler)
        test_extract = scaler_transform(test_extract, scaler)

    train_label_Con = to_onehot(train_label_Con)
    test_label_Con = to_onehot(test_label_Con)

    val_2D, val_1D, val_extract, val_label_Con, val_label_RUL = test_2D, test_1D, test_extract, test_label_Con, test_label_RUL
    val_data = [val_1D, val_2D, val_extract]
    val_label = [val_label_Con, val_label_RUL]

    input_extracted = Input((14, 2), name='Extracted_LSTM_input')
    input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
    input_2D = Input((128, 128, 2), name='CNN_input')

    Condition, RUL = mix_model_XJTU(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])

    train_data = [train_1D, train_2D, train_extract]
    train_label = [train_label_Con, train_label_RUL]
    test_data = [test_1D, test_2D, test_extract]
    test_label = [test_label_Con, test_label_RUL]

    # train_data = [data + np.random.normal(scale=opt.noise_amplitude, size=data.shape) for data in train_data]
    # test_data = [data + np.random.normal(scale=opt.noise_amplitude, size=data.shape) for data in test_data]

    weight_path = os.path.join(opt.save_dir, f'model_{opt.condition}_{opt.type}')

    if opt.load_weight:
        if os.path.exists(weight_path):
            print(f'\nLoad weight: {weight_path}\n')
            network.load_weights(weight_path)
        else:
            print(f'\nWeight file not found: {weight_path}\n')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=1)
    network.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
                    loss=['categorical_crossentropy', tf.keras.losses.MeanSquaredLogarithmicError()],
                    metrics=['acc', 'mae', tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()],
                    loss_weights=[1, 1])

    network.summary()

    history = network.fit(train_data, train_label,
                          epochs=opt.epochs,
                          batch_size=opt.batch_size,
                          validation_data=(val_data, val_label),
                          callbacks=[callback])  # Include early stopping callback

    network.save(weight_path)

    # ------------------------- PREDICT -------------------------------------
    Condition, RUL = mix_model_XJTU(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])
    network.load_weights(weight_path)
    Condition, RUL = network.predict(test_data)
    Condition = back_onehot(Condition)
    test_label_Con = back_onehot(test_label_Con)

    print("Shape of train_data:", [x.shape for x in train_data])
    print("Shape of test_data:", [x.shape for x in test_data])

    # Validation matrix
    r2, mae_, mse_, acc = all_matric_XJTU(test_label_RUL, RUL, test_label_Con, Condition)
    Condition_acc = round(acc * 100, 4)
    RUL_mae = round(mae_, 4)
    RUL_r_square = round(r2, 4)
    RUL_mean_squared_error = round(mse_, 4)

    print(f'\n----------Score in test set: \n Condition acc: {Condition_acc}, mae: {RUL_mae}, r2: {RUL_r_square}, rmse: {RUL_mean_squared_error}\n')

def main_XJTU_SHAP(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con):
    if opt.scaler is not None:
        print(f'\nUsing scaler: {opt.scaler}--------------\n')
        scaler = get_scaler(opt.scaler)  # Helper function to get scaler instance
        train_1D = scaler_transform(train_1D, scaler)
        train_extract = scaler_transform(train_extract, scaler)
        test_1D = scaler_transform(test_1D, scaler)
        test_extract = scaler_transform(test_extract, scaler)

    train_label_Con = to_onehot(train_label_Con)
    test_label_Con = to_onehot(test_label_Con)

    val_2D, val_1D, val_extract, val_label_Con, val_label_RUL = test_2D, test_1D, test_extract, test_label_Con, test_label_RUL
    val_data = [val_1D, val_2D, val_extract]

    input_extracted = Input((14, 2), name='Extracted_LSTM_input')
    input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
    input_2D = Input((128, 128, 2), name='CNN_input')

    RUL = mix_model_XJTU_SHAP(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)

    # get three types of different forms from original data
    train_data = [train_1D, train_2D, train_extract]
    test_data = [test_1D, test_2D, test_extract]

    weight_path = os.path.join(opt.save_dir, f'model_{opt.condition}_{opt.type}')

    if opt.only_test == False:
      if opt.load_weight:
          if os.path.exists(weight_path):
              print(f'\nLoad weight: {weight_path}\n')
              network.load_weights(weight_path)
          else:
              print(f'\nWeight file not found: {weight_path}\n')
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=1)
      network.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
                      loss=['categorical_crossentropy', tf.keras.losses.MeanSquaredLogarithmicError()],
                      metrics=['acc', 'mae', tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()],
                      loss_weights=[1, 1])

      network.summary()

      history = network.fit(train_data, train_label_RUL,
                            epochs=opt.epochs,
                            batch_size=opt.batch_size,
                            validation_data=(val_data, val_label_RUL),
                            callbacks=[callback])  # Include early stopping callback

      network.save(weight_path)

    # ------------------------- PREDICT -------------------------------------
    RUL = mix_model_XJTU_SHAP(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)
    network.load_weights(weight_path)

    # Create the first model up to the merged_value_0 layer
    first_model_output = network.layers[-2].output  # Extract output up to merged_value_0
    first_model = Model(inputs=[input_1D, input_2D, input_extracted], outputs=first_model_output)

    # Create the second model containing only the RUL layer
    final_layer_output = network.layers[-1].output
    second_model = Model(inputs=first_model_output, outputs=final_layer_output)

    # Verify the architecture of the models
    first_model.summary()
    second_model.summary()
    first_model.load_weights(weight_path)
    second_model.load_weights(weight_path)

    # SHAP explainer train_1D, train_2D, train_extract
    feature_names = np.array(['Denoised data', '2D data', '1D data'])
    extracted_train_data = first_model.predict(train_data)
    extracted_test_data = first_model.predict(test_data)

    # Your existing code for creating the SHAP values
    explainer_shap = shap.DeepExplainer(second_model, extracted_train_data)
    shap_values = np.squeeze(explainer_shap.shap_values(extracted_test_data, check_additivity=False))

    # Generate the SHAP summary plot without showing it
    shap.summary_plot(shap_values, features=extracted_test_data, feature_names=feature_names, show=False)

    ax = plt.gca()

    # Customize the x-axis to use the 10^x format
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'$10^{{{int(x):d}}}$'))


    # Save the plot to a file
    plt.savefig(os.path.join(opt.save_dir, 'shap_summary_plot.png'))

def main_XJTU_ML(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con):
    if opt.scaler is not None:
        print(f'\nUsing scaler: {opt.scaler}--------------\n')
        scaler = get_scaler(opt.scaler)  # Helper function to get scaler instance
        train_1D = scaler_transform(train_1D, scaler)
        train_extract = scaler_transform(train_extract, scaler)
        test_1D = scaler_transform(test_1D, scaler)
        test_extract = scaler_transform(test_extract, scaler)


    val_2D, val_1D, val_extract, val_label_Con, val_label_RUL = test_2D, test_1D, test_extract, test_label_Con, test_label_RUL
    train_extract = train_extract.reshape(train_extract.shape[0], train_extract.shape[1]*train_extract.shape[2])
    val_extract = val_extract.reshape(val_extract.shape[0], val_extract.shape[1]*val_extract.shape[2])

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(train_extract, train_label_RUL)
    ridge_pred = ridge.predict(val_extract)
    ridge_rmse = mean_squared_error(val_label_RUL, ridge_pred, squared=False)
    print("Ridge Regression RMSE:", np.round(ridge_rmse, 4))

    # Random Forest Regression
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_extract, train_label_RUL)
    rf_pred = rf.predict(val_extract)
    rf_rmse = mean_squared_error(val_label_RUL, rf_pred, squared=False)
    print("Random Forest Regression RMSE:", np.round(rf_rmse, 4))

    # XGBoost Regression
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 100)
    xg_reg.fit(train_extract, train_label_RUL)
    xgb_pred = xg_reg.predict(val_extract)
    xgb_rmse = mean_squared_error(val_label_RUL, xgb_pred, squared=False)
    print("XGBoost Regression RMSE:", np.round(xgb_rmse, 4))

if __name__ == '__main__':
  opt = parse_opt()
  print(f"Current training bearing: {opt.train_bearing}") 
  print(f"Current test bearing: {opt.test_bearing}\n") 
  print(f"Current noise: {opt.noise_amplitude}\n")

  # start_save_data(opt)
  if opt.type == 'PHM' and opt.case == 'case1':
    from utils.load_PHM_data import train_1D, train_2D, train_extract, train_label_RUL,\
                                    test_1D, test_2D, test_extract, test_label_RUL
    main_PHM(opt, train_1D, train_2D, train_extract, train_label_RUL, test_1D, test_2D, test_extract, test_label_RUL)
  elif opt.type == 'PHM' and opt.case == 'case2':
    from utils.load_PHM_data import train_1D, train_2D, train_extract, train_label_Con, train_label_RUL,\
                                    test_1D, test_2D, test_extract, test_label_Con, test_label_RUL
    main_XJTU(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con)
  else:
    from utils.load_XJTU_data import train_1D, train_2D, train_extract, train_label_Con, train_label_RUL,\
                                     test_1D, test_2D, test_extract, test_label_Con, test_label_RUL
    main_XJTU_SHAP(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con)
    # main_XJTU(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con)
