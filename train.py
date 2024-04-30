from model.MIX_1D_2D import mix_model_PHM, mix_model_XJTU
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
import argparse
import os
import shap
import tensorflow as tf
import warnings

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
    
    parser.add_argument('--predict',      default=True, type=bool)
    parser.add_argument('--mix_model',    default=True,  type=bool)
    parser.add_argument('--encoder_train',default=False, type=bool)
    parser.add_argument('--PCAlabel',     default=False, type=bool)
    parser.add_argument('--load_weight',  default=False, type=bool)  
    parser.add_argument('--only_test',    default=False, type=bool) 
    parser.add_argument('--only_RUL',     default=False, type=bool) 

    
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

# Train and test for XJTU data ############################################################################################
def main_XJTU(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con, only_test=False, only_RUL=True):  
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
  
  train_label_Con = to_onehot(train_label_Con)
  test_label_Con  = to_onehot(test_label_Con)

  val_2D, val_1D, val_extract, val_label_Con, val_label_RUL = test_2D, test_1D, test_extract, test_label_Con, test_label_RUL
  val_data = [val_1D, val_2D, val_extract]
  val_label = [val_label_Con, val_label_RUL]

  input_extracted = Input((14, 2), name='Extracted_LSTM_input')
  input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
  input_2D = Input((128, 128, 2), name='CNN_input')
  
  if only_RUL:
    RUL = mix_model_XJTU(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, only_RUL, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)

  else:
    Condition, RUL = mix_model_XJTU(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, only_RUL, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])

  # get three types of different forms from original data-------------------------------
  train_data = [train_1D, train_2D, train_extract]
  train_label = [train_label_Con, train_label_RUL]
  test_data = [test_1D, test_2D, test_extract]
  test_label = [test_label_Con, test_label_RUL]
  
  weight_path = os.path.join(opt.save_dir, f'model_{opt.condition}_{opt.type}')

  if only_test == False:
    if opt.load_weight:
      if os.path.exists(weight_path):
        print(f'\nLoad weight: {weight_path}\n')
        network.load_weights(weight_path)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=1)
    network.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
                    loss=['categorical_crossentropy', tf.keras.losses.MeanSquaredLogarithmicError()], 
                    metrics=['acc', 'mae', tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()], 
                    loss_weights=[1, 1],
  #                   run_eagerly=True
                      ) # https://keras.io/api/losses/ 
    network.summary()

    tf.debugging.set_log_device_placement(True)

    history = network.fit(train_data , train_label,
                          epochs     = opt.epochs,
                          batch_size = opt.batch_size,
                          validation_data = (val_data, val_label))
    network.save(weight_path)

  # ------------------------- PREDICT -------------------------------------
  if only_RUL:
    RUL = mix_model_XJTU(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, only_RUL, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)
    network.load_weights(weight_path)
    RUL = network.predict(test_data)
  else:
    Condition, RUL = mix_model_XJTU(opt, lstm_model, resnet_34, lstm_extracted_model, input_1D, input_2D, input_extracted, only_RUL, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])
    network.load_weights(weight_path)
    Condition, RUL = network.predict(test_data)
    Condition = back_onehot(Condition)
    test_label_Con = back_onehot(test_label_Con)

  # SHAP explainer
  feature_names = ['1D', '2D', 'Extract']
  explainer_shap = shap.DeepExplainer(network, train_data)
  shap_values = explainer_shap.shap_values(test_data, check_additivity=False)
  shap.summary_plot(shap_values, features=test_data, feature_names=feature_names)

  # Validation matrix
  if only_RUL == False:
    r2, mae_, mse_, acc = all_matric_XJTU(test_label_RUL, RUL, test_label_Con, Condition)
    Condition_acc = round(acc*100, 4)
    RUL_mae = round(mae_, 4)
    RUL_r_square = round(r2, 4)
    RUL_mean_squared_error = round(mse_, 4)

    print(f'\n----------Score in test set: \n Condition acc: {Condition_acc}, mae: {RUL_mae}, r2: {RUL_r_square}, rmse: {RUL_mean_squared_error}\n' )

if __name__ == '__main__':
  opt = parse_opt()
  print(f"Current training bearing: {opt.train_bearing}") 
  print(f"Current test bearing: {opt.test_bearing}\n") 

  # start_save_data(opt)
  if opt.type == 'PHM' and opt.case == 'case1':
    from utils.load_PHM_data import train_1D, train_2D, train_extract, train_label_RUL,\
                                    test_1D, test_2D, test_extract, test_label_RUL
    main_PHM(opt, train_1D, train_2D, train_extract, train_label_RUL, test_1D, test_2D, test_extract, test_label_RUL, opt.only_test)
  elif opt.type == 'PHM' and opt.case == 'case2':
    from utils.load_PHM_data import train_1D, train_2D, train_extract, train_label_Con, train_label_RUL,\
                                    test_1D, test_2D, test_extract, test_label_Con, test_label_RUL
    main_XJTU(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con, opt.only_RUL, opt.only_test)
  else:
    from utils.load_XJTU_data import train_1D, train_2D, train_extract, train_label_Con, train_label_RUL,\
                                     test_1D, test_2D, test_extract, test_label_Con, test_label_RUL
    main_XJTU(opt, train_1D, train_2D, train_extract, train_label_RUL, train_label_Con, test_1D, test_2D, test_extract, test_label_RUL, test_label_Con, opt.only_RUL, opt.only_test)