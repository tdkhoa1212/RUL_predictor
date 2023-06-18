from train import parse_opt
from model.resnet import resnet_101
from model.LSTM import lstm_extracted_model, lstm_model
from model.MIX_1D_2D import mix_model_PHM, mix_model_XJTU
from utils.tools import all_matric_XJTU, all_matric_PHM, back_onehot
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from model.autoencoder import autoencoder_model
from os.path import join
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from scipy.signal import savgol_filter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter("ignore")
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=130)

opt = parse_opt()

# Loading data ######################################################
if opt.type == 'PHM' and opt.case == 'case1':
  from utils.load_PHM_data import test_1D, test_2D, test_extract, test_label_RUL, test_idx
elif opt.type == 'PHM' and opt.case == 'case2':
  from utils.load_PHM_data import test_1D, test_2D, test_extract, test_label_Con, test_label_RUL, test_idx
else:
  from utils.load_XJTU_data import train_1D, train_2D, train_extract, train_label_Con, train_label_RUL,\
                                     test_1D, test_2D, test_extract, test_label_Con, test_label_RUL

def Predict(data, opt):
  # Loading model ############################################################
  input_extracted = Input((14, 2), name='Extracted_LSTM_input')
  input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
  input_2D = Input((128, 128, 2), name='CNN_input')

  if opt.type == 'PHM' and opt.case == 'case1':
    RUL = mix_model_PHM(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, False)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)
  else:
    Condition, RUL = mix_model_XJTU(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, False)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])

  # Loading weights ############################################################
  weight_path = os.path.join(opt.save_dir, f'model_{opt.condition}_{opt.type}')
  print(f'\nLoad weight: {weight_path}\n')
  network.load_weights(weight_path)
  
  # Prediction #####################################
  if opt.type == 'PHM' and opt.case == 'case1':
    RUL = network.predict(data)
    return RUL 
  else:
    Condition, RUL = network.predict(data)
    return Condition, RUL 

def main(opt):
  num = 0
  for name in opt.test_bearing:
    t_1D, t_2D, t_extract = test_1D[num: num+test_idx[name]], test_2D[num: num+test_idx[name]], test_extract[num: num+test_idx[name]]
    print(f'\nShape 1D of {name} data: {t_1D.shape}')
    print(f'Shape 2D of {name} data: {t_2D.shape}')

    if opt.type == 'PHM' and opt.case == 'case1':
      RUL = Predict([t_1D, t_2D, t_extract], opt)
    else:
      Condition, RUL = Predict([t_1D, t_2D, t_extract], opt)

    if opt.type == 'PHM' and opt.case == 'case1':
      t_label_RUL = test_label_RUL[num: num+test_idx[name]]
      num += test_idx[name]
      r2, mae_, mse_ = all_matric_PHM(t_label_RUL, RUL)
    else:
      Condition = back_onehot(Condition)
      t_label_Con = test_label_Con[num: num+test_idx[name]]
      t_label_RUL = test_label_RUL[num: num+test_idx[name]]
      num += test_idx[name]
      r2, mae_, mse_, acc = all_matric_XJTU(t_label_RUL, RUL, t_label_Con, Condition)
      acc = round(acc*100, 4)

    mae_ = round(mae_, 4)
    rmse_ = round(mse_, 4)
    r2 = round(r2, 4)

    if opt.type == 'PHM' and opt.case == 'case1':
      print(f'\n-----{name}:      R2: {r2}, MAE: {mae_}, RMSE: {rmse_}-----')
    else:
      print(f'\n-----{name}:      R2: {r2}, MAE: {mae_}, RMSE: {rmse_}, Acc: {acc}-----')

    # Simulating the graphs --------------------------------------------------------
    plt.plot(t_label_RUL, c='b', label='Actual RUL', linewidth=7.0)
    x_RUL = np.arange(RUL.shape[0])
    RUL = RUL.reshape(RUL.shape[0], )
    smoothed_RUL = savgol_filter(RUL, 8, 2)
    plt.plot(smoothed_RUL, c='r', label='Smoothed Prediction', linewidth=7.0)
    plt.scatter(x_RUL, RUL, c='orange', label='Raw Prediction', s=50)
    plt.title(opt.type + f' - {name}', fontsize=18)
    plt.legend()
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("RUL", fontsize=18)
    plt.savefig(join(opt.save_dir+'/images', opt.type, f'{name}.png'))
    plt.close()

def denoiseComparison(train_1D, train_2D, opt):
  raw_signal = train_1D[0].reshape(1, 2, 32768)
  raw_signal_filter = np.where(raw_signal>0, 1, -1)
  print(f"\nShape of raw signal: {raw_signal.shape}\n")
  EC_XJTU_path = join(opt.save_dir, f'XJTU.h5')
  model = autoencoder_model('XJTU')
  model.summary()
  model.load_weights(EC_XJTU_path)

  denoised_signal = raw_signal_filter * model.predict(raw_signal*raw_signal_filter)
  # Demonstrating signal-----------------
  x_raw = np.squeeze(raw_signal.reshape(32768, 2)[:, 0]) 
  x_denoised = np.squeeze(denoised_signal.reshape(32768, 2)[:, 0]) 
  plt.plot(x_raw[:200], c='black', label='Raw signal')
  plt.plot(x_denoised[:200], c='orange', linestyle='dashed', label='Denoised signal')
  plt.xlabel('Time')
  plt.ylabel('Magnitude')
  plt.legend()
  plt.savefig(join(opt.save_dir+'/images', f'compareSignals.png'))
  plt.show()
  
  image = np.squeeze(train_2D[0][:, :, 0])
  plt.imshow(image)
  plt.savefig(join(opt.save_dir+'/images', f'2DImage.png'))
  plt.show()
  
if __name__ == '__main__':
  warnings.filterwarnings("ignore", category=FutureWarning)
  main(opt)
  # denoiseComparison(train_1D, train_2D, opt)
