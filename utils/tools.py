import numpy as np
import os
from keras import backend as K
import pandas as pd
import pickle as pkl
from numpy import save
from numpy import load
from sklearn.decomposition import PCA
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from utils.extract_features import extracted_feature_of_signal
from sklearn.metrics import r2_score, accuracy_score
from os import path
from os.path import join, exists
import scipy
import statistics
from sklearn.cluster import SpectralClustering
from model.autoencoder import autoencoder_model
from scipy.stats import kurtosis

    
def hankel_svdvals(data, hankel_window_size, slice_window_size):
    """ Slices data in 'slice_window_size' and compute hankel matrix singular values with 'hankel_window_size' """

    n_slices = len(data)//slice_window_size
    hankel_svd = []

    for i in range(n_slices):
        sample_data = data[slice_window_size*i : slice_window_size*(i+1)]
        c = sample_data[0: hankel_window_size]
        r = sample_data[hankel_window_size - 1: ]
        h = scipy.linalg.hankel(c, r)

        hankel_svd.append(scipy.linalg.svdvals(h))

    return hankel_svd

def correlation_coeffs(data, baseline, norm_interval, filter_window_size, filter_polyorder):
    """ Normalizes data, select baseline data and compute the correlation coefficients """

    a, b = norm_interval
    diff = b - a

    MIN = min([min(x) for x in data])
    MAX = max([max(x) for x in data])
    DIFF = MAX - MIN

    data_norm = [diff*(x - MIN)/(DIFF) + a for x in data]
    
    x = data_norm[baseline]
    
    R = [sum(np.multiply(x, y))/(np.sqrt(sum(x**2)*sum(y**2))) for y in data_norm]

    # Passing savgol filter
    return scipy.signal.savgol_filter(R, filter_window_size, filter_polyorder)

#----------------------#### General ####------------------------------------------------
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
def accuracy_m(y_true, y_pred):
  correct = 0
  total = 0
  for i in range(len(y_true)):
      act_label = np.argmax(y_true[i]) # act_label = 1 (index)
      pred_label = np.argmax(y_pred[i]) # pred_label = 1 (index)
      if(act_label == pred_label):
          correct += 1
      total += 1
  accuracy = (correct/total)
  return accuracy

def to_onehot(label, m_po=3):
  new_label = np.zeros((len(label), m_po))
  for idx, i in enumerate(label):
    new_label[idx][int(i-1.)] = 1.
  return new_label

def back_onehot(label):
  a = []
  for i in label:
    a.append(np.argmax(i)+1)
  return np.array(a)

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_pred - K.mean(y_true) ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return SS_res/SS_tot 

def r2_numpy(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  np.sum(( y_pred - np.mean(y_true) )**2)
    SS_tot = np.sum(( y_true - np.mean(y_true) )**2)
    return SS_res/SS_tot 

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def rmse(y_true, y_pred):
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())

def all_matric_PHM(y_true_rul, y_pred_rul):
    y_true_rul = np.squeeze(y_true_rul)
    y_pred_rul = np.squeeze(y_pred_rul)
    
    r2 = r2_score(y_true_rul, y_pred_rul)
    mae_ = mae(y_true_rul, y_pred_rul)
    rmse_ = rmse(y_true_rul, y_pred_rul)
    return r2, mae_, rmse_

def all_matric_XJTU(y_true_rul, y_pred_rul, y_true_con, y_pred_con):
    y_true_rul = np.squeeze(y_true_rul)
    y_pred_rul = np.squeeze(y_pred_rul)
    y_true_con = np.squeeze(y_true_con)
    y_pred_con = np.squeeze(y_pred_con)
    
    acc = accuracy_score(y_true_con, y_pred_con)
    r2 = r2_score(y_true_rul, y_pred_rul)
    mae_ = mae(y_true_rul, y_pred_rul)
    rmse_ = rmse(y_true_rul, y_pred_rul)
    return r2, mae_, rmse_, acc
    
#----------------------save_data.py------------------------------------------------
def load_file(path, save_path):
  a = [i for i in os.listdir(path)]
  all_file = []
  for i in range(len(a)):
    name = str(i)+'.csv'
    root = join(path, name)
    if exists(root):
      df=pd.read_csv(root, header=None, names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'])
      df = np.expand_dims(np.array(df)[1:], axis=0).astype(np.float32)
      if all_file == []:
        all_file = df
      else:
        all_file = np.concatenate((all_file, df))
  save_df(all_file, save_path)

def read_data_as_df(base_dir):
  '''
  saves each file in the base_dir as a df and concatenate all dfs into one
  '''
  if base_dir[-1]!='/':
    base_dir += '/'

  dfs=[]
  for f in sorted(os.listdir(base_dir)):
    if f[:3] == 'acc':
      df=pd.read_csv(base_dir+f, header=None, names=['hour', 'minute', 'second', 'microsecond', 'horiz accel', 'vert accel'])
      dfs.append(df)
  return pd.concat(dfs)

def process(base_dir, out_file):
  '''
  dumps combined dataframes into pkz (pickle) files for faster retreival
  '''
  df=read_data_as_df(base_dir)
  # assert df.shape[0]==len(os.listdir(base_dir))*DATA_POINTS_PER_FILE
  with open(out_file, 'wb') as pfile:
    pkl.dump(df, pfile)
  print('{0} saved'.format(out_file))
  print(f'Shape: {df.shape}\n')

#----------------------Load_data.py------------------------------------------------
def load_df(pkz_file):
    df = load(pkz_file)
    return df

def save_df(out_file, df):
  save(out_file, df)

def scaler(signal, scale_method):
  scale = scale_method().fit(signal)
  return scale.transform(signal), scale

def scaler_transform(signals, scale_method):
  data = []
  scale = scale_method()
  for signal in signals:
    if len(signal.shape) < 2:
      signal = np.expand_dims(signal, axis=-1)
    data.append(scale.fit_transform(signal))
  return np.array(data)

def extract_feature_image(df, opt, type_data, feature_name='horiz accel', type_=None):
    WAVELET_TYPE = 'morl'

    if type_ == 'PHM':
      DATA_POINTS_PER_FILE=2560
      if feature_name == 'horiz accel':
          data = df[4]
      else:
          data = df[5]
    else:
      DATA_POINTS_PER_FILE=32768
      if feature_name == 'Horizontal_vibration_signals':
          data = df[:, 0].astype(np.float32)
      else:
          data = df[:, 1].astype(np.float32)

    WIN_SIZE = DATA_POINTS_PER_FILE//128
    
    if type_data == '2d':
        data = np.array([np.mean(data[i: i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])
        coef, _ = pywt.cwt(data, np.linspace(1, 128, 128), WAVELET_TYPE)
        # transform to power and apply logarithm?!
        coef = np.log2(coef**2 + 0.001)
        coef = (coef - coef.min())/(coef.max() - coef.min())
        coef = np.expand_dims(coef, axis=-1)
    if type_data=='extract' or type_data=='1d':
        coef = np.expand_dims(data, axis=-1)
    return coef

def denoise(signals):
    all_signal = []
    for x in signals:
        L1, L2, L3 = pywt.wavedec(x, 'coif7', level=2)
        all_ = np.expand_dims(np.concatenate((L1, L2, L3)), axis=0)
        if all_signal == []:
          all_signal = all_
        else:
          all_signal = np.concatenate((all_signal, all_))
        # all_signal.append(nr.reduce_noise(y=x, sr=2559, hop_length=20, time_constant_s=0.1, prop_decrease=0.5, freq_mask_smooth_hz=25600))
    return np.expand_dims(all_signal, axis=-1)

def compute_PCA(x):
  pca = PCA(n_components=1)
  pca.fit(x)
  return pca.singular_values_[0]

def convert_to_image(name_bearing, opt, type_data, time=None, type_=None):
    '''
    This function is to get data and label from FPT points to later
    name_bearing: bearing name
    type_data: 1d, 2d, extract
    time: FPT
    type_: PHM, XJTU
    '''
    data = {'x': [], 'y': []}
    # if type_data == '2d':
    #   print('-'*10, f'Convert to 2D data', '-'*10, '\n')
    # else:
    #   print('-'*10, f'Maintain 1D data', '-'*10, '\n')
    
    num_files = len([i for i in os.listdir(name_bearing)])
    if opt.encoder_train and type_data != '2d':
      model = autoencoder_model(type_)
      EC_PHM_path = join(opt.save_dir, f'{type_}.h5')
      model.load_weights(EC_PHM_path)
      
    if type_ == 'PHM':
      for i in range(num_files):
          name = f"acc_{str(i+1).zfill(5)}.csv"
          file_ = join(name_bearing, name)
          if path.exists(file_):
              df = pd.read_csv(file_, header=None)
              coef_h = extract_feature_image(df, opt, type_data, feature_name='horiz accel', type_=type_)
              coef_v = extract_feature_image(df, opt, type_data, feature_name='vert accel', type_=type_)
              if (opt.noise_amplitude != 0):
                 coef_h += np.random.normal(0, opt.noise_amplitude, len(coef_h))
                 coef_v += np.random.normal(0, opt.noise_amplitude, len(coef_v))
              x_ = np.concatenate((coef_h, coef_v), axis=-1)

              if type_data == '1d' and opt.PCAlabel == True:
                data['y'].append(compute_PCA(x_))

              if opt.encoder_train and type_data != '2d':
                x_ = np.expand_dims(x_.reshape(x_.shape[1], x_.shape[0]), axis=0)
                raw_signal_filter = np.where(x_>0, 1, -1)
                x_ = raw_signal_filter * model.predict(x_*raw_signal_filter, verbose = 0, batch_size = 32)
                x_ = np.squeeze(x_)
                x_ = x_.reshape(x_.shape[1], x_.shape[0])

              x_ = x_.tolist()
              data['x'].append(x_)
    else:
      if opt.encoder_train:
        print('\n' + '#'*10 + 'USING ENCODE' + '\n' + '#'*10)
        model = autoencoder_model(type_)
        EC_XJTU_path = join(opt.save_dir, f'{type_}.h5')
        model.load_weights(EC_XJTU_path)
      for i in range(num_files):
          name = f"{str(i+1)}.csv"
          file_ = join(name_bearing, name)
          if path.exists(file_):
              df = np.array(pd.read_csv(file_, header=None))[1:]
              coef_h = extract_feature_image(df, opt, type_data, feature_name='Horizontal_vibration_signals', type_=type_)
              coef_v = extract_feature_image(df, opt, type_data, feature_name='Vertical_vibration_signals', type_=type_)
              x_ = np.concatenate((coef_h, coef_v), axis=-1)

              if type_data == '1d' and opt.PCAlabel == True:
                data['y'].append(compute_PCA(x_))

              if opt.encoder_train and type_data != '2d':
                x_ = np.expand_dims(x_.reshape(x_.shape[1], x_.shape[0]), axis=0)
                raw_signal_filter = np.where(x_>0, 1, -1)
                x_ = raw_signal_filter * model.predict(x_*raw_signal_filter, verbose = 0, batch_size = 32)
                x_ = np.squeeze(x_)
                x_ = x_.reshape(x_.shape[1], x_.shape[0])

              x_ = x_.tolist()
              data['x'].append(x_)
    
    data['x'] = np.array(data['x'])
    
    ################ Create linear label based on FPT points #########################
    if opt.PCAlabel == False:
      t_label = np.linspace(1, 0, len(data['x'][time: ]))
      data['y'] = t_label
      nameB, shapeB = name_bearing.split('/')[-1], data['x'].shape
      # print(f'Original shape of {nameB} data: {shapeB}')
      t_data = data['x'][time: ]
      data['x'] = t_data
    else:
      if type_data == '1d':
        data['y'] = convert_1_to_0(np.array(data['y']))
        
    ############## 1D-data to extraction data #####################
    if type_data=='extract':
      # print('-'*10, 'Convert to Extracted data', '-'*10, '\n')
      hor_data = extracted_feature_of_signal(np.array(data['x'])[:, :, 0])
      ver_data = extracted_feature_of_signal(np.array(data['x'])[:, :, 1])
      data_x = np.concatenate((hor_data, ver_data), axis=-1)
      data['x'] = data_x
    
    ############### scale data ##############################
    if type_data=='1d' or type_data=='extract':
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
        print(f'\nUse scaler: {opt.scaler}--------------\n')

        if opt.scaler != None:
          hor_data = np.array(data['x'])[:, :, 0]
          ver_data = np.array(data['x'])[:, :, 1]
          # print('-'*10, f'Use scaler: {opt.scaler}', '-'*10, '\n')
          if opt.scaler == 'FFT':
            hor_data = np.expand_dims(FFT(hor_data), axis=-1)
            ver_data = np.expand_dims(FFT(ver_data), axis=-1)
          elif opt.scaler == 'denoise':
            hor_data = denoise(hor_data)
            ver_data = denoise(ver_data)
          else:
            hor_data = scaler_transform(hor_data, scaler)
            ver_data = scaler_transform(ver_data, scaler)
          data_x = np.concatenate((hor_data, ver_data), axis=-1)
          data['x'] = data_x
        else:
          # print('-'*10, 'Raw data', '-'*10, '\n')
          data['x'] = np.array(data['x'])
    #     data['y'], _ = fit_values(2.31e-5, 0.99, 1.10, 1.68e-93, 28.58, np.array(data['y']))

    x_shape = data['x'].shape
    y_shape = np.array(data['y']).shape
    print(f'Train data shape: {x_shape}   Train label shape: {y_shape}')
    return data

def FFT(signals):
  fft_data = []
  for signal in signals:
    signal = np.fft.fft(signal)
    signal = np.abs(signal) / len(signal)
    signal = signal[range(signal.shape[0] // 2)]
    signal = np.expand_dims(signal, axis=0)
    if fft_data == []:
      fft_data = signal
    else:
      fft_data = np.concatenate((fft_data, signal))
  return fft_data

# ---------------------- Load_predict_data.py----------------------
def seg_data(data, length):
  all_data = {}
  num=0
  for name in length:
    all_data[name] = data[num: num+length[name]]
    num += length[name]
  return all_data

def percent_error(y_true, y_pred):
    y_pred = y_pred.reshape(-1, )
    E = 100.*(y_true - y_pred)/y_true
    E = E.astype(np.float32)
    A = []
    for i in E:
        if i <= 0.:
            A.append(np.exp(-np.log(0.5)*(i/5.)))
        else:
            A.append(np.exp(np.log(0.5)*(i/20.)))
    score = []
    for i in range(1, 12):
        score.append(A[i])
    return np.mean(score)

# ----------------------Creating label----------------------

def fit_values(k, b, Y, M, B, rrms):  # fit_values(2.31e-5, 0.99, 1.10, 1.68e-93, 28.58, rrms_1)
    y_array = []
    x = []
    for i in range(len(rrms)):
        x.append(i)
        y = Y + M*pow(i,B)
        if(y < 1.1):
            y = k*x[i] + b
        y_array.append(y)
    return y_array, x

def gen_rms(col):
    return np.squeeze(np.sqrt(np.mean(col**2)))

def convert_1_to_0(data, _1_to_0=False):
    if np.min(data) != np.max(data):
      f_data = (data - np.min(data))/(np.max(data) - np.min(data))
    else:
      f_data = np.ones_like(data)
    f_data = 1-f_data if _1_to_0 else f_data
    return 1-f_data

def predict_time_2(data, length_seg=None):
  h = []
  for i in data[:, :, 0]:
    h.append(gen_rms(i))
  h0 = convert_1_to_0(h)
#   length_seg = 50
  num_seg = len(h0)//length_seg
  h_seg = []
  for i in range(num_seg):
    h_seg.append(h0[i: i+length_seg])
  h_seg = np.array(h_seg)

  # Apply clustering learning model ---------------------------------
  kmeans_1 = SpectralClustering(3, n_init=100, assign_labels='discretize').fit(h_seg) # https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation
  time = 0
  type_all = np.array(kmeans_1.labels_)
  type_normal = type_all[0]
  for idx, i in enumerate(type_all):
    if i != type_normal:
      break
    time = idx

  normal_time = (time+1)*length_seg
  return normal_time

def percent_error(y_true, y_pred):
    y_pred = y_pred.reshape(-1, )
    E = 100.*(y_true - y_pred)/y_true
    E = E.astype(np.float32)
    SD = statistics.stdev(E.tolist())
    A = []
    for i in E:
        i = i/100.
        if i <= 0.:
            A.append(np.exp(-np.log(0.5)*(i/5.)))
        else:
            A.append(np.exp(np.log(0.5)*(i/20.)))
    return np.mean(A), SD

def getting_data(saved_dir, bearing_list, opt, get_index=False):
    _1D_data = []
    _2D_data = []
    extract_data = []

    # Creating empty folders to catch labels--------------
    if opt.type == 'PHM' and opt.case == 'case1':
        RUL_labels = []
        state = 0
    else:
        RUL_labels = []
        Con_labels = []
        state = 1

    if get_index:
        idx = {}

    # Arranging data and labels in scheme---------------
    for name in bearing_list:
        for data_type in opt.data_type:
            # Loading data and labels-----------------------
            data_file = f"{name}_data_{data_type}.npy"
            label_file = f"{name}_label_RUL.npy"

            data = load_df(join(saved_dir, data_file))
            RUL_label = load_df(join(saved_dir, label_file))

            if get_index:
                idx[name] = RUL_label.shape[0]

            if state == 1:
                Con_label = load_df(join(saved_dir, label_file))

            # Getting 1D data and labels-----------------------------------
            if data_type == '1d':
                if len(_1D_data) == 0:
                    _1D_data = data
                    RUL_labels = RUL_label
                    if state == 1:
                        Con_labels = Con_label
                else:
                    _1D_data = np.concatenate((_1D_data, data))
                    RUL_labels = np.concatenate((RUL_labels, RUL_label))
                    if state == 1:
                        Con_labels = np.concatenate((Con_labels, Con_label))

            # Getting 2D data --------------------------------------------
            elif data_type == '2d':
                if len(_2D_data) == 0:
                    _2D_data = data
                else:
                    _2D_data = np.concatenate((_2D_data, data))

            # Getting extract data ---------------------------------------
            else:
                if len(extract_data) == 0:
                    extract_data = data
                else:
                    extract_data = np.concatenate((extract_data, data))

    if opt.type == 'PHM' and opt.case == 'case1':
        if get_index:
            return _1D_data, _2D_data, extract_data, RUL_labels, idx
        else:
            return _1D_data, _2D_data, extract_data, RUL_labels
    else:
        if get_index:
            return _1D_data, _2D_data, extract_data, RUL_labels, Con_labels, idx
        else:
            return _1D_data, _2D_data, extract_data, RUL_labels, Con_labels

def predict_time(data_l, nor = 20):
  '''
  INPUT =========================================
  data: - type: float32 (should be)
        - shape: 1D (for example: (1802, ))
  nor: length of normal data

  OUTPUT =======================================
  fpt: - type: int32 (should be)
       - shape: a interger value in N 
  '''
  data = []
  for i in data_l:
    data.append(kurtosis(i[:, 0]))
  # data, _ = fit_values(2.41e-5, 1.01, 1.08, 3.22e-86, 26.30, data)
  # data = data/np.mean(data[:300])
  # data = savgol_filter(data, 15, 2)
  normal = data[:nor]
  mean_normal = np.mean(normal)
  std_normal = np.std(normal)

  limit_up = mean_normal + 3*std_normal
  limit_down = mean_normal - 3*std_normal

  fpt = 0
  i_n = 0
  check_c = 1
  for idx, i in enumerate(data):
    if i > limit_up or i < limit_down:
      if idx - check_c == 1:
        i_n += 1
      else:
        i_n = 0
      check_c = idx

      if i_n == 2:
        fpt = idx-2
        break
  return fpt