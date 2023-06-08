from os.path import join, exists
import numpy as np
import os
from train import parse_opt
from utils.tools import  save_df, convert_to_image, getting_data
from utils.train_encoder import train_EC

opt = parse_opt()
np.random.seed(1234)

# Link of original data ==================================================================================
train_dir = join(opt.main_dir_colab, 'PHM_data/Learning_set')
test_dir = join(opt.main_dir_colab, 'PHM_data/Test_set')
saved_dir = join(opt.main_dir_colab, 'PHM_data/saved_data')

if opt.case == 'case1':
  # FPT points of bearing sets ==================================================================================
  FPT = {'Bearing1_1': 1314,
        'Bearing1_2': 826,
        'Bearing1_3': 1726,
        'Bearing1_4': 1082,
        'Bearing1_5': 2412,
        'Bearing1_6': 1631,
        'Bearing1_7': 2210}

  # Train for encoder model ==================================================================================
  if opt.encoder:
    EC_PHM_path = join(opt.save_dir, 'PHM.h5')
    if exists(EC_PHM_path) == False:
      # Load saved bearing data ==================================================================================
      test_1D, test_2D, test_extract, test_label_RUL = getting_data(saved_dir, opt.test_bearing, opt)
      train_1D, train_2D, train_extract, train_label_RUL = getting_data(saved_dir, opt.train_bearing, opt)
      s_0, s_1, s_2 = train_1D.shape
      train_1D = train_1D.reshape((s_0, s_2, s_1))
      train_1D_filter = np.where(train_1D>0, 1, -1)
      train_EC(train_1D*train_1D_filter, 'PHM', EC_PHM_path, opt)


  if exists(join(saved_dir, 'Bearing2_1_data_1d.npy')) == False:
    for type_data in opt.data_type:
      # Converting data-------------------------------------------------------------------------
      print(f'\n Saving data in {opt.type} data set'+'-'*100)
      Bearing1_1 = convert_to_image(join(train_dir, 'Bearing1_1'), opt, type_data, FPT['Bearing1_1'], 'PHM')
      save_df(join(saved_dir, 'Bearing1_1_data_' + type_data + '.npy'), Bearing1_1['x'])
      save_df(join(saved_dir, 'Bearing1_1_label_RUL.npy'), Bearing1_1['y'])

      Bearing1_2 = convert_to_image(join(train_dir, 'Bearing1_2'), opt, type_data, FPT['Bearing1_2'], 'PHM')
      save_df(join(saved_dir, 'Bearing1_2_data_' + type_data + '.npy'), Bearing1_2['x'])
      save_df(join(saved_dir, 'Bearing1_2_label_RUL.npy'), Bearing1_2['y'])

      Bearing1_3 = convert_to_image(join(test_dir,  'Bearing1_3'), opt, type_data, FPT['Bearing1_3'], 'PHM')
      save_df(join(saved_dir, 'Bearing1_3_data_' + type_data + '.npy'), Bearing1_3['x'])
      save_df(join(saved_dir, 'Bearing1_3_label_RUL.npy'), Bearing1_3['y'])

      Bearing1_4 = convert_to_image(join(test_dir,  'Bearing1_4'), opt, type_data, FPT['Bearing1_4'], 'PHM')
      save_df(join(saved_dir, 'Bearing1_4_data_' + type_data + '.npy'), Bearing1_4['x'])
      save_df(join(saved_dir, 'Bearing1_4_label_RUL.npy'), Bearing1_4['y'])

      Bearing1_5 = convert_to_image(join(test_dir,  'Bearing1_5'), opt, type_data, FPT['Bearing1_5'], 'PHM')
      save_df(join(saved_dir, 'Bearing1_5_data_' + type_data + '.npy'), Bearing1_5['x'])
      save_df(join(saved_dir, 'Bearing1_5_label_RUL.npy'), Bearing1_5['y'])

      Bearing1_6 = convert_to_image(join(test_dir,  'Bearing1_6'), opt, type_data, FPT['Bearing1_6'], 'PHM')
      save_df(join(saved_dir, 'Bearing1_6_data_' + type_data + '.npy'), Bearing1_6['x'])
      save_df(join(saved_dir, 'Bearing1_6_label_RUL.npy'), Bearing1_6['y'])

      Bearing1_7 = convert_to_image(join(test_dir,  'Bearing1_7'), opt, type_data, FPT['Bearing1_7'], 'PHM')
      save_df(join(saved_dir, 'Bearing1_7_data_' + type_data + '.npy'), Bearing1_7['x'])
      save_df(join(saved_dir, 'Bearing1_7_label_RUL.npy'), Bearing1_7['y'])

  # Load saved bearing data ==================================================================================
  test_1D, test_2D, test_extract, test_label_RUL, test_idx = getting_data(saved_dir, opt.test_bearing, opt, get_index=True)
  train_1D, train_2D, train_extract, train_label_RUL = getting_data(saved_dir, opt.train_bearing, opt)
else:
  # FPT points of bearing sets ==================================================================================
  FPT = {'Bearing1_1': 500,
        'Bearing1_2': 66,
        'Bearing1_3': 573,
        'Bearing1_4': 40,
        'Bearing1_5': 161,
        'Bearing1_6': 146,
        'Bearing1_7': 757,
        'Bearing2_1': 32,
        'Bearing2_2': 249,
        'Bearing2_3': 753,
        'Bearing2_4': 139,
        'Bearing2_5': 309,
        'Bearing2_6': 129,
        'Bearing2_7': 58,
        'Bearing3_1': 67,
        'Bearing3_2': 133,
        'Bearing3_3': 82}

  # Train for encoder model ==================================================================================
  if opt.encoder:
    EC_PHM_path = join(opt.save_dir, 'PHM.h5')
    if exists(EC_PHM_path) == False:
      # Load saved bearing data ==================================================================================
      train_1D, train_2D, train_extract, train_label_RUL, train_label_Con = getting_data(saved_dir, opt.train_bearing, opt)
      s_0, s_1, s_2 = train_1D.shape
      train_1D = train_1D.reshape((s_0, s_2, s_1))
      train_1D_filter = np.where(train_1D>0, 1, -1)
      train_EC(train_1D*train_1D_filter, 'PHM', EC_PHM_path, opt)


  if exists(join(saved_dir, 'Bearing2_1_data_1d.npy')) == False:
    for type_data in opt.data_type:
      # Converting data-------------------------------------------------------------------------
      print(f'\n Saving data in {opt.type} data set'+'-'*100)
      Bearing1_1 = convert_to_image(join(train_dir, 'Bearing1_1'), opt, type_data, FPT['Bearing1_1'], 'PHM')
      Bearing1_1_label_Con = np.array([1.]*len(Bearing1_1['x']))
      save_df(join(saved_dir, 'Bearing1_1_data_' + type_data + '.npy'), Bearing1_1['x'])
      save_df(join(saved_dir, 'Bearing1_1_label_RUL.npy'), Bearing1_1['y'])
      save_df(join(saved_dir, 'Bearing1_1_label_Con.npy') , Bearing1_1_label_Con)

      Bearing1_2 = convert_to_image(join(train_dir, 'Bearing1_2'), opt, type_data, FPT['Bearing1_2'], 'PHM')
      Bearing1_2_label_Con = np.array([1.]*len(Bearing1_2['x']))
      save_df(join(saved_dir, 'Bearing1_2_data_' + type_data + '.npy'), Bearing1_2['x'])
      save_df(join(saved_dir, 'Bearing1_2_label_RUL.npy'), Bearing1_2['y'])
      save_df(join(saved_dir, 'Bearing1_2_label_Con.npy') , Bearing1_2_label_Con)

      Bearing1_3 = convert_to_image(join(test_dir,  'Bearing1_3'), opt, type_data, FPT['Bearing1_3'], 'PHM')
      Bearing1_3_label_Con = np.array([1.]*len(Bearing1_3['x']))
      save_df(join(saved_dir, 'Bearing1_3_data_' + type_data + '.npy'), Bearing1_3['x'])
      save_df(join(saved_dir, 'Bearing1_3_label_RUL.npy'), Bearing1_3['y'])
      save_df(join(saved_dir, 'Bearing1_3_label_Con.npy') , Bearing1_3_label_Con)

      Bearing1_4 = convert_to_image(join(test_dir,  'Bearing1_4'), opt, type_data, FPT['Bearing1_4'], 'PHM')
      Bearing1_4_label_Con = np.array([1.]*len(Bearing1_4['x']))
      save_df(join(saved_dir, 'Bearing1_4_data_' + type_data + '.npy'), Bearing1_4['x'])
      save_df(join(saved_dir, 'Bearing1_4_label_RUL.npy'), Bearing1_4['y'])
      save_df(join(saved_dir, 'Bearing1_4_label_Con.npy') , Bearing1_4_label_Con)

      Bearing1_5 = convert_to_image(join(test_dir,  'Bearing1_5'), opt, type_data, FPT['Bearing1_5'], 'PHM')
      Bearing1_5_label_Con = np.array([1.]*len(Bearing1_5['x']))
      save_df(join(saved_dir, 'Bearing1_5_data_' + type_data + '.npy'), Bearing1_5['x'])
      save_df(join(saved_dir, 'Bearing1_5_label_RUL.npy'), Bearing1_5['y'])
      save_df(join(saved_dir, 'Bearing1_5_label_Con.npy') , Bearing1_5_label_Con)

      Bearing1_6 = convert_to_image(join(test_dir,  'Bearing1_6'), opt, type_data, FPT['Bearing1_6'], 'PHM')
      Bearing1_6_label_Con = np.array([1.]*len(Bearing1_6['x']))
      save_df(join(saved_dir, 'Bearing1_6_data_' + type_data + '.npy'), Bearing1_6['x'])
      save_df(join(saved_dir, 'Bearing1_6_label_RUL.npy'), Bearing1_6['y'])
      save_df(join(saved_dir, 'Bearing1_6_label_Con.npy') , Bearing1_6_label_Con)

      Bearing1_7 = convert_to_image(join(test_dir,  'Bearing1_7'), opt, type_data, FPT['Bearing1_7'], 'PHM')
      Bearing1_7_label_Con = np.array([1.]*len(Bearing1_7['x']))
      save_df(join(saved_dir, 'Bearing1_7_data_' + type_data + '.npy'), Bearing1_7['x'])
      save_df(join(saved_dir, 'Bearing1_7_label_RUL.npy'), Bearing1_7['y'])
      save_df(join(saved_dir, 'Bearing1_7_label_Con.npy') , Bearing1_7_label_Con)

      Bearing2_1 = convert_to_image(join(train_dir, 'Bearing2_1'), opt, type_data, FPT['Bearing2_1'], 'PHM')
      Bearing2_1_label_Con = np.array([2.]*len(Bearing2_1['x']))
      save_df(join(saved_dir, 'Bearing2_1_data_' + type_data + '.npy'), Bearing2_1['x'])
      save_df(join(saved_dir, 'Bearing2_1_label_RUL.npy'), Bearing2_1['y'])
      save_df(join(saved_dir, 'Bearing2_1_label_Con.npy') , Bearing2_1_label_Con)

      Bearing2_2 = convert_to_image(join(train_dir, 'Bearing2_2'), opt, type_data, FPT['Bearing2_2'], 'PHM')
      Bearing2_2_label_Con = np.array([2.]*len(Bearing2_2['x']))
      save_df(join(saved_dir, 'Bearing2_2_data_' + type_data + '.npy'), Bearing2_2['x'])
      save_df(join(saved_dir, 'Bearing2_2_label_RUL.npy'), Bearing2_2['y'])
      save_df(join(saved_dir, 'Bearing2_2_label_Con.npy') , Bearing2_2_label_Con)

      Bearing2_3 = convert_to_image(join(test_dir,  'Bearing2_3'), opt, type_data, FPT['Bearing2_3'], 'PHM')
      Bearing2_3_label_Con = np.array([2.]*len(Bearing2_3['x']))
      save_df(join(saved_dir, 'Bearing2_3_data_' + type_data + '.npy'), Bearing2_3['x'])
      save_df(join(saved_dir, 'Bearing2_3_label_RUL.npy'), Bearing2_3['y'])
      save_df(join(saved_dir, 'Bearing2_3_label_Con.npy') , Bearing2_3_label_Con)

      Bearing2_4 = convert_to_image(join(test_dir,  'Bearing2_4'), opt, type_data, FPT['Bearing2_4'], 'PHM')
      Bearing2_4_label_Con = np.array([2.]*len(Bearing2_4['x']))
      save_df(join(saved_dir, 'Bearing2_4_data_' + type_data + '.npy'), Bearing2_4['x'])
      save_df(join(saved_dir, 'Bearing2_4_label_RUL.npy'), Bearing2_4['y'])
      save_df(join(saved_dir, 'Bearing2_4_label_Con.npy') , Bearing2_4_label_Con)

      Bearing2_5 = convert_to_image(join(test_dir,  'Bearing2_5'), opt, type_data, FPT['Bearing2_5'], 'PHM')
      Bearing2_5_label_Con = np.array([2.]*len(Bearing2_5['x']))
      save_df(join(saved_dir, 'Bearing2_5_data_' + type_data + '.npy'), Bearing2_5['x'])
      save_df(join(saved_dir, 'Bearing2_5_label_RUL.npy'), Bearing2_5['y'])
      save_df(join(saved_dir, 'Bearing2_5_label_Con.npy') , Bearing2_5_label_Con)

      Bearing2_6 = convert_to_image(join(test_dir,  'Bearing2_6'), opt, type_data, FPT['Bearing2_6'], 'PHM')
      Bearing2_6_label_Con = np.array([2.]*len(Bearing2_6['x']))
      save_df(join(saved_dir, 'Bearing2_6_data_' + type_data + '.npy'), Bearing2_6['x'])
      save_df(join(saved_dir, 'Bearing2_6_label_RUL.npy'), Bearing2_6['y'])
      save_df(join(saved_dir, 'Bearing2_6_label_Con.npy') , Bearing2_6_label_Con)

      Bearing2_7 = convert_to_image(join(test_dir,  'Bearing2_7'), opt, type_data, FPT['Bearing2_7'], 'PHM')
      Bearing2_7_label_Con = np.array([2.]*len(Bearing2_7['x']))
      save_df(join(saved_dir, 'Bearing2_7_data_' + type_data + '.npy'), Bearing2_7['x'])
      save_df(join(saved_dir, 'Bearing2_7_label_RUL.npy'), Bearing2_7['y'])
      save_df(join(saved_dir, 'Bearing2_7_label_Con.npy') , Bearing2_7_label_Con)

      Bearing3_1 = convert_to_image(join(train_dir, 'Bearing3_1'), opt, type_data, FPT['Bearing3_1'], 'PHM')
      Bearing3_1_label_Con = np.array([3.]*len(Bearing3_1['x']))
      save_df(join(saved_dir, 'Bearing3_1_data_' + type_data + '.npy'), Bearing3_1['x'])
      save_df(join(saved_dir, 'Bearing3_1_label_RUL.npy'), Bearing3_1['y'])
      save_df(join(saved_dir, 'Bearing3_1_label_Con.npy') , Bearing3_1_label_Con)

      Bearing3_2 = convert_to_image(join(train_dir, 'Bearing3_2'), opt, type_data, FPT['Bearing3_2'], 'PHM')
      Bearing3_2_label_Con = np.array([3.]*len(Bearing3_2['x']))
      save_df(join(saved_dir, 'Bearing3_2_data_' + type_data + '.npy'), Bearing3_2['x'])
      save_df(join(saved_dir, 'Bearing3_2_label_RUL.npy'), Bearing3_2['y']) 
      save_df(join(saved_dir, 'Bearing3_2_label_Con.npy') , Bearing3_2_label_Con)

      Bearing3_3 = convert_to_image(join(test_dir,  'Bearing3_3'), opt, type_data, FPT['Bearing3_3'], 'PHM')
      Bearing3_3_label_Con = np.array([3.]*len(Bearing3_3['x']))
      save_df(join(saved_dir, 'Bearing3_3_data_' + type_data + '.npy'), Bearing3_3['x'])
      save_df(join(saved_dir, 'Bearing3_3_label_RUL.npy'), Bearing3_3['y'])
      save_df(join(saved_dir, 'Bearing3_3_label_Con.npy') , Bearing3_3_label_Con)

  # Load saved bearing data ==================================================================================
  test_1D, test_2D, test_extract, test_label_RUL, test_label_Con, test_idx = getting_data(saved_dir, opt.test_bearing, opt, get_index=True)
  train_1D, train_2D, train_extract, train_label_RUL, train_label_Con = getting_data(saved_dir, opt.train_bearing, opt)

print('\n' + '#'*10 + f'Experimental case: {opt.case}'+ '#'*10 + '\n')
print(f'Shape of 1D training data: {train_1D.shape}')  
print(f'Shape of 1D test data: {test_1D.shape}\n')

print(f'Shape of 2D training data: {train_2D.shape}')  
print(f'Shape of 2D test data: {test_2D.shape}\n')

print(f'Shape of extract training data: {train_extract.shape}')  
print(f'Shape of extract test data: {test_extract.shape}\n')

print(f'Shape of training RUL label: {train_label_RUL.shape}')  
print(f'Shape of test RUL label: {test_label_RUL.shape}\n')