import os
import numpy as np
from train import parse_opt
from utils.tools import save_df, convert_to_image, getting_data
from utils.train_encoder import train_EC
from os.path import join, exists


opt = parse_opt()

main_dir_colab = join(opt.main_dir_colab, 'XJTU_data/XJTU-SY_Bearing_Datasets')
saved_dir = join(opt.main_dir_colab, 'XJTU_data/saved_data')

# FPT points of bearing sets ==================================================================================
if opt.case == 'case3':
  FPT = {'Bearing1_1': 76,
        'Bearing1_2': 44,
        'Bearing1_3': 60,
        'Bearing1_5': 39,
        'Bearing2_1': 453, # case 3 
        'Bearing2_2': 47, # case 3
        'Bearing2_3': 303, # case 3
        'Bearing2_4': 30, # case 3
        'Bearing2_5': 121, # case 3
        'Bearing3_1': 2344,
        'Bearing3_3': 340,
        'Bearing3_4': 1418,
        'Bearing3_5': 9}
else:
  FPT = {'Bearing1_1': 76,
          'Bearing1_2': 44,
          'Bearing1_3': 60,
          'Bearing1_5': 39,
          'Bearing2_1': 455,
          'Bearing2_2': 48,
          'Bearing2_3': 327,
          'Bearing2_4': 32,
          'Bearing2_5': 141,
          'Bearing3_1': 2344,
          'Bearing3_3': 340,
          'Bearing3_4': 1418,
          'Bearing3_5': 9}

if opt.case == 'case1' or opt.case == 'case3':
  if os.path.exists(join(saved_dir, 'Bearing2_4_data_5d.npy')) == True: # true: Bearing2_4_data_1d
    EC_XJTU_path = join(opt.save_dir, 'XJTU.h5')
    test_1D, test_2D, test_extract, test_label_RUL, test_label_Con = getting_data(saved_dir, opt.test_bearing, opt)
    train_1D, train_2D, train_extract, train_label_RUL, train_label_Con = getting_data(saved_dir, opt.train_bearing, opt)
    # s_0, s_1, s_2 = train_1D.shape
    # train_1D = train_1D.reshape((s_0, s_2, s_1))
    # train_1D_filter = np.where(train_1D>0, 1, -1)
    # train_EC(train_1D * train_1D_filter, 'XJTU', opt)

  else:
    for type_data in opt.data_type:
      print(f'\nProcessing type: {type_data}\n')
      # Train data-------------------------------------------------------------------------
      Bearing1_1_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_1')
      Bearing1_1 = convert_to_image(Bearing1_1_path, opt, type_data, FPT['Bearing1_1'], 'XJTU')
      Bearing1_1_label_Con = np.array([1.]*len(Bearing1_1['x']))
      save_df(join(saved_dir, 'Bearing1_1_data_' + type_data + '.npy'), Bearing1_1['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_1_label_RUL.npy') , Bearing1_1['y'])
        save_df(join(saved_dir, 'Bearing1_1_label_Con.npy') , Bearing1_1_label_Con)

      Bearing1_2_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_2')
      Bearing1_2 = convert_to_image(Bearing1_2_path, opt, type_data, FPT['Bearing1_2'], 'XJTU')
      Bearing1_2_label_Con = np.array([1.]*len(Bearing1_2['x']))
      save_df(join(saved_dir, 'Bearing1_2_data_' + type_data + '.npy'), Bearing1_2['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_2_label_RUL.npy') , Bearing1_2['y'])
        save_df(join(saved_dir, 'Bearing1_2_label_Con.npy') , Bearing1_2_label_Con)

      Bearing1_3_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_3')
      Bearing1_3 = convert_to_image(Bearing1_3_path, opt, type_data, FPT['Bearing1_3'], 'XJTU')
      Bearing1_3_label_Con = np.array([1.]*len(Bearing1_3['x']))
      save_df(join(saved_dir, 'Bearing1_3_data_' + type_data + '.npy'), Bearing1_3['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_3_label_RUL.npy') , Bearing1_3['y'])
        save_df(join(saved_dir, 'Bearing1_3_label_Con.npy') , Bearing1_3_label_Con)

      Bearing1_5_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_5')
      Bearing1_5 = convert_to_image(Bearing1_5_path, opt, type_data, FPT['Bearing1_5'], 'XJTU')
      Bearing1_5_label_Con = np.array([1.]*len(Bearing1_5['x']))
      save_df(join(saved_dir, 'Bearing1_5_data_' + type_data + '.npy'), Bearing1_5['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_5_label_RUL.npy') , Bearing1_5['y'])
        save_df(join(saved_dir, 'Bearing1_5_label_Con.npy') , Bearing1_5_label_Con)

      Bearing2_1_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_1')
      Bearing2_1 = convert_to_image(Bearing2_1_path, opt, type_data, FPT['Bearing2_1'], 'XJTU')
      Bearing2_1_label_Con = np.array([2.]*len(Bearing2_1['x']))
      save_df(join(saved_dir, 'Bearing2_1_data_' + type_data + '.npy'), Bearing2_1['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_1_label_RUL.npy') , Bearing2_1['y'])
        save_df(join(saved_dir, 'Bearing2_1_label_Con.npy') , Bearing2_1_label_Con)

      Bearing2_2_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_2')
      Bearing2_2 = convert_to_image(Bearing2_2_path, opt, type_data, FPT['Bearing2_2'], 'XJTU')
      Bearing2_2_label_Con = np.array([2.]*len(Bearing2_2['x']))
      save_df(join(saved_dir, 'Bearing2_2_data_' + type_data + '.npy'), Bearing2_2['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_2_label_RUL.npy') , Bearing2_2['y'])
        save_df(join(saved_dir, 'Bearing2_2_label_Con.npy') , Bearing2_2_label_Con)

      Bearing2_3_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_3')
      Bearing2_3 = convert_to_image(Bearing2_3_path, opt, type_data, FPT['Bearing2_3'], 'XJTU')
      Bearing2_3_label_Con = np.array([2.]*len(Bearing2_3['x']))
      save_df(join(saved_dir, 'Bearing2_3_data_' + type_data + '.npy'), Bearing2_3['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_3_label_RUL.npy') , Bearing2_3['y'])
        save_df(join(saved_dir, 'Bearing2_3_label_Con.npy') , Bearing2_3_label_Con)

      Bearing2_4_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_4')
      Bearing2_4 = convert_to_image(Bearing2_4_path, opt, type_data, FPT['Bearing2_4'], 'XJTU')
      Bearing2_4_label_Con = np.array([2.]*len(Bearing2_4['x']))
      save_df(join(saved_dir, 'Bearing2_4_data_' + type_data + '.npy'), Bearing2_4['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_4_label_RUL.npy') , Bearing2_4['y'])
        save_df(join(saved_dir, 'Bearing2_4_label_Con.npy') , Bearing2_4_label_Con)

      Bearing2_5_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_5')
      Bearing2_5 = convert_to_image(Bearing2_5_path, opt, type_data, FPT['Bearing2_5'], 'XJTU')
      Bearing2_5_label_Con = np.array([2.]*len(Bearing2_5['x']))
      save_df(join(saved_dir, 'Bearing2_5_data_' + type_data + '.npy'), Bearing2_5['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_5_label_RUL.npy') , Bearing2_5['y'])
        save_df(join(saved_dir, 'Bearing2_5_label_Con.npy') , Bearing2_5_label_Con)

      Bearing3_1_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_1')
      Bearing3_1 = convert_to_image(Bearing3_1_path, opt, type_data, FPT['Bearing3_1'], 'XJTU')
      Bearing3_1_label_Con = np.array([3.]*len(Bearing3_1['x']))
      save_df(join(saved_dir, 'Bearing3_1_data_' + type_data + '.npy'), Bearing3_1['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_1_label_RUL.npy') , Bearing3_1['y'])
        save_df(join(saved_dir, 'Bearing3_1_label_Con.npy') , Bearing3_1_label_Con)

      Bearing3_3_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_3')
      Bearing3_3 = convert_to_image(Bearing3_3_path, opt, type_data, FPT['Bearing3_3'], 'XJTU')
      Bearing3_3_label_Con = np.array([3.]*len(Bearing3_3['x']))
      save_df(join(saved_dir, 'Bearing3_3_data_' + type_data + '.npy'), Bearing3_3['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_3_label_RUL.npy') , Bearing3_3['y'])
        save_df(join(saved_dir, 'Bearing3_3_label_Con.npy') , Bearing3_3_label_Con)

      Bearing3_4_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_4')
      Bearing3_4 = convert_to_image(Bearing3_4_path, opt, type_data, FPT['Bearing3_4'], 'XJTU')
      Bearing3_4_label_Con = np.array([3.]*len(Bearing3_4['x']))
      save_df(join(saved_dir, 'Bearing3_4_data_' + type_data + '.npy'), Bearing3_4['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_4_label_RUL.npy') , Bearing3_4['y'])
        save_df(join(saved_dir, 'Bearing3_4_label_Con.npy') , Bearing3_4_label_Con)

      Bearing3_5_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_5')
      Bearing3_5 = convert_to_image(Bearing3_5_path, opt, type_data, FPT['Bearing3_5'], 'XJTU')
      Bearing3_5_label_Con = np.array([3.]*len(Bearing3_5['x']))
      save_df(join(saved_dir, 'Bearing3_5_data_' + type_data + '.npy'), Bearing3_5['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_5_label_RUL.npy') , Bearing3_5['y'])
        save_df(join(saved_dir, 'Bearing3_5_label_Con.npy') , Bearing3_5_label_Con)

      print('-'*100 + f'Saving data in {opt.type} data set'+'-'*100)
else:
  if os.path.exists(join(saved_dir, 'Bearing1_4_data_1d.npy')) == True :
    EC_XJTU_path = join(opt.save_dir, 'XJTU.h5')
    test_1D, test_2D, test_extract, test_label_RUL, test_label_Con = getting_data(saved_dir, opt.test_bearing, opt)
    train_1D, train_2D, train_extract, train_label_RUL, train_label_Con = getting_data(saved_dir, opt.train_bearing, opt)
    # s_0, s_1, s_2 = train_1D.shape
    # train_1D = train_1D.reshape((s_0, s_2, s_1))
    # train_1D_filter = np.where(train_1D>0, 1, -1)
    # train_EC(train_1D*train_1D_filter, 'XJTU', EC_XJTU_path, opt)

  # Saving the converted data ==================================================================================
  else:
    for type_data in opt.data_type:
      # Train data-------------------------------------------------------------------------
      Bearing1_1_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_1')
      Bearing1_1 = convert_to_image(Bearing1_1_path, opt, type_data, FPT['Bearing1_1'], 'XJTU')
      Bearing1_1_label_Con = np.array([1.]*len(Bearing1_1['x']))
      save_df(join(saved_dir, 'Bearing1_1_data_' + type_data + '.npy'), Bearing1_1['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_1_label_RUL.npy') , Bearing1_1['y'])
        save_df(join(saved_dir, 'Bearing1_1_label_Con.npy') , Bearing1_1_label_Con)

      Bearing1_2_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_2')
      Bearing1_2 = convert_to_image(Bearing1_2_path, opt, type_data, FPT['Bearing1_2'], 'XJTU')
      Bearing1_2_label_Con = np.array([1.]*len(Bearing1_2['x']))
      save_df(join(saved_dir, 'Bearing1_2_data_' + type_data + '.npy'), Bearing1_2['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_2_label_RUL.npy') , Bearing1_2['y'])
        save_df(join(saved_dir, 'Bearing1_2_label_Con.npy') , Bearing1_2_label_Con)

      Bearing1_3_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_3')
      Bearing1_3 = convert_to_image(Bearing1_3_path, opt, type_data, FPT['Bearing1_3'], 'XJTU')
      Bearing1_3_label_Con = np.array([1.]*len(Bearing1_3['x']))
      save_df(join(saved_dir, 'Bearing1_3_data_' + type_data + '.npy'), Bearing1_3['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_3_label_RUL.npy') , Bearing1_3['y'])
        save_df(join(saved_dir, 'Bearing1_3_label_Con.npy') , Bearing1_3_label_Con)

      Bearing1_4_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_4')
      Bearing1_4 = convert_to_image(Bearing1_4_path, opt, type_data, FPT['Bearing1_4'], 'XJTU')
      Bearing1_4_label_Con = np.array([1.]*len(Bearing1_4['x']))
      save_df(join(saved_dir, 'Bearing1_4_data_' + type_data + '.npy'), Bearing1_4['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_4_label_RUL.npy') , Bearing1_4['y'])
        save_df(join(saved_dir, 'Bearing1_4_label_Con.npy') , Bearing1_4_label_Con)

      Bearing1_5_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_5')
      Bearing1_5 = convert_to_image(Bearing1_5_path, opt, type_data, FPT['Bearing1_5'], 'XJTU')
      Bearing1_5_label_Con = np.array([1.]*len(Bearing1_5['x']))
      save_df(join(saved_dir, 'Bearing1_5_data_' + type_data + '.npy'), Bearing1_5['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing1_5_label_RUL.npy') , Bearing1_5['y'])
        save_df(join(saved_dir, 'Bearing1_5_label_Con.npy') , Bearing1_5_label_Con)

      Bearing2_1_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_1')
      Bearing2_1 = convert_to_image(Bearing2_1_path, opt, type_data, FPT['Bearing2_1'], 'XJTU')
      Bearing2_1_label_Con = np.array([2.]*len(Bearing2_1['x']))
      save_df(join(saved_dir, 'Bearing2_1_data_' + type_data + '.npy'), Bearing2_1['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_1_label_RUL.npy') , Bearing2_1['y'])
        save_df(join(saved_dir, 'Bearing2_1_label_Con.npy') , Bearing2_1_label_Con)

      Bearing2_2_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_2')
      Bearing2_2 = convert_to_image(Bearing2_2_path, opt, type_data, FPT['Bearing2_2'], 'XJTU')
      Bearing2_2_label_Con = np.array([2.]*len(Bearing2_2['x']))
      save_df(join(saved_dir, 'Bearing2_2_data_' + type_data + '.npy'), Bearing2_2['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_2_label_RUL.npy') , Bearing2_2['y'])
        save_df(join(saved_dir, 'Bearing2_2_label_Con.npy') , Bearing2_2_label_Con)

      Bearing2_3_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_3')
      Bearing2_3 = convert_to_image(Bearing2_3_path, opt, type_data, FPT['Bearing2_3'], 'XJTU')
      Bearing2_3_label_Con = np.array([2.]*len(Bearing2_3['x']))
      save_df(join(saved_dir, 'Bearing2_3_data_' + type_data + '.npy'), Bearing2_3['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_3_label_RUL.npy') , Bearing2_3['y'])
        save_df(join(saved_dir, 'Bearing2_3_label_Con.npy') , Bearing2_3_label_Con)

      Bearing2_4_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_4')
      Bearing2_4 = convert_to_image(Bearing2_4_path, opt, type_data, FPT['Bearing2_4'], 'XJTU')
      Bearing2_4_label_Con = np.array([2.]*len(Bearing2_4['x']))
      save_df(join(saved_dir, 'Bearing2_4_data_' + type_data + '.npy'), Bearing2_4['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_4_label_RUL.npy') , Bearing2_4['y'])
        save_df(join(saved_dir, 'Bearing2_4_label_Con.npy') , Bearing2_4_label_Con)

      Bearing2_5_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_5')
      Bearing2_5 = convert_to_image(Bearing2_5_path, opt, type_data, FPT['Bearing2_5'], 'XJTU')
      Bearing2_5_label_Con = np.array([2.]*len(Bearing2_5['x']))
      save_df(join(saved_dir, 'Bearing2_5_data_' + type_data + '.npy'), Bearing2_5['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing2_5_label_RUL.npy') , Bearing2_5['y'])
        save_df(join(saved_dir, 'Bearing2_5_label_Con.npy') , Bearing2_5_label_Con)

      Bearing3_1_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_1')
      Bearing3_1 = convert_to_image(Bearing3_1_path, opt, type_data, FPT['Bearing3_1'], 'XJTU')
      Bearing3_1_label_Con = np.array([3.]*len(Bearing3_1['x']))
      save_df(join(saved_dir, 'Bearing3_1_data_' + type_data + '.npy'), Bearing3_1['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_1_label_RUL.npy') , Bearing3_1['y'])
        save_df(join(saved_dir, 'Bearing3_1_label_Con.npy') , Bearing3_1_label_Con)

      Bearing3_2_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_2')
      Bearing3_2 = convert_to_image(Bearing3_2_path, opt, type_data, FPT['Bearing3_2'], 'XJTU')
      Bearing3_2_label_Con = np.array([3.]*len(Bearing3_2['x']))
      save_df(join(saved_dir, 'Bearing3_2_data_' + type_data + '.npy'), Bearing3_2['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_2_label_RUL.npy') , Bearing3_2['y'])
        save_df(join(saved_dir, 'Bearing3_2_label_Con.npy') , Bearing3_2_label_Con)

      Bearing3_3_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_3')
      Bearing3_3 = convert_to_image(Bearing3_3_path, opt, type_data, FPT['Bearing3_3'], 'XJTU')
      Bearing3_3_label_Con = np.array([3.]*len(Bearing3_3['x']))
      save_df(join(saved_dir, 'Bearing3_3_data_' + type_data + '.npy'), Bearing3_3['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_3_label_RUL.npy') , Bearing3_3['y'])
        save_df(join(saved_dir, 'Bearing3_3_label_Con.npy') , Bearing3_3_label_Con)

      Bearing3_4_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_4')
      Bearing3_4 = convert_to_image(Bearing3_4_path, opt, type_data, FPT['Bearing3_4'], 'XJTU')
      Bearing3_4_label_Con = np.array([3.]*len(Bearing3_4['x']))
      save_df(join(saved_dir, 'Bearing3_4_data_' + type_data + '.npy'), Bearing3_4['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_4_label_RUL.npy') , Bearing3_4['y'])
        save_df(join(saved_dir, 'Bearing3_4_label_Con.npy') , Bearing3_4_label_Con)

      Bearing3_5_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_5')
      Bearing3_5 = convert_to_image(Bearing3_5_path, opt, type_data, FPT['Bearing3_5'], 'XJTU')
      Bearing3_5_label_Con = np.array([3.]*len(Bearing3_5['x']))
      save_df(join(saved_dir, 'Bearing3_5_data_' + type_data + '.npy'), Bearing3_5['x'])
      if type_data == '1d':
        save_df(join(saved_dir, 'Bearing3_5_label_RUL.npy') , Bearing3_5['y'])
        save_df(join(saved_dir, 'Bearing3_5_label_Con.npy') , Bearing3_5_label_Con)

      print(f'\n Saving data in {opt.type} data set'+'-'*100)


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

print(f'Shape of training Con label: {train_label_Con.shape}')  
print(f'Shape of test Con label: {test_label_Con.shape}\n')
