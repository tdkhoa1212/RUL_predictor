from tensorflow.keras.layers import  Dense, concatenate, BatchNormalization, \
                                    GlobalAveragePooling1D, \
                                    GlobalAveragePooling2D, \
                                    Dropout, Reshape
import tensorflow as tf
from keras.models import Model
from model.LSTM import TransformerLayer
from keras import regularizers
import keras.backend as K

def fully_concatenate(hidden_out_1D, hidden_out_2D, hidden_out_extracted, training=None, fully=None):
    if fully:  
      # Fully connected layer 1 ###################################################################
      hidden_out_1D = BatchNormalization()(hidden_out_1D, training=training)
      hidden_out_1D = tf.keras.layers.Dense(hidden_out_1D.shape[-1],   
                                            activation='relu',
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4),
                                            activity_regularizer=regularizers.l2(1e-5))(hidden_out_1D)
      hidden_out_1D = Dropout(0.2)(hidden_out_1D, training=training)
      
      # Fully connected layer 2 ###################################################################
      hidden_out_2D = BatchNormalization()(hidden_out_2D, training=training)
      hidden_out_2D = tf.keras.layers.Dense(hidden_out_2D.shape[-1],   
                                            activation='relu',
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4),
                                            activity_regularizer=regularizers.l2(1e-5))(hidden_out_2D)
      hidden_out_2D = Dropout(0.2)(hidden_out_2D, training=training)
      
      # Fully connected layer 3 ###################################################################
      hidden_out_extracted = BatchNormalization()(hidden_out_extracted, training=training)
      hidden_out_extracted = tf.keras.layers.Dense(hidden_out_extracted.shape[-1],   
                                                   activation='relu',
                                                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                   bias_regularizer=regularizers.l2(1e-4),
                                                   activity_regularizer=regularizers.l2(1e-5))(hidden_out_extracted)
      hidden_out_extracted = Dropout(0.2)(hidden_out_extracted, training=training)
      all_ = concatenate((hidden_out_1D, hidden_out_2D, hidden_out_extracted))
      return all_
    
    # concatenate layer ###################################################################
    all_ = concatenate((hidden_out_1D, hidden_out_2D, hidden_out_extracted))
    return all_

def reshape(x):
    out = Reshape((x.shape[-2]*x.shape[-3], x.shape[-1]))(x)
    return out

def mix_model_XJTU(opt, cnn_1d_model, resnet_50, lstm_extracted_model, input_1D, input_2D, input_extracted, training=False):
  out_1D = cnn_1d_model(opt, training, input_1D)
  out_2D = resnet_50(opt)(input_2D, training=training)

  ##################### https://keras.io/api/applications/ #######################################
#   base_model_2D = tf.keras.applications.EfficientNetV2B3(include_top=False,
#                                                          input_shape=(128, 128, 2),
#                                                          weights=None)
#   out_2D = base_model_2D(input_2D, training=training)

  out_extracted = lstm_extracted_model(opt, training, input_extracted)
  
  # 1D branch-------------------------------------------
  network_1D = Model(input_1D, out_1D, name='network_1D')
  # 2D branch-------------------------------------------
  network_2D = Model(input_2D, out_2D, name='network_2D')
  # Extracted branch-------------------------------------------
  network_extracted = Model(input_extracted, out_extracted, name='network_extracted')
  
  # Hidden layers----------------------------------------------------------
  hidden_out_1D = network_1D([input_1D])
  hidden_out_2D = network_2D([input_2D])
  hidden_out_extracted = network_extracted([input_extracted])
  
  # Transformer layers-----------------------------------------------------
  rul_hidden_out_1D = TransformerLayer(hidden_out_1D, hidden_out_1D.shape[-1], training=training)
  rul_hidden_out_2D = reshape(hidden_out_2D)
  rul_hidden_out_2D = TransformerLayer(rul_hidden_out_2D, rul_hidden_out_2D.shape[-1], training=training)
  rul_hidden_out_extracted = TransformerLayer(hidden_out_extracted, hidden_out_extracted.shape[-1], training=training)

  # Global average pooling layers--------------------------------------------
  con_hidden_out_1D = GlobalAveragePooling1D()(hidden_out_1D)
  con_hidden_out_2D = GlobalAveragePooling2D()(hidden_out_2D)
  con_hidden_out_extracted = GlobalAveragePooling1D()(hidden_out_extracted)
  
  # Fully connected layers---------------------------------------------------
  merged_value_0 = fully_concatenate(rul_hidden_out_1D, rul_hidden_out_2D, rul_hidden_out_extracted, training=training, fully=False)
  merged_value_1 = fully_concatenate(con_hidden_out_1D, con_hidden_out_2D, con_hidden_out_extracted, training=training, fully=False)
    
  # Final output layers---------------------------------------------------
  Condition = Dense(3, 
                    activation='softmax', 
                    name='Condition')(merged_value_1)
  RUL = Dense(1, 
              activation='sigmoid', 
              name='RUL')(merged_value_0)
  return Condition, RUL

def mix_model_PHM(opt, cnn_1d_model, resnet_50, lstm_extracted_model, input_1D, input_2D, input_extracted, training=False):
  out_1D = cnn_1d_model(opt, training, input_1D)
  out_2D = resnet_50(opt)(input_2D, training=training)

  ##################### https://keras.io/api/applications/ #######################################
#   base_model_2D = tf.keras.applications.EfficientNetV2B3(include_top=False,
#                                                          input_shape=(128, 128, 2),
#                                                          weights=None)
#   out_2D = base_model_2D(input_2D, training=training)

  out_extracted = lstm_extracted_model(opt, training, input_extracted)
  
  # 1D branch-------------------------------------------
  network_1D = Model(input_1D, out_1D, name='network_1D')
  # 2D branch-------------------------------------------
  network_2D = Model(input_2D, out_2D, name='network_2D')
  # Extracted branch-------------------------------------------
  network_extracted = Model(input_extracted, out_extracted, name='network_extracted')
  
  # Hidden layers----------------------------------------------------------
  hidden_out_1D = network_1D([input_1D])
  hidden_out_2D = network_2D([input_2D])
  hidden_out_extracted = network_extracted([input_extracted])
  
  # Transformer layers-----------------------------------------------------
  rul_hidden_out_1D = TransformerLayer(hidden_out_1D, hidden_out_1D.shape[-1], training=training)
  rul_hidden_out_2D = reshape(hidden_out_2D)
  rul_hidden_out_2D = TransformerLayer(rul_hidden_out_2D, rul_hidden_out_2D.shape[-1], training=training)
  rul_hidden_out_extracted = TransformerLayer(hidden_out_extracted, hidden_out_extracted.shape[-1], training=training)
  
  # Fully connected layers---------------------------------------------------
  merged_value_0 = fully_concatenate(rul_hidden_out_1D, rul_hidden_out_2D, rul_hidden_out_extracted, training=training, fully=False)
    
  # Final output layer---------------------------------------------------
  RUL = Dense(1, 
              activation='sigmoid', 
              name='RUL')(merged_value_0)
  return RUL