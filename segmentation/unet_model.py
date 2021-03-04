from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *

def double_conv_layer(x, size, dropout=0.0, batch_norm = True):
  axis = 3
  conv = Conv2D(size, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
  if batch_norm is True:
      conv = BatchNormalization(axis=axis)(conv)
  conv = Activation('relu')(conv)
  conv = Conv2D(size, (3, 3), padding='same', kernel_initializer='glorot_uniform')(conv)
  if batch_norm is True:
      conv = BatchNormalization(axis=axis)(conv)
  conv = Activation('relu')(conv)
  if dropout > 0:
      conv = SpatialDropout2D(dropout)(conv)
  return conv

def decoder_double_conv_layer(x, size, encoder_layer_cat, dropout = 0.0, batch_norm = True):
  up = UpSampling2D(size = (2,2))(x)
  conv = Conv2D(size, (2, 2), padding='same', kernel_initializer='glorot_uniform')(up)
  conv = Activation('relu')(conv)
  conc = concatenate([encoder_layer_cat, conv], axis = 3)
  conv = double_conv_layer(conc, size, dropout, batch_norm)
  return conv
  

def unet(initial_filters = 32, input_size = (256, 256, 1), dropout = 0.0):
  filters = initial_filters

  inputs = Input(input_size)
  
  conv_0 = double_conv_layer(inputs, filters, dropout = dropout)
  pool_0 = MaxPooling2D(pool_size=(2, 2))(conv_0)

  conv_1 = double_conv_layer(pool_0, 2*filters, dropout = dropout)
  pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

  conv_2 = double_conv_layer(pool_1, 4*filters, dropout = dropout)
  pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

  conv_3 = double_conv_layer(pool_2, 8*filters, dropout = dropout)
  pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

  conv_4 = double_conv_layer(pool_3, 16*filters, dropout = dropout)
  
  up_conv_3 = decoder_double_conv_layer(conv_4, 8*filters, conv_3, dropout = dropout)
  
  up_conv_2 = decoder_double_conv_layer(up_conv_3, 4*filters, conv_2, dropout = dropout)

  up_conv_1 = decoder_double_conv_layer(up_conv_2, 2*filters, conv_1, dropout = dropout)
  
  up_conv_0 = decoder_double_conv_layer(up_conv_1, filters, conv_0, dropout = dropout)

  conv_final = Conv2D(1, (1, 1))(up_conv_0)
  conv_final = Activation('sigmoid')(conv_final)

  model = Model(inputs, conv_final, name="unet_32")

  return model
