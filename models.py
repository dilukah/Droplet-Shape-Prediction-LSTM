import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Reshape, AveragePooling2D , AveragePooling3D, MaxPooling3D, BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Bidirectional

def create_shape_prediction_lstm_model(num_timesteps, num_features, num_outputs):
    input_shape = (num_timesteps, num_features)
    model = Sequential()
    model.add(LSTM(units=256, input_shape= input_shape, return_sequences=True))

    model.add(TimeDistributed(Dense(num_outputs,activation='sigmoid')))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def create_shape_prediction_lstm_var_seq_len(num_features, num_outputs, lstm_units = 256):
    input_shape = (None, num_features) #seq length is free to be any size, when training however, a batch should have same seq length.
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape= input_shape, return_sequences=True))

    model.add(TimeDistributed(Dense(num_outputs,activation='linear')))
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
    return model

def create_inverse_lstm_model(num_timesteps, num_features, num_outputs, lstm_units = 100):
    input_shape = (num_timesteps, num_features)
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape= input_shape, return_sequences=True))

    model.add(TimeDistributed(Dense(num_outputs,activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_single_step_model(num_features, num_outputs, dense_units = 100):
    
    model = Sequential()
    model.add(Dense(num_outputs, input_shape = (num_features,),activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_lstm_one_to_many(num_features, num_outputs, lstm_units = 50):
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=(1, num_features)))
    model.add(Dense(num_outputs))
    model.compile(optimizer='adam', loss='mse')
    return model