import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics
from tensorflow import keras
from tensorflow.keras import layers


def get_data(filename):
    data = pd.read_csv(filename)
    '''change the date column in a data frame to datetime'''
    data['date']= pd.to_datetime(data['date'])

    '''Extracting some datetime features 
    like year, month, day of month, and day of week'''
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['dayOfMonth'] = data['date'].dt.day
    data['dayOfWeek'] = data['date'].dt.dayofweek
    return data

def get_time_series_data(filename, store_nbr, family):
    """Read the data from the file and return the time series data"""
    #data = pd.read_csv(filename)
    #condition = (data['store_nbr'] == store_nbr) & (data['family_' + family] == True)
    ##data = data[condition]['sales'].values
    #stores = data['store_nbr'].unique()
    #dates = data['date'].unique()
    #arrayData = []
    #for date in dates:
    #    arrayData.append(data.loc[(data['date'] == date), 'sales'].sum())
    #with open('store-sales-data.text', 'w') as f:
    #    for item in arrayData:
    #        f.write("%s\n" % item)
    arrayData = []
    with open('store-sales-data.text', 'r') as f:
        for line in f:
            arrayData.append(float(line.strip()))
    data = np.array(arrayData)
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data, scaler

def split_data_train_test(data):
    #train = data[data['date']<='15-08-2017'].reset_index(drop=True)
    #test = data[data['date']>'15-08-2017'].reset_index(drop=True)
    train = get_data("store_data-train.csv")
    test = get_data("store_data-test.csv")

    # Drop the 'id' column
    train.drop('id', axis=1, inplace=True)
    return train, test

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX, dataY

def split_data_train_test_time_series(data, batch_size, time_steps, look_back):
    # use 67% of data for training
    train_size = int(len(data) * 0.67)
    train_size = train_size
    # test_size = len(data) - train_size
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], time_steps, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], time_steps, X_test.shape[1]))
    return X_train, Y_train, X_test, Y_test

def split_train_validation_set(train):
    # we got 15 days only for val since our test set is 16 days only
    split_date = '2017-08-01' 

    # Create training and validation sets
    train_set = train[train['date'] < split_date]
    val_set = train[train['date'] >= split_date]

    # Drop the 'date' column
    train_set.drop('date', axis=1, inplace=True)
    val_set.drop('date', axis=1, inplace=True)
    return train_set, val_set

def create_training_validation_data(train_set, val_set, test):
    # Define X and y for training and validation
    X_train = train_set.drop('sales', axis=1)  # Features for training
    label = train_set['sales']  # Target for training
    X_val = val_set.drop('sales', axis=1)  # Features for validation
    y_val = val_set['sales']  # Target for validation

    # Ensure that the data types are appropriate (e.g., float32)
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    label = label.astype(np.float32)
    y_val = y_val.astype(np.float32)

    X_test = test.drop(['id','sales','date'], axis=1)
    y_test = test['sales']
    y_test = y_test.astype(np.float32)
    return X_train, label, X_val, y_val, X_test, y_test


def make_pipeline(X_train, X_val, X_test):
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on X_train and transform X_train
    X_train = scaler.fit_transform(X_train)

    # Use the same scaler to transform X_val and X_test
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def create_model(X_train):
    model = keras.Sequential([
        layers.Input(shape=X_train.shape[1]),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    # Compile the model with a suitable loss function and optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_model_time_series(batch_size, time_steps, look_back):
    # LSTM(4, batch_input_shape=(batch_size, time_steps, features), stateful=True)
    model = keras.Sequential([
        layers.LSTM(4, batch_input_shape=(batch_size, time_steps, look_back), stateful=True, return_sequences=True),
        #layers.Dropout(0.2),
        layers.LSTM(4, stateful=True, return_sequences=True),
        #layers.Dropout(0.2),
        layers.LSTM(4, stateful=True),
        layers.Dense(units=1)
    ])

    # Compile the model with a suitable loss function and optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


def do_fit(model, X_train, label, X_val, y_val):
    # Train the model
    model.fit(X_train, label, validation_data=(X_val, y_val), epochs=10, verbose=1)
    return model

def do_fit_time_series(model, X_train, Y_train, epochs, batch_size):
    # fit the LSTM network
    for i in range(epochs):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=2, shuffle=False)
        if i % 10 == 0:
            model.reset_states()
    return model

def save_model(model, filename):
    model.save("./" + filename)

def load_model(modelname):
    model = keras.models.load_model("./" + modelname)
    return model

def do_test(model, X_train, label, X_test, y_test):
    X_test=X_test.astype(np.float32)
    y_test_pred = model.predict(X_test)

    X_train=X_train.astype(np.float32)
    y_train_pred = model.predict(X_train)

    train_loss = sklearn.metrics.mean_squared_error(label.astype(np.float32), y_train_pred)
    test_loss = sklearn.metrics.mean_squared_error(y_test, y_test_pred)
    print("\n")
    print("\n")
    print("Loss Values:\n")
    print("L2(MSE) train loss:", train_loss, " test loss:", test_loss)
    train_loss = sklearn.metrics.mean_absolute_error(label.astype(np.float32), y_train_pred)
    test_loss = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
    print("L1(MAE) train loss:", train_loss, " test loss:", test_loss)
    train_loss = sklearn.metrics.r2_score(label.astype(np.float32), y_train_pred)
    test_loss = sklearn.metrics.r2_score(y_test, y_test_pred)
    print("R2 train loss:", train_loss, " test loss:", test_loss)
    print("\n")
    print("\n")

def do_test_time_series(model, X_train, Y_train, X_test, Y_test, scaler, batch_size):
    # invert predictions
    trainPredict = model.predict(X_train, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(X_test, batch_size=batch_size)
    model.reset_states()
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    Y_train = scaler.inverse_transform([Y_train])
    Y_test = scaler.inverse_transform([Y_test])
    trainScore = np.sqrt(sklearn.metrics.mean_squared_error(Y_train[0], trainPredict[:,0]))
    testScore = np.sqrt(sklearn.metrics.mean_squared_error(Y_test[0], testPredict[:,0]))
    print("\n")
    print("\n")
    print("Loss Values:\n")
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Test Score: %.2f RMSE' % (testScore))
    #trainScore = sklearn.metrics.r2_score(Y_train[0].astype(np.float32), trainPredict[:,0])
    #testScore = sklearn.metrics.r2_score(Y_test[0], testPredict[:,0])
    #print("Train Score: %.2f R2" % trainScore)
    #print("Test Score: %.2f R2" % testScore)
    print("\n")
    print("\n")

def do_predictions(model, X_test, test):
    X_test=X_test.astype(np.float32)
    y_test_pred = model.predict(X_test)

    # Assuming test dataframe has an 'id' column
    predictions_df = pd.DataFrame({'id': test['id'], 'sales': y_test_pred.flatten()})

    predictions_df['sales'] = predictions_df['sales'].clip(lower=0)  # Clip negative values to 0
    predictions_df.to_csv('/pred.csv', index=False)

def display_prediction_time_series(model, data, X_test, X_train, scaler, batch_size, look_back):
    testPredict = model.predict(X_test, batch_size=batch_size)
    model.reset_states()
    trainPredict = model.predict(X_train, batch_size=batch_size)
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(data))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

def main():
    # best lookback size
    #look_back = 128
    #batch_size = 1
    #epochs = 100
    #time_steps = 1
    #data, scaler = get_time_series_data("store_data.csv", 1, "AUTOMOTIVE")
    #X_train, Y_train, X_test, Y_test = split_data_train_test_time_series(data, batch_size, time_steps, look_back)
    #model = create_model_time_series(batch_size, time_steps, look_back)
    #model = do_fit_time_series(model, X_train, Y_train, epochs, batch_size)
    #save_model(model, "model-time-series-" + str(look_back) + "-" + str(batch_size) + ".keras")
    ##model = load_model("model-time-series-" + str(look_back) + "-" + str(batch_size) + ".keras")
    #do_test_time_series(model, X_train, Y_train, X_test, Y_test, scaler, batch_size)
    #display_prediction_time_series(model, data, X_test, X_train, scaler, batch_size, look_back)

    data = get_data("store_data.csv")
    train, test = split_data_train_test(data)
    train_set, val_set = split_train_validation_set(train)
    X_train, label, X_val, y_val, X_test, y_test = create_training_validation_data(train_set, val_set, test)
    X_train, X_val, X_test = make_pipeline(X_train, X_val, X_test)
    model = create_model(X_train)
    model = do_fit(model, X_train, label, X_val, y_val)
    save_model(model, "model.keras")
    #model = load_model("model.keras")
    do_test(model, X_train, label, X_test, y_test)

if __name__ == "__main__":
    main()
