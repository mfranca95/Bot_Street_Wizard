# GENERAL NOTES
## This program requires keys from the ALPACA API in order to function. 
## Please ensure that you posses the necesary keys in order to continue

### This program uses 3 diferent machine learning algorithms in order to predict the possible movement of a stock price based on technical analysis.
### The algorithms are the following:
### - Long Short-term Memory
### - ARIMA
### - Neural Networks

### This program creates a Web UI using the library StreamLit
### Please make sure to install all the necesary dependencies

#### Disclamer: Do not take the result of this algorithm as financial advice. Like a wise trader once said "Technical Analysis is Astrology for stock traders"



# ********* Import of required libaries *************

import pandas as pd
import numpy as np
import math
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import alpaca_trade_api as tradeapi
from pathlib import Path
import holoviews as hv
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import requests
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot

# ********* Defining global variables part 1 ***************
if 'alpaca_api_key' not in st.session_state:
    st.session_state.disabled_search = True
    st.session_state.alpaca_api_key = ''
    st.session_state.alpaca_secret_key = ''

# ********* Alpacas key capture and sidebar logic enabler ************* 
with st.form(key='alpaca_keys'):
    alpaca_api_key = st.text_input(label="Please input your Alpaca Key")
    alpaca_secret_key = st.text_input(label="Please input your Alpaca Secret Key")
    submited = st.form_submit_button('Confirm')
    if submited:
        st.session_state.alpaca_api_key = alpaca_api_key
        st.session_state.alpaca_secret_key = alpaca_secret_key
        st.session_state.disabled_search = False

# ********* Defining global variables part 2 ***************
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = []
    st.session_state['df'] = {}
    st.session_state['all_tickers_predictions'] = {}
    st.session_state['all_predictions'] = {}
    st.session_state['predictions'] = []

if 'result_list' not in st.session_state or st.session_state['result_list'] is None:
    st.session_state['result_list'] = []

target_drop_list_5 = ['close','high','low','trade_count','open','volume','vwap','signal_15', 'signal_30']
target_drop_list_15 = ['close','high','low','trade_count','open','volume','vwap','signal_5', 'signal_30']
target_drop_list_30 = ['close','high','low','trade_count','open','volume','vwap','signal_5', 'signal_15']

timeframe = "1Day"
train_start_date = "2003-12-14"
train_end_date = "2023-12-14"
start_date = pd.Timestamp(train_start_date, tz="America/New_York").isoformat()
end_date = pd.Timestamp(train_end_date, tz="America/New_York").isoformat()

# def begining(start):
#     start_date = pd.Timestamp(start, tz="America/New_York").isoformat()
#     return start_date

# def end(end):
#     end_date = pd.Timestamp(end, tz="America/New_York").isoformat()
#     return end_date

# ************* Alpaca API object and post key enabler logic *******************
if st.session_state.disabled_search == False:
    alpaca = tradeapi.REST(
        st.session_state.alpaca_api_key,
        st.session_state.alpaca_secret_key,
        api_version="v2")

    # *************** Global functions ********************
    def df_portfolio_single(ticker): 
        info = alpaca.get_bars(
            symbol=ticker,
            timeframe= timeframe,
            start = start_date,
            end = end_date
        ).df
        return info

    def new_ticker(ticker):
        # print(ticker)
        if ticker == None:
            return
        if ticker not in st.session_state['tickers']:
            st.session_state['tickers'].append(ticker)
            df = df_preper(st.session_state['ticker'])
            st.session_state['df'][ticker] = df
            printer(df,ticker)
        else:
            printer(st.session_state['df'][ticker],ticker)

    def new_ticker_input():
        ticker = st.session_state['ticker']
        new_ticker(ticker)

    def new_ticker_drop():
        ticker = st.session_state['drop_ticker']
        new_ticker(ticker)

    def new_ticker_sim():
        ticker = st.session_state['sim_ticker']
        new_ticker(ticker)
        
    def printer(df,ticker):
        ### *-*-*-*-*-* Main screen -*-*-*-*-*-*-*-*
        st.write("""# {}""".format(ticker))
        df = st.session_state['df'][ticker]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Info","LSTM Model", "ARIMA Model", "NN Model", "Trade"])

        with tab1:
            st.write(""" ### Basic historic info """)
            st.dataframe(df)

            st.write(""" ### Historic Close """)
            st.line_chart(df['close'])

            st.write(""" ### Historic Volume """)
            st.line_chart(df['volume'])

            st.write(""" ### Historic Volume-Weighted average price """)
            st.line_chart(df['vwap'])

        with tab2:
            st.write(""" ### Long short-term memory model """)
            if 'result_list_LSTM' not in st.session_state or st.session_state['result_list_LSTM'] is None:
                st.session_state['result_list_LSTM'] = []

            # if st.button("Run Test"):
            #     st.write("Button pressed!")
            # checkbox = st.checkbox("Run simulation")
            # if checkbox:
            
            # result_list = st.button("Run simulation", on_click= LSTM_execution)

            # if ticker not in st.session_state['all_predictions']:
            #     st.session_state['all_predictions'][ticker] = []

            # if st.session_state['result_list'] not in st.session_state['all_predictions'][ticker]:
            st.session_state['result_list_LSTM'] = LSTM_execution(df)
                # st.session_state['all_predictions'].append(ticker[st.session_state['result_list']])
            
            index_list = ['5 days ', ' 15 days ', ' 30 days ']
            result_list_trans = ['NA','NA','NA']
            
            for x in [0,1,2]:
                if st.session_state['result_list_LSTM'][x] == 1:
                    result_list_trans[x] = 'Buy'
                if st.session_state['result_list_LSTM'][x] == -1:
                    result_list_trans[x] = 'Sell'
                if st.session_state['result_list_LSTM'][x] == 0:
                    result_list_trans[x] = 'Hold'
            
            ## ------------ Result Table -------------- ##

            list_df = pd.DataFrame(result_list_trans, columns=[' Predictions '])
            list_df.index = index_list
            st.table(list_df)

            ## ---------------------------------------- ##

        with tab3:
            st.write(""" ### Autoregressive integrated moving average model """)

            
        with tab4:
            st.write(""" ### Neural Network model """)

            if 'result_list_NN' not in st.session_state or st.session_state['result_list_NN'] is None:
                st.session_state['result_list_NN'] = []

            st.session_state['result_list_NN'] = NN_execution(df)

            ## ------------ Result Table -------------- ##

            list_df = pd.DataFrame(st.session_state['result_list_NN'], columns=[' Predictions '])
            list_df.index = index_list
            st.table(list_df)

            ## ---------------------------------------- ##

    def signal_creator_5(row):
        percent_diff = (row['close'] - row['shifted_close_5']) / row['shifted_close_5']
        if percent_diff > 0.03:
            return 1
        elif percent_diff < -0.03:
            return -1
        else:
            return 0
        
    def signal_creator_15(row):
        percent_diff = (row['close'] - row['shifted_close_15']) / row['shifted_close_15']
        if percent_diff > 0.03:
            return 1
        elif percent_diff < -0.03:
            return -1
        else:
            return 0

    def signal_creator_30(row):
        percent_diff = (row['close'] - row['shifted_close_30']) / row['shifted_close_30']
        if percent_diff > 0.03:
            return 1
        elif percent_diff < -0.03:
            return -1
        else:
            return 0

    def df_preper(ticker):
        df = df_portfolio_single(ticker)
        # df = df.drop(columns=['symbol'])
        
        # 5 day shift
        df['shifted_close_5'] = df['close'].shift(periods=5)
        df['signal_5'] = 0
        print(df)
        df['signal_5'] = df.apply(signal_creator_5, axis=1)

        # 15 day shift
        df['shifted_close_15'] = df['close'].shift(periods=15)
        df['signal_15'] = 0
        print(df)
        df['signal_15'] = df.apply(signal_creator_15, axis=1)

        # 30 day shift
        df['shifted_close_30'] = df['close'].shift(periods=30)
        df['signal_30'] = 0
        df['signal_30'] = df.apply(signal_creator_30, axis=1)

        # Columns and NA drops to prevent data leakege
        df.dropna(inplace=True)
        df.drop(columns=['shifted_close_5','shifted_close_15','shifted_close_30'],inplace=True)
        print(df)
        return df

    def checker(ticker):
        for x in st.session_state['tickers']:
            if ticker == st.session_state['tickers'][x]:
                return True
            else:
                return False
            
    def data_features_separator(df):
        features = df.drop(columns = ['signal_5', 'signal_15', 'signal_30'], axis=1)
        return features

    # ***************** LSTM Model ************************

    def create_dataset(X, y, time_steps=5):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps), :])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    def builder_LSTM(model,time_steps,X):
        model.add(LSTM(150, return_sequences=True, input_shape=(time_steps, X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(75, return_sequences=False))
        model.add(Dense(50,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(25,activation='relu'))
        model.add(Dense(1,activation='linear'))
        return model

    def interpret_prediction(prediction):
        if prediction > 0.03:
            return 1  # Buy
        elif prediction < -0.03:
            return -1  # Sell
        else:
            return 0  # Hold
        
    def eval(model, X_test, y_test):
        test_loss = model.evaluate(X_test, y_test)
        return test_loss

    def train_LSTM(df, target):
        features = data_features_separator(df)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)   
        time_steps = 5
        X, y = create_dataset(scaled_features, target.values, time_steps)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=13)

        model = Sequential()

        model = builder_LSTM(model,time_steps,X)

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=10, batch_size=32)

        evaluation = eval(model,X_test,y_test)
        print(evaluation)

        return model

    def LSTM_aplication(model, df, target):
        features = data_features_separator(df)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)   
        time_steps = 5
        X, y = create_dataset(scaled_features, target.values, time_steps)
        aplication_size = int(len(X)*0.9)
        X_test_app = X[aplication_size:]
        predictions = model.predict(X_test_app)
        results = [interpret_prediction(p[0]) for p in predictions]
        return results[-1]
        
    def LSTM_execution(df):
        prediction_list = []

        target_5 = df.drop(columns = target_drop_list_5, axis=1)
        target_15 = df.drop(columns = target_drop_list_15, axis=1)
        target_30 = df.drop(columns = target_drop_list_30, axis=1)

        model_5 = train_LSTM(df,target_5)
        model_15 = train_LSTM(df,target_15)
        model_30 = train_LSTM(df,target_30)

        prediction_list.append(LSTM_aplication(model_5,df,target_5))
        prediction_list.append(LSTM_aplication(model_15,df,target_15))
        prediction_list.append(LSTM_aplication(model_30,df,target_30))

        # if prediction_list not in st.session_state['predictions']:
        #     st.session_state['predictions'].append(prediction_list)
        #     df = df_preper(st.session_state['ticker'])
        #     st.session_state['all_predictions'][prediction_list] = df

        print('prediction list inside execution function:', prediction_list)
        return prediction_list

    # *************** ARIMA Model ************************
    def train_ARIMA(df):
        series = df['close']
        series.index = pd.to_datetime(series.index).to_period('D')
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        print(model_fit.summary())
        # Split the DataFrame into training and testing sets (90% train, 10% test)
        train_data, test_data = df[0:int(len(df)*0.9)], df[int(len(df)*0.9):]
        # Prepare the training and testing data (using 'close' prices)
        train_arima = train_data['close']
        test_arima = test_data['close']
        # Initialize history with the training data
        history = [x for x in train_arima]
        # Prepare a list to store the forecasts
        predictions = []
        # Loop over the test data
        for t in range(len(test_arima)):
            # Fit the ARIMA model
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            # Forecast the next value
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            # Add the actual observation to history for the next loop
            history.append(test_arima.iloc[t])

    # ****************** NN Model ************************
    def prepare_data(data, n_steps, n_days):
        X, y = [], []
        for i in range(len(data) - n_steps - n_days):
            X.append(data[i : i + n_steps, 0])
            y.append(data[i + n_steps : i + n_steps + n_days, 0])
        return np.array(X), np.array(y)

    def builder_NN(model,n_steps,n_days):
        model.add(Dense(units=64, activation='relu', input_dim=n_steps))
        model.add(Dense(units=n_days, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def NN_train_config(df,n_steps,n_days,a,b):
        data = df['close'].values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        model = Sequential()
        model = builder_NN(model,n_steps,n_days)
        X, y = prepare_data(data_scaled, n_steps, n_days)
        model.fit(X,y, epochs=10, verbose=1)
        X_test, _ = prepare_data(data_scaled, n_steps, n_days)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        threshold = 0.02
        if (prediction[:-1] > (1 + threshold) * data[-1, 0]).all():
            signal = "Buy"
        elif (prediction[:-1] < (1 - threshold) * data[-1, 0]).all():
            signal = "Sell"
        else:
            signal = "Hold"
        return signal

    def NN_execution(df):
        prediction_list = []

        prediction_list.append(NN_train_config(df,5,5,-5,-10))
        prediction_list.append(NN_train_config(df,15,15,-15,-30))
        prediction_list.append(NN_train_config(df,30,30,-30,-60))

        return prediction_list

    # **************** Web UI Code ***********************

    ### *-*-*-*-*-* SideBar -*-*-*-*-*-*-*-*

    ticker = st.sidebar.text_input("Ticker symbol",
                                on_change=new_ticker_input, 
                                key='ticker', 
                                disabled=st.session_state.disabled_search)

    selected_stock = st.sidebar.selectbox("Stock on file",
                                        st.session_state['tickers'],
                                        on_change=new_ticker_drop,
                                        key='drop_ticker',
                                        disabled= st.session_state.disabled_search)









