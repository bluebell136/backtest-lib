# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent import BaseAgent
from backtest import Backtest, Generator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier

import dask.dataframe as dd
import datetime
import joblib
import os
import numpy as np
import pandas as pd

class Agent(BaseAgent):

    def __init__(self, 
            name:str, 
            default_quantity:int, 
            model:RandomForestClassifier, 
            scaler:StandardScaler,
        ):
        """
        Trading agent implementation example.

        :param name:
            str, agent name
        :param default_quantity:
            int, quantity to be used with every order
        :param model:
            RandomForestClassifier, trained model instance
        :param scaler:
            StandardScaler, scaler instance with parameters inherent to the preprocessed data
        """
        super(Agent, self).__init__(name)
        
        # static attributes from arguments
        self.default_quantity = default_quantity
        self.model = model
        self.scaler = scaler
        
        # dynamic attributes
        self.time_to_trade = False

    def on_quote(self, market_id:str, book_state:pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/size for 10 levels
        """

        # transform features
        x = book_state[1:].copy()
        x = self.scaler.transform(x.values.reshape(1, len(x)))
        self.y_hat = self.model.predict(x)[0] # predict next mid price movement

        # open a position
        qt_sum = sum([t.quantity for t in self.get_filtered_trades(market_id, "buy")]) - \
                 sum([t.quantity for t in self.get_filtered_trades(market_id, "sell")])

        # ...
        if self.get_filtered_trades(market_id):
            last_order = self.get_filtered_trades(market_id)[-1]
        else:
            last_order = None

        # trading time
        if self.time_to_trade:

            if self.y_hat == +1 and qt_sum == 0 or (qt_sum < 0 and book_state["L1-AskPrice"] < last_order.price):
                self.submit_order(market_id, "buy", self.default_quantity)  # buy at best ask
            if self.y_hat == -1 and qt_sum == 0 or (qt_sum > 0 and book_state["L1-BidPrice"] > last_order.price):
                self.submit_order(market_id, "sell", self.default_quantity)  # sell at best bid

            # ...
            if last_order:

                if last_order.timestamp > book_state["TIMESTAMP_UTC"] + pd.Timedelta(hours=1) and qt_sum > 0:
                    print("one hour passed by without an option to make a profit.")
                    self.submit_order(market_id, "sell", self.default_quantity)  # sell at best bid
                if last_order.timestamp > book_state["TIMESTAMP_UTC"] + pd.Timedelta(hours=1) and qt_sum < 0:
                    print("one hour passed by without an option to make a profit.")
                    self.submit_order(market_id, "buy", self.default_quantity)  # buy at best ask

        # ...

    def on_trade(self, market_id:str, trades_state:pd.Series):
        pass

    def on_news(self, market_id:str, news_state:pd.Series):
        pass

    def on_time(self, timestamp:pd.Timestamp, timestamp_next:pd.Timestamp):
        """
        This method is called with every iteration and provides the timestamps
        for both current and next iteration. The given interval may be used to
        submit orders before a specific point in time.

        :param timestamp:
            pd.Timestamp, timestamp recorded
        :param timestamp_next:
            pd.Timestamp, timestamp recorded in next iteration
        """

        # close active positions and cancel active orders two minutes before the market closes
        if timestamp.time() >= datetime.time(16, 28) and self.time_to_trade:
            for market_id, market in self.markets.items():

                # cancel active orders for this market
                if self.get_filtered_orders(market_id, status="ACTIVE"):
                    [self.cancel_order(order) for order in self.get_filtered_orders(market_id, status="ACTIVE")]

                # close positions for this market
                if self.exposure[market_id] > 0:
                    self.submit_order(market_id, "sell", self.default_quantity)
                if self.exposure[market_id] < 0:
                    self.submit_order(market_id, "buy", self.default_quantity)
                
                # ...
                self.time_to_trade = False
            
        # we want to start trading 15 minutes after XETRA has opened
        if timestamp.time() >= datetime.time(8, 15) and not self.time_to_trade:
            self.time_to_trade = True  
                
        # ...

def get_data(sources:list, start_date:str, end_date:str):
    """
    Load data.

    :param sources:
        list, [<source_id>, *]
    :param start_date:  
        str, date string
    :param end_date:  
        str, date string
    """
    
    # identify stocks, transform dates from YYYY-MM-DD to YYYYMMDD
    stocks = set(x.split(".")[0] for x in sources)
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")
    
    print(f"load data for stocks {stocks} and date range [{start_date}, {end_date}] ...")

    # identify all paths available in directory
    path_list = [os.path.join(pre, f) for pre, _, sub 
        in os.walk("./data") for f in sub if not f.startswith((".", "_"))
    ]
    # consider only the specified stocks
    path_list = [path for path in path_list if any(
        x in path for x in stocks
    )]
    
    # filter paths to include only BOOK files for the specified date range 
    path_filter = filter(
        lambda path: "BOOK" in path.upper(), path_list)    
    path_filter = filter(
        lambda path: start_date <= path.split("_")[-2] <= end_date, path_filter)
    
    # load multiple paths using dask (similar to pandas interface)
    df = dd.read_csv(
        list(path_filter), 
        compression="gzip", 
        parse_dates=["TIMESTAMP_UTC"],
    ).compute()
    df = df.set_index("TIMESTAMP_UTC")
    
    return df

def fit_model(df:pd.DataFrame, start_date:str, end_date:str):
    """
    Fit RandomForestClassifier on preprocessed data. 

    :param df:
        pd.DataFrame, preprocessed data
    :return model:
        RandomForestClassifier, trained model instance
    :return scaler:
        StandardScaler, scaler instance with parameters inherent to the preprocessed data
    """
    
    # ...
    if start_date:
        df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]

    print(f"fit model ...")
        
    # create targets
    df["Mid"] = (df["L1-AskPrice"] + df["L1-BidPrice"]) / 2
    df["return"] = df["Mid"].pct_change()
    df["return"] = np.sign(df["return"]) # y=1 (up), y=-1 (down), y=0 (no change)
    df["return_next"] = df["return"].shift(-1) # make prediction task
    df = df.dropna()
    df = df.drop(["Mid", "return"], axis=1)

    # perform train-test-split
    df_train, df_test = train_test_split(df, test_size=0.1, shuffle=False)

    # handle imbalances
    train_balanced = df_train.groupby("return_next")
    n_min = train_balanced.size().min()
    train_balanced = train_balanced.sample(n_min) # IMPORTANT: this requires pandas 1.1.0 or later!

    # standardize the data: (x - mu) / sigma
    scaler = StandardScaler()
    x_train = train_balanced.iloc[:, :-1].values
    x_train = scaler.fit_transform(x_train)
    x_test = df_test.iloc[:, :-1].values
    x_test = scaler.transform(x_test)
    y_train = train_balanced["return_next"].values
    y_test = df_test["return_next"].values

    # estimate model
    model = RandomForestClassifier(n_estimators=100, criterion="entropy", ccp_alpha=0.005)
    model.fit(x_train, y_train)

    # training performance
    predictions = model.predict(x_train)
    acc_train = np.mean(predictions == y_train)
    # print(f"train acc.: {np.round(acc_train, 4)}")
    unique, counts = np.unique(predictions, return_counts=True)
    # print(dict(zip(unique, counts)))

    # test performance
    predictions = model.predict(x_test)
    acc_test = np.mean(predictions == y_test)
    # print(f"test acc.: {np.round(acc_test, 4)}")
    unique, counts = np.unique(predictions, return_counts=True)
    # print(dict(zip(unique, counts)))

    return model, scaler  # Return model and scaler

if __name__ == "__main__":

    # TODO: SELECT SOURCES. You may delete or comment out the rest.

    sources = [
        # BASF (this strategy is designed only for trading in one stock!)
        "BAS.BOOK", "BAS.TRADES", "BAS.NEWS",
    ]

    # TODO: INSTANTIATE YOUR ML MODEL. Only if you need to make predictions.

    # option 1: load trained model
    try:
        model = joblib.load("./randomforestmodel.joblib") # load trained model
        scaler = joblib.load("./scaler.joblib") # load scaling method
    # option 2: if not available, fit new model 
    except:
        # preprocess data
        df = get_data(
            sources=sources,
            start_date="2016-01-01",
            end_date="2016-01-04",
        )  
        # train model based on preprocessed data
        model, scaler = fit_model(
            df=df,
            start_date="2016-01-01",
            end_date="2016-01-04",
        )
        # save model, scaler
        joblib.dump(model, "./randomforestmodel.joblib") # save trained model
        joblib.dump(scaler, "./scaler.joblib") # save scaler 

    # TODO: SELECT DATE RANGE. Please use format 'YYYY-MM-DD'.

    start_date = "2016-03-01" 
    end_date = "2016-03-31"

    # TODO: INSTANTIATE YOUR TRADING AGENT. You may submit multiple agents. 

    agent = Agent(
        name="agent_example_3_aka_profitmaschine",
        default_quantity=100,
        model=model,
        scaler=scaler,
    )

    # instantiate generator
    generator = Generator(
        sources=sources,
        start_date=start_date,
        end_date=end_date,
    )
    # instantiate backtest
    backtest = Backtest(
        agent=agent,
        generator=generator,
    )

    # run backtest
    results = backtest.run(
        verbose=True,
        interval=1_000, # report every 1_000 events
    )
