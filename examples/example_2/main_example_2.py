# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent import BaseAgent
from backtest import Backtest, Generator
from model import BaseModel

import dask.dataframe as dd
import datetime
import numpy as np
import os
import pandas as pd

import tensorflow as tf
tf.get_logger().setLevel("INFO") # suppress debugging information

class Agent(BaseAgent):

    def __init__(self, 
            name:str, 
            default_quantity:int,
            model:tf.keras.Model,
            steps:int,
        ):
        """
        Trading agent implementation example.

        :param name:
            str, agent name
        :param default_quantity:
            int, quantity to be used with every order
        :param model:
            tf.keras.Model, trained model instance
        :param steps:
            int, number of observations to consider in time-series
        """
        super(Agent, self).__init__(name)

        # static attributes from arguments
        self.default_quantity = default_quantity
        self.model = model
        self.steps = steps

        # dynamic attributes
        self.history = []

    def on_quote(self, market_id:str, book_state:pd.Series):
        """
        This method is called after a new quote.
        
        This method uses a model that returns two binary target variables:
        - significance: does average price over next steps exceed threshold?
        - direction: does next price move up or down?
        
        IMPORTANT: It would also be sensible to use multiple models and 
        orchestrate them in some way.
        
        IMPORTANT: When dealing with predictions for decision support, always
        try to implement safeguards to ensure that strange behaviour does not
        negatively affect your performance.
        
        IMPORTANT: As predictions are computationally expensive, it would make
        sense to use a two-step process:
        - first, use some very small (and therefore fast) model or some other
        way to determine whether an event is actually interesting
        - then, only if the event is interesting, make further (slower, more 
        expensive) predictions

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/size for 10 levels
        """

        # drop timestamp, create np.ndarray
        x = book_state.values[1:]
        x = np.array(x, dtype=np.float)
        
        # preprocess price and quantity separately
        x[0::2] = preprocess_price(x[0::2])
        x[1::2] = preprocess_quantity(x[1::2])
        
        # update history and limit to last <steps> steps
        self.history.append(x)
        self.history = self.history[-self.steps:]
        
        # skip if history does not yet contain <steps> steps
        if len(self.history) < self.steps:
            return
        
        # build sequence
        sequence = np.array(self.history, dtype=np.float)
        sequence = sequence.reshape((-1, 100, 40))
        
        # make predictions
        y_significance, y_direction = self.model.predict(sequence)[0]

        # if significance is predicted to be larger 
        if y_significance > 0.9:

            # predicted positive price trend: buy
            if y_direction > 0.51:
                self.submit_order(market_id, "buy", self.default_quantity)

            # predicted negative price trend: sell
            elif y_direction < 0.49:
                self.submit_order(market_id, "sell", self.default_quantity)

        # ...

    def on_trade(self, market_id:str, trades_state:pd.Series):
        pass

    def on_news(self, market_id:str, news_state:pd.Series):
        pass

    def on_time(self, timestamp:pd.Timestamp, timestamp_next:pd.Timestamp):
        pass

class Model(BaseModel):

    def __init__(self, name):
        """
        Neural network implementation.

        :param name:
            str, model name
        """
        super(Model, self).__init__(name)
        
        pass

    def build(self, input_shape):
        """
        Set as instance attributes all layers to be used in call method.

        :param input_shape:
            tuple, input shape where first dimension is batch_size
        """
        
        # LSTM layer takes sequence and returns dense representation
        self.layer_1 = tf.keras.layers.LSTM(
            units=100, 
            return_sequences=False,
        )
        
        # output layer needs two units for two target variables
        self.layer_2 = tf.keras.layers.Dense(
            units=2, 
            activation="sigmoid",
        )

    def call(self, input_tensor):
        """
        Implement a single forward pass using the defined layers.

        :param x:
            tf.Tensor, batch
        """
        
        # ...
        x = self.layer_1(input_tensor)
        x = self.layer_2(x)

        return x

class SequenceGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,
            batch_size, 
            steps, 
            shuffle=False, 
            limit=None,
            data=None, 
            targets=None, 
            flag=None, 
        ):
        """
        Simple generator that yields batches of sequential data and targets.
        
        :param batch_size:
            int, number of sequences per batch
        :param steps:
            int, number of observations per sequence
        :param shuffle:
            bool, shuffle batches? (note that this does not shuffle samples!)
        :param limit:
            int, limit the total number of batches
        :param data:
            np.ndarray, input data to be transformed into sequences
        :param targets:
            np.ndarray, input targets
        :param flag:
            np.ndarray, discontinuity flag for gaps in the data, optional
        """
        
        # from arguments
        self.batch_size = batch_size
        self.steps = steps
        self.shuffle = shuffle
        self.limit = limit
        
        # set attributes
        self.set_data(data)
        self.set_targets(targets)
        self.set_indexer(flag) # must be updated after data

    def set_data(self, data):
        """
        Set data, using numpy stride_tricks to efficiently generate sequences,
        that is at indices i..i+steps. 
        
        :param data:
            np.ndarray, input data to be transformed into sequences
        """
        
        # generate view with index range 0..n-steps+1
        shape = (data.shape[0] - self.steps + 1, self.steps) + data.shape[1:]
        strides = (data.strides[0],) + data.strides
        
        # use `as_strided` to avoid copying data
        data = np.lib.stride_tricks.as_strided(data, shape, strides)
        data = data[:-1] # index range 0..n-steps(+1-1)
        
        self.data = data
    
    def set_targets(self, targets):
        """
        Set targets corresponding to data, that is at index i+steps+1.
        
        :param targets:
            np.ndarray, input targets
        """
        
        # ....
        self.targets = targets[self.steps:] # index range steps..n
        
    def set_indexer(self, flag):
        """
        Set indexer that is used to access batches of data and targets.
        
        :param flag:
            np.ndarray, discontinuity flag for gaps in the data, optional
        """
        
        # locate flagged indices >= offset
        if flag is not None:
            flagged = np.arange(-self.steps, self.steps + 1)[None, :] + \
                np.flatnonzero(flag)[:, None]
            flagged = np.unique(flagged)
            flagged = flagged[flagged >= self.steps]
        else:
            flagged = np.empty((0,))
        
        # extend flagged indices to (0,) + flagged + (length,)
        flagged = np.insert(flagged, 0, 0)
        flagged = np.insert(flagged, flagged.shape[0], self.data.shape[0])

        # create batches based on (start, end) index tuples
        indexer = [(
            np.arange(flagged[i], flagged[i+1] - self.batch_size,
                self.batch_size)[None, :] + \
            np.arange(0, self.batch_size * 2,
                self.batch_size)[:, None]).T
            for i in np.arange(flagged.size - 1)]
        
        # ensure that indexer is integer-based
        indexer = np.concatenate(indexer, axis=0)
        indexer = indexer.astype(int)
        
        # shuffle batches
        if self.shuffle:
            np.random.shuffle(indexer) # inplace shuffling!
            
        # LIMIT NUMBER OF GENERATED BATCHES MANUALLY!
        if self.limit:
            indexer = indexer[:self.limit]
        
        self.indexer = indexer
    
    def __getitem__(self, batch_index):
        """
        Get generator contents at batch_index. 
        
        Note that this behaviour could also be achieved by implementing both
        `__iter__` and `__next__` method.
        """
        
        # get start, end positions for batch
        start, end = self.indexer[batch_index]
        
        # get batch
        data = self.data[start:end]
        targets = self.targets[start:end]
        
        return data, targets
    
    def __len__(self):
        """
        Return total number of batches. This is necessary because, when 
        iterating over the SequenceGenerator instance, the for loop running
        inside tf.keras.Model.fit wants to know how many batches to expect.
        """
        
        # ...
        return self.indexer.shape[0] # number of batches

# utility functions ---

def get_data(start_date, end_date):
    """
    Get data for a given date range.
    
    :param start_date:
        str, start date in format YYYY-MM-DD
    :param end_date:
        str, end date in format YYYY-MM-DD
    """
    
    print(f"load data for all stocks and date range [{start_date}, {end_date}] ...")
    
    # transform dates from YYYY-MM-DD to YYYYMMDD
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")

    # identify all paths available in directory
    path_list = [os.path.join(pre, f) for pre, _, sub 
        in os.walk("./data") for f in sub if not f.startswith((".", "_"))
    ]
    
    path_filter = path_list
    
    # filter paths to include all BOOK files for specified date range
    path_filter = filter(
        lambda path: "BOOK" in path.upper(), path_filter)    
    path_filter = filter(
        lambda path: start_date <= path.split("_")[-2] <= end_date, path_filter)

    path_list = list(path_filter)

    # load multiple paths using dask (similar to pandas interface)
    df = dd.read_csv(path_list, compression="gzip").compute()

    # drop datetime column
    df = df.drop(["TIMESTAMP_UTC"], axis=1)

    return df

def get_labels(df:pd.DataFrame):
    """
    Get labels for a given dataframe.
    
    IMPORTANT: Significance of a price trend could be measured in many 
    different ways, this is just a random example. 
    
    :param df:
        ..., ...
    """
    
    print(f"get labels ...")

    # get L1 bid-, ask-, and mid-price
    bid, ask = df["L1-BidPrice"], df["L1-AskPrice"]
    mid = (bid + ask) / 2 
    mid = mid * (bid != 0) * (ask != 0) # 0 if either side is 0

    # label 1: y_significance ---

    # get rolling mean for the <significance_window> steps after each price
    mid_mean = mid.iloc[::-1] # reverse
    mid_mean = mid_mean.rolling(SIGNIFICANCE_WINDOW).mean() # rolling mean
    mid_mean = mid_mean.iloc[::-1] # reverse again

    # generate label
    label = abs(mid_mean - mid) >= SIGNIFICANCE_THRESHOLD
    y_significance = label.astype(int)
    
    # report class imbalance
    class_imbalance = y_significance.sum() / len(y_significance.index)
    print(f"class imbalance for y_significance: {class_imbalance}".format(
        class_imbalance=class_imbalance,
    ))

    # label 2: y_direction ---

    # get mid-price direction
    mid_delta = mid.diff(1).fillna(1)
    mid_dir = np.sign(mid_delta)
    
    # generate label
    shift = mid != mid.shift(1)
    label = mid_dir.where(shift, np.nan).fillna(method="bfill") # backward-fill
    y_direction = label.replace(-1, 0).astype(int)
    
    # report class imbalance
    class_imbalance = y_direction.sum() / len(y_direction.index)
    print(f"class imbalance for y_direction: {class_imbalance}".format(
        class_imbalance=class_imbalance,
    ))

    return pd.DataFrame({
        "LABEL_SIGNIFIANCE": y_significance,
        "LABEL_DIRECTION": y_direction,
    })

def preprocess_price(x:np.ndarray):
    """
    Preprocess price-based data.
    
    :param x:
        np.ndarray, unprocessed data
    :return x:
        np.mdarray, scaled data
    """

    lower_limit = 0
    upper_limit = 300

    return (x - lower_limit) / (upper_limit - lower_limit)

def preprocess_quantity(x:np.ndarray):
    """
    Preprocess quantity-based data.
    
    :param x:
        np.ndarray, unprocessed data
    :return x:
        np.mdarray, scaled data
    """

    lower_limit = 0
    upper_limit = 50_000

    return (x - lower_limit) / (upper_limit - lower_limit)

def fit_model(model):
    """
    Fit model.
    
    IMPORTANT: Using the keras TimeseriesGenerator, your time-series will 
    include gaps (auctions, between days), which is technically wrong. Please 
    come up with a solution to this issue yourself! 
    
    IMPORTANT: Using class weights with multi-label classification is not
    necessarily trivial - we set our parameters to approximately achieve class
    balance in both labels. However, this is meant for demonstration purposes
    and does not constitute a good solution!

    :param model:
        tf.keras.Model, untrained Model instance
    """

    # load data for train, valid period as pd.DataFrame
    x_train = get_data(
        start_date="2016-01-04", 
        end_date="2016-01-04",
    )
    y_train = get_labels(x_train)

    # filter NaN rows, transform pd.DataFrame to np.ndarray
    keep_these_rows = ~ y_train.isna().any(axis=1)
    x_train = x_train.loc[keep_these_rows].values
    y_train = y_train.loc[keep_these_rows].values

    # preprocess price- and quantity-based features separately
    x_train[0::2] = preprocess_price(x_train[0::2])
    x_train[1::2] = preprocess_quantity(x_train[1::2])
    
    # instantiate data generator
    input_train = SequenceGenerator(
        batch_size=128,
        steps=SEQUENCE_LENGTH,
        shuffle=True,
        limit=1_000, # use only 1_000 batches
        data=x_train,
        targets=y_train,
        flag=None,
    )

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["binary_accuracy"],
    )

    # fit model based on data generator
    model.fit(x=input_train, epochs=1)

    # save model, subclassed model requires format 'tf' instead of 'h5'
    model.save("./trained_model.tf", save_format="tf")

    return model

def evaluate_model(model):
    """
    Evaluate model.

    IMPORTANT: Using the keras TimeseriesGenerator, your time-series will 
    include gaps in between days, which is technically wrong. Please come
    up with a solution to this issue yourself. 

    :param model:
        tf.keras.Model, trained Model instance
    """

    # load data for test period as pd.DataFrame
    x_test = get_data(
        start_date="2016-02-01", 
        end_date="2016-02-01",
    )
    y_test = get_labels(x_test)

    # filter NaN rows, transform pd.DataFrame to np.ndarray
    keep_these_rows = ~ y_test.isna().any(axis=1)
    x_test = x_test.loc[keep_these_rows].values
    y_test = y_test.loc[keep_these_rows].values

    # preprocess price- and quantity-based features separately
    x_test[0::2] = preprocess_price(x_test[0::2])
    x_test[1::2] = preprocess_quantity(x_test[1::2])

    # instantiate data generator
    input_test = SequenceGenerator(
        batch_size=128,
        steps=SEQUENCE_LENGTH,
        shuffle=True,
        limit=1_000, # use only 1_000 batches
        data=x_test,
        targets=y_test,
        flag=None, 
    )

    # evaluate model based on data generator
    model.evaluate(x=input_test)

if __name__ == "__main__":

    # TODO: SELECT SOURCES. You may delete or comment out the rest.

    sources = [
        # BASF
        "BAS.BOOK", "BAS.TRADES", "BAS.NEWS",
        # Bayer
        "BAY.BOOK", "BAY.TRADES", "BAY.NEWS",
        # ...
    ]

    # TODO: INSTANTIATE YOUR ML MODEL. Only if you need to make predictions.
    
    # global constants accessible in entire file
    # settings are for demonstration purposes only, not necessarily sensible!
    SEQUENCE_LENGTH = 100
    SIGNIFICANCE_THRESHOLD = 0.05 
    SIGNIFICANCE_WINDOW = 5000
    
    # option 1: load trained model
    try:
        model = tf.keras.models.load_model("./trained_model.tf")
    # option 2: if not available, fit new model 
    # please note that training a neural network can take a long time!
    except:
        model = Model(
            name="model_example_2",
        )
        model = fit_model(model)
    # in either case: evaluate model
    finally:
        print("evaluate model ...")
        evaluate_model(model)

    # TODO: SELECT DATE RANGE. Please use format 'YYYY-MM-DD'.

    start_date = "2016-03-01"
    end_date = "2016-03-31"

    # TODO: INSTANTIATE YOUR TRADING AGENT. You may submit multiple agents.

    agent = Agent(
        name="agent_example_2",
        default_quantity=100,
        model=model,
        steps=SEQUENCE_LENGTH,
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


