# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent import BaseAgent
from backtest import Backtest, Generator

import numpy as np
import pandas as pd

class Agent(BaseAgent):

    def __init__(self, name:str, 
            trading_qt:int, 
            threshold_open:float, 
            threshold_stop_loss:float, 
            *args, **kwargs,
        ):
        """
        Trading agent implementation.

        The backtest iterates over a set of sources and alerts the trading agent
        whenever a source is updated. These updates are conveyed through method
        calls to ...

        - on_quote(self, market_id, timestamp, book_state)
        - on_trade(self, market_id, timestamp, trade_state)
        - on_news(self, market_id, timestamp, news_state)
        - on_time(self, timestamp, timestamp_next)

        ..., all of which you are expected to implement yourself. In order to
        interact with a market, the trading agent may use the methods ...

        - submit_order(self, market_id, side, quantity, limit=None)
        - cancel_order(self, order)

        ... that create and delete orders waiting to be executed against the
        respective market's order book. Besides, this class implements a set
        of attributes ...

        - exposure (per market)
        - pnl_realized (per market)
        - pnl_unrealized (per market)
        - exposure_total
        - pnl_realized_total
        - pnl_unrealized_total
        - exposure_left
        - transaction_costs

        ... that may be used to monitor trading agent performance. The agent
        may also access attributes of related class instances, using the
        container attributes ...

        - orders -> [<Order>, *]
        - trades -> [<Trade>, *]
        - markets -> {<market_id>: <Market>, *}

        For more information, you may list all attributes and methods as well
        as access the docstrings available in the base class using
        `dir(BaseAgent)` and `help(BaseAgent.<method>)`, respectively.

        :param name:
            str, agent name
        :param trading_qt:
            int, ...
        :param threshold_open:
            float, ...
        :param threshold_stop_loss:
            float, ...
        :param reference_price:
            dict, ...
        :param eod:
            
        """
        super(Agent, self).__init__(name, *args, **kwargs)
        
        # TODO: YOUR IMPLEMENTATION GOES HERE

        # static attributes from arguments
        self.trading_qt = trading_qt
        self.threshold_open = threshold_open
        self.threshold_stop_loss = threshold_stop_loss
        
        # dynamic attributes
        self.reference_price = {}
        self.eod = False

    def on_quote(self, market_id:str, book_state:pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/quantity for ten levels
        """

        # TODO: YOUR IMPLEMENTATION GOES HERE

        # set reference price
        if market_id not in self.reference_price.keys():
            self.reference_price[market_id] = self.markets[market_id].mid_point
            print(f"{market_id}: reference price set to {self.reference_price[market_id]}")

        # open a position if neither order nor open position exists
        if self.exposure[market_id] == 0 and not self.get_filtered_orders(market_id, status='ACTIVE') and not self.eod:
            # ...
            if book_state.loc["L1-AskPrice"] < self.reference_price[market_id] * (1 - self.threshold_open):
                self.submit_order(market_id, "buy", self.trading_qt) # submit buy market order
            # ...
            elif book_state.loc["L1-BidPrice"] > self.reference_price[market_id] * (1 + self.threshold_open):
                self.submit_order(market_id, "sell", self.trading_qt) # submit sell market order

        # close a position if (1) price reverts (PROFIT), or (2) price increases/decreases further (LOSS)
        if self.exposure[market_id] > 0:
            if (book_state.loc["L1-BidPrice"] > self.reference_price[market_id] or
                    book_state.loc["L1-BidPrice"] < self.reference_price[market_id] * (1-self.threshold_stop_loss)):
                self.submit_order(market_id, "sell", self.trading_qt)
                self.reference_price[market_id] = self.markets[market_id].mid_point # udpate reference price
        # ...
        elif self.exposure[market_id] < 0:
            if (book_state.loc["L1-AskPrice"] < self.reference_price[market_id] or
                    book_state.loc["L1-AskPrice"] > self.reference_price[market_id] * (1 + self.threshold_stop_loss)):
                self.submit_order(market_id, "buy", self.trading_qt)
                self.reference_price[market_id] = self.markets[market_id].mid_point # udpate reference price

    def on_trade(self, market_id:str, trades_state:pd.Series):
        """
        This method is called after a new trade.

        :param market_id:
            str, market identifier
        :param trade_state:
            pd.Series, including timestamp, price, quantity
        """

        # TODO: YOUR IMPLEMENTATION GOES HERE

        pass

    def on_news(self, market_id:str, news_state:pd.Series):
        """
        This method is called after a news message.

        :param market_id:
            str, market identifier
        :param news_state:
            pd.Series, including timestamp, message
        """

        # TODO: YOUR IMPLEMENTATION GOES HERE

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

        # TODO: YOUR IMPLEMENTATION GOES HERE

        if timestamp.hour == 7 and self.eod:
            self.eod = False # start of (next) trading day

        if timestamp.hour == 15 and timestamp.minute > 25 and not self.eod:
            for market_id, market in self.markets.items():
                
                # cancel active orders
                if self.get_filtered_orders(market_id, status="ACTIVE"):
                    [self.cancel_order(o) for o in self.get_filtered_orders(market_id, status="ACTIVE")]

                # close position
                if self.exposure[market_id] > 0:
                    self.submit_order(market_id, "sell", self.trading_qt)
                if self.exposure[market_id] < 0:
                    self.submit_order(market_id, "buy", self.trading_qt)
            
            self.eod = True
            print("end of day is reached, close all positions")

if __name__ == '__main__':

    # TODO: SELECT SOURCES. You may delete or comment out the rest.

    sources = {
        # BASF
        "BAS.BOOK": "./data/period_2/Books_BAS_DE_20160401_20160430.csv.gz",
        "BAS.TRADES": "./data/period_2/Trades_BAS_DE_20160401_20160430.json",
        # BAYER
        "BAY.BOOK": "./data/period_2/Books_BAY_DE_20160401_20160430.csv.gz",
        "BAY.TRADES": "./data/period_2/Trades_BAY_DE_20160401_20160430.json",
        # ...
    }

    # TODO: SELECT DATE RANGE. Please use format 'YYYY-MM-DD'.

    start_date = "2016-04-01"
    end_date = "2016-04-04"

    # TODO: INSTANTIATE YOUR TRADING AGENT. You may submit multiple agents.

    agent = Agent(
        name="test_agent", 
        trading_qt=100, 
        threshold_open=0.003, 
        threshold_stop_loss=0.01,
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
    results = backtest.run(verbose=True, interval=100) 
