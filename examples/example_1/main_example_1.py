# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent import BaseAgent
from backtest import Backtest, Generator

import datetime
import numpy as np
import pandas as pd

class Agent(BaseAgent):

    def __init__(self, name:str,
            default_quantity:int,
            threshold_open:float,
            threshold_stop_loss:float,
        ):
        """
        Trading agent implementation example.

        :param name:
            str, agent name
        :param default_quantity:
            int, ...
        :param threshold_open:
            float, ...
        :param threshold_stop_loss:
            float, ...
        """
        super(Agent, self).__init__(name)

        # static attributes from arguments
        self.default_quantity = default_quantity
        self.threshold_open = threshold_open
        self.threshold_stop_loss = threshold_stop_loss

        # dynamic attributes
        self.reference_prices = {}
        self.eod = False # flag set for the last 5 minutes of a trading day

    def on_quote(self, market_id:str, book_state:pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/size for 10 levels
        """

        # first, set reference price if not yet listed
        if market_id not in self.reference_prices:
            self.reference_prices[market_id] = self.markets[market_id].mid_point

        # no position exists for this market, also (1) no active order exists and (2) time not past 15:25
        if not self.exposure[market_id] and not self.get_filtered_orders(market_id, status="ACTIVE") and not self.eod:

            # submit buy market order if: best ask < threshold
            if book_state["L1-AskPrice"] < self.reference_prices[market_id] * (1 - self.threshold_open):
                self.submit_order(market_id, "buy", self.default_quantity)

            # submit sell market order if: best bid > threshold
            elif book_state["L1-BidPrice"] > self.reference_prices[market_id] * (1 + self.threshold_open):
                self.submit_order(market_id, "sell", self.default_quantity)

        # long position exists for this market
        if self.exposure[market_id] > 0:

            # close long position if: (1) best bid > reference price (profit) or (2) best bid < threshold stop-loss
            if (book_state["L1-BidPrice"] > self.reference_prices[market_id] or # profit
                    book_state["L1-BidPrice"] < self.reference_prices[market_id] * (1 - self.threshold_stop_loss)): # loss
                self.submit_order(market_id, "sell", self.default_quantity)
                self.reference_prices[market_id] = self.markets[market_id].mid_point # udpate reference price

        # short position exists for this market
        elif self.exposure[market_id] < 0:

            # close short position if: (1) best ask < reference price (profit) or (2) best ask > threshold stop-loss
            if (book_state["L1-AskPrice"] < self.reference_prices[market_id] or # profit
                    book_state["L1-AskPrice"] > self.reference_prices[market_id] * (1 + self.threshold_stop_loss)): # loss
                self.submit_order(market_id, "buy", self.default_quantity)
                self.reference_prices[market_id] = self.markets[market_id].mid_point # udpate reference price

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

        # set flag if time is past 16:25, triggered only once
        if timestamp.time() >= datetime.time(16, 25) and not self.eod:
            for market_id, _ in self.markets.items():

                # cancel active orders for this market
                [self.cancel_order(order) for order in self.get_filtered_orders(market_id, status="ACTIVE")]

                # close positions for this market
                if self.exposure[market_id] > 0:
                    self.submit_order(market_id, "sell", self.default_quantity)
                if self.exposure[market_id] < 0:
                    self.submit_order(market_id, "buy", self.default_quantity)

            # ...
            self.eod = True

        # unset flag at start of (next) trading day
        if timestamp.day != timestamp_next.day:
            self.eod = False
            
        # ...

if __name__ == "__main__":

    # TODO: SELECT SOURCES. You may delete or comment out the rest.

    sources = [
        # BASF
        "BAS.BOOK", "BAS.TRADES", "BAS.NEWS",
        # Bayer
        "BAY.BOOK", "BAY.TRADES", "BAY.NEWS",
        # ...
    ]

    # TODO: SELECT DATE RANGE. Please use format 'YYYY-MM-DD'.

    start_date = "2016-03-01"
    end_date = "2016-03-31"

    # TODO: INSTANTIATE YOUR TRADING AGENT. You may submit multiple agents.

    agent = Agent(
        name="agent_example_1",
        default_quantity=100,
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
    results = backtest.run(
        verbose=True,
        interval=1_000, # report every 1_000 events
    )
