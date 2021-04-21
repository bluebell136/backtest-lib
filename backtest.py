#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from market import Market, Order, Trade
from ast import literal_eval

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

"""
Given below are the columns and dtypes corresponding to the different source
types 'BOOK', 'TRADES' and 'NEWS'.

- 'BOOK' is based on TRTH book data and includes timestamp (UTC) as well as
limit order book levels 1-10, each with bid price, bid size, ask price and
ask size.

  "TIMESTAMP_UTC": str,
  "L1-BidPrice": float, "L1-BidSize": int,
  "L1-AskPrice": float, "L1-AskSize": int,
  "L2-BidPrice": float, "L2-BidSize": int,
  "L2-AskPrice": float, "L2-AskSize": int,
  "L3-BidPrice": float, "L3-BidSize": int,
  "L3-AskPrice": float, "L3-AskSize": int,
  "L4-BidPrice": float, "L4-BidSize": int,
  "L4-AskPrice": float, "L4-AskSize": int,
  "L5-BidPrice": float, "L5-BidSize": int,
  "L5-AskPrice": float, "L5-AskSize": int,
  "L6-BidPrice": float, "L6-BidSize": int,
  "L6-AskPrice": float, "L6-AskSize": int,
  "L7-BidPrice": float, "L7-BidSize": int,
  "L7-AskPrice": float, "L7-AskSize": int,
  "L8-BidPrice": float, "L8-BidSize": int,
  "L8-AskPrice": float, "L8-AskSize": int,
  "L9-BidPrice": float, "L9-BidSize": int,
  "L9-AskPrice": float, "L9-AskSize": int,
  "L10-BidPrice": float, "L10-BidSize": int,
  "L10-AskPrice": float, "L10-AskSize": int,

- 'TRADES' is based on TRTH trades data and includes timestamp (UTC) as well
as price and volume.

  "TIMESTAMP_UTC": str,
  "Price": object,
  "Volume": object,

- 'NEWS' is based on scraped twitter data related to the observed stocks and
includes timestamp (UTC) as well as the message body (...).

  "TIMESTAMP_UTC": str,
  "message": str,

"""

# backtest engine ---

class Backtest:

    def __init__(self, agent, generator):
        """
        Backtest evaluates a trading strategy based on multiple markets.

        :param agent:
            Agent, trading agent instance
        :param generator
            Generator, data generator instance
        """

        # from arguments
        self.agent = agent
        self.generator = generator

        # data updated chunk-wise by the generator
        self.sources = None
        self.monitor = None

        # identify symbols as market_ids
        symbols = set(source_id.split(".")[0] for source_id
            in self.generator.sources
        )
        # setup market instances
        for market_id in symbols:
            Market(market_id)
        # access market instances
        self.markets = Market.instances

    def market_step(self, market_id, step):
        """
        Update book_state and match standing orders.

        :param market_id:
            str, market identifier
        :param step:
            int, backtest step
        """

        # get book, trades state required to update book
        source_book = self.sources[f"{market_id}.BOOK"]
        source_trades = self.sources[f"{market_id}.TRADES"]

        # update book state based on historical data
        self.markets[market_id].update(
            book_state=source_book.iloc[step, :],
            trades_state=source_trades.iloc[step, :],
        )
        # match standing agent orders based on book state
        self.markets[market_id].match()

    def agent_step(self, source_id, step):
        """
        Alert agent by sending any updated state through the corresponding
        method.

        :param source_id:
            str, source identifier
        :param step:
            int, backtest step
        """

        # get market_id
        market_id = source_id.split(".")[0]
        # get state required to alert agent
        source_state = self.sources[source_id].iloc[step, :]

        # alert agent every time that book is updated
        if source_id.endswith("BOOK"):
            self.agent.on_quote(
                market_id,
                book_state=source_state,
            )
        # alert agent every time that trades happen
        if source_id.endswith("TRADES"):
            self.agent.on_trade(
                market_id,
                trades_state=source_state,
            )
        # alert agent every time that news happen
        if source_id.endswith("NEWS"):
            self.agent.on_news(
                market_id,
                news_state=source_state,
            )

        # alert agent with time interval between this and next step
        try:
            self.agent.on_time(
                timestamp=self.monitor.iloc[step, 0],
                timestamp_next=self.monitor.iloc[step+1, 0],
            )
        # will fail for the last timestep per monitor
        except:
            pass

    def run(self, verbose=True, interval=100):
        """
        Iterate through sources and update market_state.

        :param verbose:
            bool, print updates with each iteration, default is True
        """

        for sources, monitor in self.generator:

            # update data
            self.sources = sources
            self.monitor = monitor

            # ...
            for step, timestamp, *monitor_state in self.monitor.itertuples():

                # get source_id per updated source
                updated_sources = (monitor
                    .iloc[:, 1:]
                    .columns[monitor_state]
                    .values
                )
                # get market_id for book update (trades update is optional)
                updated_markets = [col.split(".")[0] for col in updated_sources
                    if col.endswith("BOOK")
                ]

                # step 1: update book_state -> based on original data
                # step 2: match standing orders -> based on pre-trade state
                for market_id in updated_markets:
                    self.market_step(market_id, step)

                # step 3: alert agent -> based on original data
                for source_id in updated_sources:
                    self.agent_step(source_id, step)

                # print status
                if verbose and not step % interval:
                    print(self.agent)

            # ...
            for market in self.markets.values():

                # run reset routine before each data update
                market.reset()

        # print after test ...
        print("DONE. pnl: {pnl_real}, pnl_unreal: {pnl_unreal}".format(
            pnl_real=self.agent.pnl_realized,
            pnl_unreal=self.agent.pnl_unrealized,
        ))

# data generator ---

class Generator:

    def __init__(self, sources, start_date, end_date):
        """
        Generator class that, in order to save memory, yields source data one
        day at a time.

        :param sources:
            dict, {<source_id>: <source_path>, *}
        :param start_date:
            str, date string, default is "2016-01-01"
        :param end_date:
            str, date string, default is "2016-03-31"
        """

        # from arguments
        self.sources = sources
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        # ...
        self.time_delta = pd.Timedelta(1, "D")

    @staticmethod
    def _load_sources(sources_input, date):
        """
        Load .csv files into dataframes and store them in a dictionary together
        with their corresponding key.

        :param sources_input:
            dict, {<source_id>: <source_path>, *}
        :param date:
            pd.Timestamp, date to filter data by
        :return sources_output:
            dict, {<source_id>: <pd.DataFrame>, *}
        """

        datetime = "TIMESTAMP_UTC"
        sources_output = {}

        for source_id, source_path in sources_input.items():

            # load book files as .csv(.gz)
            if "BOOK" in source_id:
                df = pd.read_csv(
                    source_path,
                    parse_dates=[datetime],
                )
            # load trades files as .json
            if "TRADES" in source_id:
                df = pd.read_json(
                    source_path,
                    convert_dates=True,
                )
            # load news files as .json
            if "NEWS" in source_id:
                df = pd.read_json(
                    source_path,
                    convert_dates=True,
                )

            # remove timezone from timestamp
            timestamp = df[datetime]
            df[datetime] = pd.DatetimeIndex(timestamp).tz_localize(None)
            # filter dataframe by date
            df = df.loc[df[datetime].dt.date == date.date()]

            # ...
            sources_output[source_id] = df

        return sources_output

    @staticmethod
    def _align_sources(sources):
        """
        Consolidate and split again all sources so that each source dataframe
        contains a state for each ocurring timestamp across all sources.

        :param sources:
            dict, {<source_id>: <pd.DataFrame>, *}, only original timestamps
        :return sources:
            dict, {<source_id>: <pd.DataFrame>, *}, aligned timestamps
        """

        datetime = "TIMESTAMP_UTC"

        # unpack dictionary
        id_list, df_list = zip(*sources.items())
        # rename columns and use id as prefix, exclude timestamp
        add_prefix = lambda id, df: df.rename(columns={x: f"{id}__{x}"
            for x in df.columns[1:]
        })
        df_list = list(map(add_prefix, id_list, df_list))
        # merge sources horizontally using full outer join
        df_merged = pd.concat([
            df.set_index(datetime) for df in df_list
        ], axis=1, join="outer").reset_index()

        # split merged_df into original df_list
        df_list = [pd.concat([
            df_merged[[datetime]], # timestamp
            df_merged[[x for x in df_merged.columns if id in x]
        ]], axis=1) for id in id_list]
        # rename columns and remove prefix, exclude timestamp
        del_prefix = lambda df: df.rename(columns={x: x.split("__")[1]
            for x in df.columns[1:]
        })
        df_list = list(map(del_prefix, df_list))
        # pack dictionary
        sources = dict(zip(id_list, df_list))

        return sources

    @staticmethod
    def _monitor_sources(sources):
        """
        In addition to the sources dict, return a monitor dataframe that keeps
        track of changes in state across all sources.

        :param sources:
            dict, {<source_id>: <pd.DataFrame>, *}, with aligned timestamp
        :return monitor:
            pd.DataFrame, changes per source and timestamp
        """

        datetime = "TIMESTAMP_UTC"

        # setup dictionary based on timestamp
        datetime_index = list(sources.values())[0][datetime]
        monitor = {datetime: datetime_index}

        # track changes per source and timestamp in series
        for key, df in sources.items():
            monitor[key] = ~ df.iloc[:, 1:].isna().all(axis=1)

        # build monitor as dataframe from series
        monitor = pd.DataFrame(monitor)

        return monitor

    def __iter__(self):
        """
        Iterate over each single date between the specified date_start and
        date_end, and yield

        :yield sources:
            dict, {<source_id>: <pd.DataFrame>, *}, with aligned timestamp
        :yield monitor:
            pd.DataFrame, changes per source and timestamp
        """

        while self.start_date <= self.end_date:

            # load data
            print("(SYSTEM) load data for {date} ...".format(
                date=self.start_date.date(),
            ))
            sources = self._load_sources(
                sources_input=self.sources,
                date=self.start_date
            )
            # process data
            sources = self._align_sources(sources)
            monitor = self._monitor_sources(sources)

            # yield sources, monitor if non-empty
            if len(monitor.index) > 0:
                yield sources, monitor

            # increment this_date
            self.start_date += self.time_delta

    def __next__(self):
        return self
