# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from market import Market, Order, Trade

import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

"""
Given below are the columns and dtypes corresponding to the different source
types 'BOOK', 'TRADES' and 'NEWS'.

- 'BOOK' is based on TRTH book data and includes timestamp (UTC) as well as
limit order book levels 1-10, each with bid price, bid size, ask price and
ask size.

  "TIMESTAMP_UTC": pd.Timestamp,
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

  "TIMESTAMP_UTC": pd.Timestamp,
  "Price": [<float>, *],
  "Volume": [<int>, *],

- 'NEWS' is based on scraped twitter data related to the observed stocks and
includes timestamp (UTC) as well as language of a tweet, the tweet itself, the
number of retweets, favorites, comments and times quoted.

  "TIMESTAMP_UTC": pd.Timestamp,
  "language": [<str>, *],
  "text": [<str>, *],
  "retweets": [<int>, *],
  "favorites": [<int>, *],
  "comments": [<int>, *],
  "quoted": [<int>, *],

"""

class Backtest:

    timestamp_global = None # most recent timestamp across all sources

    def __init__(self, agent, generator):
        """
        Backtest evaluates a trading agent based on a data generator instance
        that yields events based on a set of sources.

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
        Update market state and match standing orders.

        :param market_id:
            str, market identifier
        :param step:
            int, backtest step
        """

        # get corresponding book, trades source
        source_book = self.sources[f"{market_id}.BOOK"]
        source_trades = self.sources[f"{market_id}.TRADES"]

        # update market state based on historical data
        self.markets[market_id].update(
            book_state=source_book.iloc[step, :],
            trades_state=source_trades.iloc[step, :],
        )
        # match standing agent orders against pre-trade state
        self.markets[market_id].match()

    def agent_step(self, source_id, step):
        """
        Alert agent by sending any event through the corresponding method.

        :param source_id:
            str, source identifier
        :param step:
            int, backtest step
        """

        # get market_id
        market_id = source_id.split(".")[0]
        # get event required to alert agent
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
        # will fail for the last step per monitor frame
        except:
            pass

    def run(self, verbose=True, interval=100):
        """
        Iterate over sources and update market_state.

        :param verbose:
            bool, print updates with each iteration, default is True
        """

        # print before backtest ..
        print("\n(INFO) start backtest ...\n")

        # ...
        for sources, monitor in self.generator:

            # update data
            self.sources = sources
            self.monitor = monitor

            # ...
            for step, timestamp, *monitor_state in self.monitor.itertuples():

                # update global timestamp
                self.__class__.timestamp_global = timestamp

                # get source_id per updated source
                updated_sources = (monitor
                    .iloc[:, 1:]
                    .columns[monitor_state]
                    .values
                )
                # get market_id for book event (trades event is optional)
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

                # print agent status
                if verbose and not step % interval:
                    print(self.agent)

            # run reset routine before each data update
            for _, market in self.markets.items():
                market.reset()

        # print after backtest ...
        print("\n(INFO) pnl_realized: {pnl_1}, pnl_unrealized: {pnl_2}\n".format(
            pnl_1=self.agent.pnl_realized,
            pnl_2=self.agent.pnl_unrealized,
        ))

class Generator:

    def __init__(self, sources, start_date, end_date):
        """
        Generator class that yields source data one day at a time, in order to
        save memory.

        :param sources:
            list, [<source_id>, *]
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
        self.directory = os.path.join(
            os.path.dirname(__file__), # preprend to get absolute path
            "data" # instead of "./data"
        )
        self.time_delta = pd.Timedelta(1, "D")

    @staticmethod
    def _load_sources(directory, sources, date):
        """
        Load .csv(.gz) and .json files into dataframes and store them in a
        dictionary together with their corresponding key.

        :param directory:
            str, directory to load files from
        :param sources:
            dict, [<source_id>, *]
        :param date:
            pd.Timestamp, date to filter data by
        :return sources_output:
            dict, {<source_id>: <pd.DataFrame>, *}
        """

        datetime = "TIMESTAMP_UTC"
        sources_output = dict()

        # identify all paths available in directory
        path_list = [os.path.join(pre, f) for pre, _, sub
            in os.walk(directory) for f in sub if not f.startswith((".", "_"))
        ]

        # ...
        for source_id in sources:

            # identify matching criteria
            market_id, event_id = source_id.split(".")
            date_string = str(date.date()).replace("-", "")

            path_filter = path_list

            # require matching market_id
            path_filter = filter(
                lambda path: market_id.lower() in path.lower(), path_filter)
            # require matching event_id
            path_filter = filter(
                lambda path: event_id.lower() in path.lower(), path_filter)
            # require matching date
            path_filter = filter(
                lambda path: date_string in path, path_filter)

            path_filter = list(path_filter)

            # there should be exactly one matching path
            if path_filter:
                source_path = path_filter[0]
            # otherwise, raise Exception
            else:
                raise Exception("(ERROR) found no data for {source_id}".format(
                    source_id=source_id,
                ))

            # load event_id 'BOOK' as .csv(.gz)
            if "BOOK" in source_id:
                df = pd.read_csv(source_path, parse_dates=[datetime])
            # load event_id 'TRADES' as .json
            if "TRADES" in source_id:
                df = pd.read_json(source_path, convert_dates=True)
            # load event_id 'NEWS' as .json
            if "NEWS" in source_id:
                df = pd.read_json(source_path, convert_dates=True)

            # if dataframe is empty, break
            if not len(df.index) > 0:
                break

            # make timestamp timezone-unaware
            df[datetime] = pd.DatetimeIndex(df[datetime]).tz_localize(None)

            # add dataframe to output dictionary
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
        date_end, and yield both sources dictionary and monitor dataframe.

        :yield sources:
            dict, {<source_id>: <pd.DataFrame>, *}, with aligned timestamp
        :yield monitor:
            pd.DataFrame, changes per source and timestamp
        """

        while self.start_date <= self.end_date:

            # try to load, process and yield data
            try:
                print("(INFO) load data for {date} ...".format(
                    date=self.start_date.date(),
                ))
                sources = self._load_sources(
                    directory=self.directory,
                    sources=self.sources,
                    date=self.start_date,
                )
                sources = self._align_sources(sources)
                monitor = self._monitor_sources(sources)
                yield sources, monitor
            # should no data be available, pass
            except:
                pass
            # continue with next date
            finally:
                self.start_date += self.time_delta

    def __next__(self):
        return self


