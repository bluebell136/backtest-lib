# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import textwrap

class Market:

    instances = dict() # instance store
    timestamp_global = pd.NaT # most recent timestamp across all instances

    def __init__(self, market_id):
        """
        Market class, implemented as a stateful object in order to ensure
        price-time-priority in agent order execution. This means that liquidity
        denoted by (<timestamp>, <quantity>) is added to and removed from the
        price levels available in the market state. There are two states that
        are continually updated ...

        - _state (post-trade, consistent with historical data)
        - _match_bid, _match_ask (pre-trade, temporary)

        ... that reflect the post-trade market state (based on original data)
        and pre-trade market state (used to match agent orders against),
        respectively. In order to manipulate these states, this class implements
        two methods ...

        - update(self, book_state, trades_state)
        - match(self)

        ... that (1) update both pre-trade as well as post-trade state and then
        (2) match standing agent orders corresponding to either side. Also, this
        class implements a set of market statistics ...

        - timestamp (instance attribute)
        - timestamp_global (class attribute)
        - best_bid
        - best_ask
        - mid_point
        - tick_size (inferred from book_state)
        - volume (daily, based on historical trades)
        - vwap (daily, based on historical trades)

        ... that may be used by the agent.

        Note that, in this implementation, orders submitted by the agent DO NOT
        HAVE MARKET IMPACT!

        All market instances are stored in and may be accessed through the
        `instances` class attribute (dictionary).

        :param market_id:
            str, market identifier
        """

        # static attributes from arguments
        self.market_id = market_id

        # global attributes update
        self.__class__.instances.update({market_id: self})

        # original representation for book_state, trades_state
        self._book_state = pd.Series()
        self._trades_state = pd.Series()

        # dictionary representation for book_state in t-1, t
        self._book_last = dict()
        self._book_this = dict()

        # list representation for historical trades_state updates
        self._trades_historical = list()

        # updated attributes
        self._timestamp = pd.NaT
        self._best_bid = np.NaN
        self._best_ask = np.NaN
        self._mid_point = np.NaN
        self._tick_size = np.NaN

        # post-trade state, {<price>: [(<timestamp>, <quantity>), *], *}
        self._state = dict()

    # original input ---

    @property
    def book_state(self):
        """
        Current book_state in original format.

        :return book_state:
            pd.Series, ...
        """

        return self._book_state

    @property
    def trades_state(self):
        """
        Current trades_state in original format. May be empty given that the
        trades_state is optional.

        :return trades_state:
            pd.Series, ...
        """

        return self._trades_state

    # current statistics (based on book_state) ---

    @property
    def timestamp(self):
        """
        Current timestamp corresponding to the most recent market update.

        :return timestamp:
            pd.Timestamp, ...
        """

        return self._timestamp

    @property
    def best_bid(self):
        """
        Current best_bid based on book_state.

        :return best_bid:
            float, ...
        """

        return self._best_bid

    @property
    def best_ask(self):
        """
        Current best_ask based on book_state.

        :return best_ask:
            float, ...
        """

        return self._best_ask

    @property
    def mid_point(self):
        """
        Current mid_point based on book_state.

        :return mid_point:
            float, ...
        """

        return self._mid_point

    @property
    def tick_size(self):
        """
        Current tick_size based on book_state, inferred from price levels using
        their greatest common divisor.

        :return tick_size:
            float, ...
        """

        return self._tick_size

    # aggregated statistics (based on trades_state) ---

    @property
    def volume(self): # (timestamp, price, quantity) = x
        """
        Daily historical trading volume up to the current timestamp based on
        trades_state. Note that this value does not reflect agent-based trades.

        :return volume:
            int, ...
        """

        return sum(x[2] for x in self._trades_historical)

    @property
    def vwap(self): # (timestamp, price, quantity) = x
        """
        Daily historical VWAP up to the current timestamp based on trades_state.
        Note that this value does not reflect agent-based trades.

        :return vwap:
            float, ...
        """

        return sum(x[1] * x[2] for x in self._trades_historical) / self.volume

    # update market ---

    @staticmethod
    def _add_liquidity(liquidity_list, timestamp, quantity):
        """
        Add liquidity to a given liquidity_list, that is a partial quantity
        tagged with its corresponding timestamp.

        :param liquidity_list:
            list, (timestamp, quantity) tuples for a given price level
        :param timestamp:
            pd.Timestamp, timestamp to add
        :param quantity:
            int, liquidity to add
        :return liquidity_list:
            list, (timestamp, quantity) tuples + added liquidity
        """

        # bypass empty quantity
        if not quantity:
            return liquidity_list

        # convert to dictionary, timestamps are supposed to be unique
        liquidity = dict(liquidity_list)

        # aggregate added quantity with pre-existent quantity
        liquidity[timestamp] = liquidity.get(timestamp, 0) + quantity

        # convert to list of tuples
        liquidity_list = liquidity.items()

        # sort by timestamp
        liquidity_list = sorted(liquidity_list, key=lambda x: x[0])
        # remove liquidity with empty quantity
        liquidity_list = list(filter(lambda x: x[1], liquidity_list))

        return liquidity_list

    @staticmethod
    def _use_liquidity(liquidity_list, quantity):
        """
        Use liquidity from a given liquidity_list, starting with the quantity
        tagged with the oldest available timestamp.

        Note that quantity will never exceed liquidity. There are two cases to
        consider ...
        - self.update: cannot happen in historical data
        - self.match: controls for using more than what is available

        :param liquidity_list:
            list, (timestamp, quantity) tuples for a given price level
        :param quantity:
            int, quantity to use from liquidity_list
        :return liquidity_list:
            list, (timestamp, quantity) tuples - used liquidity
        """

        # bypass empty liquidity_list
        if not liquidity_list:
            return liquidity_list
        # bypass empty quantity
        if not quantity:
            return liquidity_list

        # determine used liquidity
        timestamp_list, quantity_list = zip(*liquidity_list)
        quantity_cumsum = np.cumsum(quantity_list)
        i = np.argwhere(quantity_cumsum >= quantity).flatten()[0]
        liquidity_list = liquidity_list[i+1:]

        # determine partial liquidity to prepend
        timestamp = timestamp_list[i]
        remainder = quantity_cumsum[i] - quantity
        insert = (timestamp, remainder)

        # prepend (timestamp, quantity_left) to liquidity_list
        if remainder:
            liquidity_list.insert(0, insert)

        # sort by timestamp
        liquidity_list = sorted(liquidity_list, key=lambda x: x[0])
        # remove liquidity with empty quantity
        liquidity_list = list(filter(lambda x: x[1], liquidity_list))

        return liquidity_list

    @staticmethod
    def _restore_liquidity(liquidity_list, liquidity_list_init, quantity):
        """
        Restore liquidity less than or equal to the liquidity used between
        last state (liquidity_list_init) and this state (liquidity_list).

        Other than _add_liquidity, _restore_liquidity includes the initial
        timestamps and is preprended to the liquidity_list.

        :param liquidity_list:
            list, (timestamp, quantity) tuples for a given price level (t)
        :param liquidity_list_init:
            list, (timestamp, quantity) tuples for a given price level (t-1)
        :param quantity:
            int, quantity to restore
        :return liquidity_list:
            list, (timestamp, quantity) tuples + restored liquidity
        :return quantity:
            int, remaining quantity surplus
        """

        # convert list to dictionary, timestamps must to be unique
        liquidity = dict(liquidity_list)
        liquidity_init = dict(liquidity_list_init)

        # ...
        for timestamp in sorted(liquidity_init):
            difference = max(
                liquidity_init.get(timestamp, 0) - liquidity.get(timestamp, 0),
                0 # proceed only if quantity_init (t-1) >= quantity (t)
            )
            restored = min(
                quantity, # remaining quantity
                difference # difference that can be restored
            )
            liquidity[timestamp] = liquidity.get(timestamp, 0) + restored
            quantity -= restored

        # convert to list of tuples
        liquidity_list = liquidity.items()

        # sort liquidity_list by timestamp
        liquidity_list = sorted(liquidity_list, key=lambda x: x[0])
        # remove liquidity with empty quantity
        liquidity_list = list(filter(lambda x: x[1], liquidity_list))

        return liquidity_list, quantity

    def update(self, book_state, trades_state):
        """
        Update both internal and external representation of the order book,
        that is `state_internal` (post-trade) and `state_external` (pre-trade).

        Use methods `add_liquidity` and `use_liquidity` in order to add and
        remove liquidity to and from a given price level.

        :param book_state:
            pd.Series, book data
        :param trades_state:
            pd.Series, trades data, aggregated per timestamp
        """

        # set original representation for book_state, trades_state
        self._book_state = book_state
        self._trades_state = trades_state # optional

        # get list representation
        timestamp, *book_state = book_state.values
        _, *trades_state = trades_state.values

        # set update-related attributes based on list representation
        self._timestamp = self.__class__.timestamp_global = timestamp
        self._best_bid = round(book_state[0], 3) # <L1-BidPrice>
        self._best_ask = round(book_state[2], 3) # <L1-AskPrice>
        self._mid_point = round((self._best_bid + self._best_ask) / 2, 3)

        # infer tick_size from price levels in book_state
        self._tick_size = np.gcd.reduce(
            (np.array(book_state)[0::2] * 1e3).astype(int)
        ) / 1e3 # identify greatest common divisor

        # set dictionary representation for t-1 (_book_last), t (_book_this)
        self._book_last = self._book_this
        self._book_this = dict(zip(book_state[0::2], book_state[1::2]))

        # function to return mid point given book_state as input dictionary (0 if empty)
        mid_point = lambda input: sum(
            list(input)[:2] # [<L1-BidPrice>, <L1-AskPrice>]
        ) / 2
        # function to test whether price is on opposite sides of last and this mid point
        opposite_side = lambda price: (
            (mid_point(self._book_this) - price) * (mid_point(self._book_last) - price)
        ) < 0

        # POST-TRADE STATE, {<price>: [(<timestamp>, <quantity>), *], *} ...

        # create deepcopy to later reconstruct timestamps in pre-trade state
        STATE_COPY_INIT = copy.deepcopy(self._state)

        # book_difference, [(<price>, <quantity>), *]
        book_difference = []

        # book_difference: compute ...
        for price in set(self._book_this) | set(self._book_last):
            # price not on opposite sides: remove (this - last) quantity
            if not opposite_side(price):
                book_difference.append(
                    (price, self._book_this.get(price, 0) - self._book_last.get(price, 0))
                )
            # price on opposite sides: remove last quantity, add this quantity
            if opposite_side(price):
                book_difference.append( # remove
                    (price, self._book_last.get(price, 0) * (-1))
                )
                book_difference.append( # add
                    (price, self._book_this.get(price, 0))
                )

        # book_difference: apply ...
        for price, qdiff in book_difference:
            # if positive qdiff: add liquidity to a given price level
            if qdiff > 0:
                self._state[price] = self._add_liquidity(
                    self._state.get(price, []),
                    self._timestamp, abs(qdiff),
                )
            # if negative qdiff: use liquidity from a given price level
            if qdiff < 0:
                self._state[price] = self._use_liquidity(
                    self._state.get(price, []),
                    abs(qdiff),
                )

        # PRE-TRADE STATE, {<price>: [(<timestamp>, <quantity>), *], *} ...

        # create deepcopy to isolate state from pre-trade changes
        STATE_COPY_POST = copy.deepcopy(self._state)

        # filter price levels on bid side (< mid_point), ask side (> mid_point)
        self._match_bid = {price: liquidity_list for price, liquidity_list
            in STATE_COPY_POST.items() if price < self._mid_point
        }
        self._match_ask = {price: liquidity_list for price, liquidity_list
            in STATE_COPY_POST.items() if price > self._mid_point
        }

        # trades_state: revert ...
        if isinstance(trades_state[0], list):
            for price, quantity in zip(*trades_state):

                # assign roles side_1st (standing side), side_2nd (matching side)
                if price < mid_point(self._book_last):
                    side_1st, side_2nd = self._match_bid, self._match_ask
                if price > mid_point(self._book_last):
                    side_1st, side_2nd = self._match_ask, self._match_bid

                # standing side (1): restore liquidity (t-1), use original timestamp(s)
                side_1st[price], surplus = self._restore_liquidity(
                    side_1st.get(price, []), STATE_COPY_INIT.get(price, []),
                    quantity,
                )
                # standing side (2): add liquidity (t), use current timestamp, only in case of surplus
                side_1st[price] = self._add_liquidity(
                    side_1st.get(price, []),
                    self._timestamp, surplus,
                )
                # matching side (1): add liquidity (t), use current timestamp
                side_2nd[price] = self._add_liquidity(
                    side_2nd.get(price, []),
                    self._timestamp, quantity,
                )

                # keep track of historical trade
                self._trades_historical.append(
                    (self._timestamp, price, quantity)
                )

        # sort price levels on bid side (DESCENDING), ask side (ASCENDING)
        self._match_bid = dict(
            sorted(self._match_bid.items(), reverse=True)
        )
        self._match_ask = dict(
            sorted(self._match_ask.items(), reverse=False)
        )

    # match orders against market ---

    def _match_limit(self, order, state, state_compete, side, limit):
        """
        Match limit order, use only liquidity given at price levels better than
        the specified order limit. Note that this method receives only sorted
        orders and market state, corresponding to each other in terms of side.
        Longer-standing liquidity on the competing side is given priority over
        agent order.

        :param order:
            Order, order instance with side corresponding to state
        :param state:
            dict, state filtered by side corresponding to order
        :param state_compete:
            dict, state filtered by side competing with order
        :param side:
            str, needed only to determine better_than operator
        :param limit:
            float, needed only to determine better_than operator
        :return state:
            dict, state after order excecution
        """

        # select operator to understand if price is better than limit
        better_than = lambda price, limit: {
            "buy": np.less_equal,
            "sell": np.greater_equal,
        }[side](price, limit)

        # ...
        for price, liquidity_list in state.items():

            # break matching algorithm when price is worse than limit
            if not better_than(price, limit):
                break

            # determine how much quantity can be used by agent order
            quantity_available = sum(q for _, q in liquidity_list)
            quantity_blocked = sum(q for t, q in state_compete.get(price, [])
                if t <= order.timestamp # standing orders are prioritized
            )
            quantity_available = max(0, quantity_available - quantity_blocked)
            quantity_used = min(quantity_available, order.quantity_left)

            # execute (partial) order at this price level
            if quantity_used:
                order.execute(self._timestamp, quantity_used, price)

            # take liquidity
            state[price] = self._use_liquidity(state[price], quantity_used)

        return state

    def _match_market(self, order, state, state_compete):
        """
        Match market order, use all available liquidity. Note that this method
        receives only sorted orders and market_state, corresponding to each
        other in terms of side. Longer-standing liquidity on the competing side
        is given priority over agent order.

        :param order:
            Order, order instance with side corresponding to state
        :param state:
            dict, state filtered by side corresponding to order
        :param state_compete:
            dict, state filtered by side competing with order
        :return state:
            dict, state after order excecution
        """

        # ...
        for price, liquidity_list in state.items():

            # determine how much quantity can be used by agent order
            quantity_available = sum(q for _, q in liquidity_list)
            quantity_blocked = sum(q for t, q in state_compete.get(price, [])
                if t <= order.timestamp # standing orders are prioritized
            )
            quantity_available = max(0, quantity_available - quantity_blocked)
            quantity_used = min(quantity_available, order.quantity_left)

            # execute (partial) order at this price level
            if quantity_used:
                order.execute(self._timestamp, quantity_used, price)

            # take liquidity
            state[price] = self._use_liquidity(state[price], quantity_used)

        return state

    def match(self):
        """
        Match standing buy orders against ask state, and standing sell orders
        against bid state.

        Use methods `match_market` and `match_limit` to differentiate between
        market and limit orders.
        """

        # match buy orders against ask state
        for order in self.orders_buy:
            # limit order
            if order.limit:
                self._match_ask = self._match_limit(order,
                    state=self._match_ask, state_compete=self._match_bid,
                    side=order.side, limit=order.limit,
                )
            # market order
            else:
                self._match_ask = self._match_market(order,
                    state=self._match_ask, state_compete=self._match_bid,
                )

        # match sell orders against bid state
        for order in self.orders_sell:
            # limit order
            if order.limit:
                self._match_bid = self._match_limit(order,
                    state=self._match_bid, state_compete=self._match_ask,
                    side=order.side, limit=order.limit,
                )
            # market order
            else:
                self._match_bid = self._match_market(order,
                    state=self._match_bid, state_compete=self._match_ask,
                )

    # reset market ---

    def reset(self):
        """
        Run reset routine on method call.
        """

        # reset trades on daily basis
        del self._trades_historical[:]

    # available orders ---

    @property
    def orders(self):
        """
        View based on Order.history, includes all AGENT orders filtered by
        status 'ACTIVE', older-than-current timestamp and corresponding
        market_id.

        :return orders:
            list, filtered Order instances
        """

        orders = Order.history

        # orders must have status 'ACTIVE'
        orders = filter(lambda order: order.status == "ACTIVE", orders)
        # orders must have been submitted before current timestamp
        orders = filter(lambda order: order.timestamp <= self._timestamp, orders)
        # orders must have corresponding market_id
        orders = filter(lambda order: order.market_id == self.market_id, orders)

        return list(orders)

    @property
    def orders_buy(self):
        """
        View based on orders, includes all AGENT buy orders, sorted according
        to price-time-priority.

        :return orders:
            list, filtered and sorted buy Order instances
        """

        orders = self.orders

        # filtered by bid side
        orders = filter(lambda order: order.side == "buy", orders)
        # sort by (1) limit DESCENDING and (2) time ASCENDING
        orders = sorted(orders, key=lambda x: x.timestamp)
        orders = sorted(orders, key=lambda x: x.limit or min(self._match_ask), reverse=True)

        return list(orders)

    @property
    def orders_sell(self):
        """
        View based on orders, includes all AGENT sell orders, sorted according
        to price-time-priority.

        :return orders:
            list, filtered and sorted sell Order instances
        """

        orders = self.orders

        # filtered ask side
        orders = filter(lambda order: order.side == "sell", orders)
        # sort by (1) limit ASCENDING and (2) time ASCENDING
        orders = sorted(orders, key=lambda x: x.timestamp)
        orders = sorted(orders, key=lambda x: x.limit or max(self._match_bid), reverse=False)

        return list(orders)

    # string representation ---

    def __str__(self):
        """
        Provide string representation based on post-trade state.
        """

        market_id = self.market_id
        timestamp = self._timestamp

        best_bid = self._best_bid
        best_bid_size = self._book_state["L1-BidSize"]
        best_ask = self._best_ask
        best_ask_size = self._book_state["L1-AskSize"]

        mid_point = self._mid_point
        last_price = (self._trades_historical[-1:][0][1]
            if self._trades_historical else np.NaN
        )

        # string representation
        string = textwrap.dedent(f"""
        ---
        market_id:      {market_id}
        timestamp:      {timestamp}
        ---
        L1-BidPrice:    {best_bid}
        L1-BidSize:     {best_bid_size}
        L1-AskPrice:    {best_ask}
        L1-AskSize:     {best_ask_size}
        ---
        mid_point:      {mid_point}
        last_price:     {last_price}
        """)

        return string

class Order:

    history = list() # instance store

    def __init__(self, timestamp, market_id, side, quantity, limit=None):
        """
        Instantiate order.

        Note that an order can have different statuses:
        - 'ACTIVE': default
        - 'FILLED': set in Order.execute when there is no quantity left
        - 'CANCELLED': set in Order.cancel
        - 'REJECTED': set in Order.__init__

        Note that all order instances are stored in and may be accessed through
        the `history` class attribute (list).

        :param timestamp:
            pd.Timestamp, date and time that order was submitted
        :param market_id:
            str, market identifier
        :param side:
            str, either 'buy' or 'sell'
        :param quantity:
            int, number of shares ordered
        :param limit:
            float, limit price to consider, optional
        """

        # static attributes from arguments
        self.timestamp = timestamp
        self.market_id = market_id
        self.side = side
        self.quantity = quantity
        self.limit = limit
        self.order_id = len(self.__class__.history)

        # dynamic attributes
        self.quantity_left = quantity
        self.status = "ACTIVE"
        self.related_trades = []

        # assert order parameters
        try:
            self._assert_params()
        # set status 'REJECTED' if parameters are invalid
        except Exception as error:
            print("(SYSTEM) order {order_id} was rejected: {error}".format(
                order_id=self.order_id,
                error=error,
            ))
            self.status = "REJECTED"
        # ...
        else:
            print("(SYSTEM) order {order_id} was accepted: {self}".format(
                order_id=self.order_id,
                self=self,
            ))

        # global attributes update
        self.__class__.history.append(self)

    def _assert_params(self):
        """
        Assert market-related order parameters and provide information about
        an erroneous order submission. Note that program execution is supposed
        to continue.
        """

        # first, assert that market exists
        assert self.market_id in Market.instances, \
            "market_id '{market_id}' is not available".format(
                market_id=self.market_id,
            )

        # ...
        tick_size = Market.instances[self.market_id].tick_size
        timestamp = Market.instances[self.market_id].timestamp

        # assert that market state is available
        assert not pd.isnull(timestamp), \
            "trading is yet to start for market '{market_id}'".format(
                market_id=self.market_id
            )
        # assert that side is valid
        assert self.side in ["buy", "sell"], \
            "side can only take values 'buy' and 'sell', not '{side}'".format(
                side=self.side,
            )
        # assert that quantity is valid
        assert float(self.quantity).is_integer(), \
            "quantity can only take integer values".format(
                quantity=self.quantity,
            )
        # if specified, assert that limit is valid
        if self.limit:
            assert not (self.limit * 1e3) % (tick_size * 1e3), \
                "limit {limit} is too granular for tick_size {tick_size}".format(
                    limit=self.limit,
                    tick_size=tick_size,
                )

    def execute(self, timestamp, quantity, price):
        """
        Execute order. Note that an order is split into multiple trades if it
        is matched across multiple prices levels. The matching algorithm is
        implemented in the `Backtest` class.

        :param timestamp:
            pd.Timestamp, date and time that order was executed
        :param quantity:
            int, matched quantity
        :param price:
            float, matched price
        """

        # execute order (partially)
        trade = Trade(timestamp, self.market_id, self.side, quantity, price)
        self.related_trades.append(trade)

        # update remaining quantity
        self.quantity_left -= quantity

        # set status 'FILLED' if self.quantity_left is exhausted
        if not self.quantity_left:
            self.status = "FILLED"

    def cancel(self):
        """
        Delete remaining quantity of active order.
        """

        # set status 'CANCELLED' if order is still active
        if not self.status in ["CANCELLED, FILLED, REJECTED"]:
            self.status = "CANCELLED"

    def __str__(self):
        """
        Provide string representation.
        """

        string = "{side} {market_id} with {quantity}@{limit}, {time}".format(
            time=self.timestamp,
            market_id=self.market_id,
            side=self.side,
            quantity=self.quantity,
            limit=self.limit or 'market',
        )

        return string

class Trade:

    history = list() # instance store

    def __init__(self, timestamp, market_id, side, quantity, price):
        """
        Instantiate trade.

        Note that all trade instances are stored in and may be accessed through
        the `history` class attribute (list).

        :param timestamp:
            pd.Timestamp, date and time that trade was created
        :param market_id:
            str, market identifier
        :param side:
            str, either 'buy' or 'sell'
        :param quantity:
            int, number of shares executed
        :param price:
            float, price of shares executed
        """

        # static attributes from arguments
        self.timestamp = timestamp
        self.market_id = market_id
        self.side = side
        self.quantity = quantity
        self.price = price
        self.trade_id = len(self.__class__.history)

        # ...
        print("(SYSTEM) trade {trade_id} was executed: {self}".format(
            trade_id=self.trade_id,
            self=self,
        ))

        # global attributes update
        self.__class__.history.append(self)

    def __str__(self):
        """
        Provide string representation.
        """

        string = "{side} {market_id} with {quantity}@{price}, {time}".format(
            time=self.timestamp,
            market_id=self.market_id,
            side=self.side,
            quantity=self.quantity,
            price=self.price,
        )

        return string
