"""
TWAP (Time-Weighted Average Price) baseline.

TWAP is the textbook reference strategy for order execution: spread the parent
order evenly over the available time instead of dumping it all at once.

Why this is not simply "always send an order"
---------------------------------------------
The previous version returned action 0 on every step. With the current config
that behaves correctly only by coincidence: parent_order_size / order_fixed_size
= 1000 / 100 = 10 child orders, and the execution window happens to contain
exactly 10 steps. If order_fixed_size were changed to 200, "always trade" would
try to execute 2000 shares of a 1000-share parent order, and the baseline would
be silently wrong. This version computes the schedule from the config, so it
stays a real TWAP for any combination of parameters.
"""

import math

from abides_core.utils import str_to_ns


class TWAPAgent:
    """
    Deterministic schedule: by step k, the agent should have sent
    (k / total_steps) of the parent order. It sends one child order whenever it
    is behind that schedule, and holds otherwise.

    This agent does not learn anything. Its only purpose is to give a reference
    number that the RL agent must beat.
    """

    # Action ids of the markets-execution-v0 environment.
    ACTION_MARKET_ORDER = 0
    ACTION_LIMIT_ORDER = 1
    ACTION_HOLD = 2

    def __init__(self, config, use_limit_orders=False):
        """
        Arguments:
            config:           the loaded project config (see common.env_factory).
            use_limit_orders: if False (default) each child order is a market
                              order, which always fills but pays the spread.
                              If True, limit orders at the near touch are used
                              instead: cheaper, but they may not fill.
        """
        environment_config = config["environment"]

        self.parent_order_size = environment_config["parent_order_size"]
        self.order_fixed_size = environment_config["order_fixed_size"]

        # How many decision steps fit inside the execution window. The agent is
        # woken up once per timestep_duration, so this is simply the ratio of the
        # two durations. str_to_ns converts "00:10:00" / "60s" to nanoseconds.
        execution_window_ns = str_to_ns(environment_config["execution_window"])
        timestep_ns = str_to_ns(environment_config["timestep_duration"])
        self.total_steps = int(execution_window_ns / timestep_ns)

        self.trade_action = (
            self.ACTION_LIMIT_ORDER
            if use_limit_orders
            else self.ACTION_MARKET_ORDER
        )

        # How many child orders are needed to cover the whole parent order.
        self.orders_needed = math.ceil(
            self.parent_order_size / self.order_fixed_size
        )

        self.reset()

    def reset(self):
        """Clear the per-episode counter. Must be called before every episode."""
        self.step_index = 0
        self.quantity_sent = 0

    def predict(self, observation=None):
        """
        Return the action for the current step.

        The observation is ignored on purpose: TWAP is a fixed time schedule and
        does not react to the market. That is exactly what makes it a fair
        baseline for the RL agent, which does react.
        """
        self.step_index += 1

        # Target cumulative quantity that should have been sent by the end of
        # this step, so that the last step lands exactly on the full parent size.
        target_quantity = (
            self.parent_order_size * self.step_index / self.total_steps
        )

        # Trade if we are behind schedule and still have shares left to send.
        behind_schedule = self.quantity_sent < target_quantity
        has_remaining = self.quantity_sent < self.parent_order_size

        if behind_schedule and has_remaining:
            self.quantity_sent += self.order_fixed_size
            return self.trade_action

        return self.ACTION_HOLD
