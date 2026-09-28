"""
Project-local variant of the ABIDES execution environment.

Why this file exists
--------------------
markets-execution-v0 always sends a child order of exactly order_fixed_size:

    abides/abides-gym/abides_gym/envs/markets_execution_environment_v0.py:305-324
        {"type": "MKT", "direction": self.direction, "size": self.order_fixed_size}

Once a partial fill leaves fewer than order_fixed_size shares outstanding, the
next order necessarily overshoots the parent order. Stage 0 measured exactly that:
patient_then_panic oversold in 51 of 200 episodes on gate_d (2696 shares in total,
worst single episode 93 shares), and even the market-order-only rules oversold on
thin days. Every overshoot was smaller than order_fixed_size, which is the
signature of "partial fill, then a full-size order".

This file fixes it WITHOUT touching abides/: it subclasses the environment and
overrides only the action-to-orders mapping, clamping each child order to what is
still outstanding. Nothing under abides/ is modified - see ENV_CHANGES.md.

Why the remaining quantity is read from the agent, not from the tracker
-----------------------------------------------------------------------
The obvious candidate, self.custom_metrics_tracker.remaining_quantity, is STALE at
the first step of every episode and would silently produce size-0 orders:

  * it is written only inside raw_state_to_done()
    (markets_execution_environment_v0.py:596-601);
  * reset() never calls raw_state_to_done - core_environment.py:49-102 calls only
    raw_state_to_state, at line 99;
  * the tracker is built once in __init__
    (markets_execution_environment_v0.py:133-135) and never rebuilt;
  * step() calls this mapping at core_environment.py:141, BEFORE raw_state_to_done
    at line 151.

So after a completed episode it still reads 0, min(order_fixed_size, 0) == 0, and a
size-0 order is discarded with nothing but a warnings.warn (trading_agent.py:621-624
for market orders, :511-512 for limit orders, order_book.py:182-186 in the book) -
an invisible HOLD on step 1 of every episode.

self.gym_agent, by contrast, is rebuilt by core_environment.reset() on every episode
(core_environment.py:70-79), and its holdings are updated on every fill
(trading_agent.py:771-780). It is also the exact quantity raw_state_to_done and
raw_state_to_reward are computed from, so it cannot drift from the environment's own
accounting.
"""

import gym
from gym.envs.registration import register

from abides_gym.envs import SubGymMarketsExecutionEnv_v0


VARIANT_ENV_ID = "markets-execution-variable_last_child-v0"


class VariableLastChildExecutionEnv(SubGymMarketsExecutionEnv_v0):
    """
    markets-execution-v0 with exactly one change: a child order is never larger
    than the quantity still outstanding.

    Everything else - observation, reward, done condition, info dict - is inherited
    unchanged, so results stay comparable with the stock environment on any episode
    where the clamp does not bind.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Diagnostics, so that a residual oversell can be explained rather than
        # guessed at. Read from evaluation code as env.unwrapped.<name>.
        self.clamped_order_count: int = 0
        self.suppressed_order_count: int = 0

    def reset(self):
        self.clamped_order_count = 0
        self.suppressed_order_count = 0
        return super().reset()

    def shares_remaining(self) -> int:
        """
        Shares still to execute, from the only source that is fresh at step 1 of
        every episode. See the module docstring for why the tracker is not used.
        """
        holdings = self.gym_agent.get_holdings(self.gym_agent.symbol)
        executed = holdings if self.direction == "BUY" else -holdings
        return self.parent_order_size - executed

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(self, action):
        """
        Same action ids as the parent (0 = MKT, 1 = LMT at near_touch, 2 = DO
        NOTHING), but the order size is min(order_fixed_size, shares_remaining()).
        """
        if action not in (0, 1, 2):
            raise ValueError(
                f"Action {action} is not part of the actions supported by the function."
            )

        # Count what the POLICY asked for, exactly as the parent class does, so the
        # action histogram keeps describing the policy rather than this clamp.
        self.custom_metrics_tracker.action_counter[f"action_{action}"] += 1

        if action == 2:
            return []

        size = min(self.order_fixed_size, self.shares_remaining())

        if size <= 0:
            # Nothing left to execute. Still cancel: leaving a limit order resting
            # is precisely how the residual oversell race happens, because it can
            # fill after we have already decided we were finished.
            self.suppressed_order_count += 1
            return [{"type": "CCL_ALL"}]

        if size < self.order_fixed_size:
            self.clamped_order_count += 1

        if action == 0:
            return [
                {"type": "CCL_ALL"},
                {"type": "MKT", "direction": self.direction, "size": size},
            ]

        return [
            {"type": "CCL_ALL"},
            {
                "type": "LMT",
                "direction": self.direction,
                "size": size,
                "limit_price": self.near_touch,
            },
        ]


# Registered at import time, mirroring how abides_gym registers its own ids, so
# that make_env() can keep using a single gym.make() call.
try:
    register(
        id=VARIANT_ENV_ID,
        entry_point="common.execution_env:VariableLastChildExecutionEnv",
    )
except gym.error.Error:
    # Already registered - harmless if this module gets imported twice.
    pass
