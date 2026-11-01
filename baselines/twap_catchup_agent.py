"""
TWAP that schedules on shares FILLED instead of shares SENT.

Why this agent exists
---------------------
baselines/twap_agent.py tracks quantity_sent and ignores the observation. On
gates whose child orders always fill that is a faithful TWAP. On gate_e it is
not a usable baseline, and the per-child fill diagnostic (commit 45496c1)
measured exactly why:

  * all 24 incomplete episodes out of 100 contained at least one partially
    filled child, and all 76 completed episodes contained none - the separation
    is total;
  * not one of those 24 episodes recovered, because market-order remainders are
    discarded rather than rested (recorded in 80c5590), and an agent counting
    shares SENT never learns that a share went missing;
  * the agent held on 4 of the 10 decision steps in every one of those episodes
    and spent none of them resending.

The book is not the problem: median bid depth over levels 1-5 ahead of children
1 to 5 is flat at 618, 604, 700, 674, 592, and the later children fill BETTER,
with partial-fill rates of 15%, 8%, 8%, 5% and 4%.

What this changes, and what it does not
---------------------------------------
This is NOT a cleverer baseline. On gate_e the committed TWAP is the only agent
in the project that is not fill-aware: the four manual rules in
evaluation/manual_rules.py have read remaining_shares and carried a forced
catch-up since they were written. Comparing a fill-aware policy against a
baseline that cannot recover a partial fill measures the asymmetry, not the
policy. This agent removes the asymmetry.

It deliberately adds nothing else. It reacts to one number - how many shares are
still outstanding - and to nothing about prices, spreads or depth, so it remains
a fixed time schedule that a learning policy has to beat on judgement rather
than on bookkeeping.

The equivalence property
------------------------
When every child order fills completely, this agent sends on exactly the same
steps as baselines/twap_agent.py, because outstanding shares and sent shares are
then the same quantity counted from opposite ends. _self_test() below checks
that on gate_e, gate_d_varchild and gate_b without starting a simulator, and
configs/twap_catchup_criteria.yaml's Run A checks it against the simulator on
200 real episodes.

baselines/twap_agent.py is left untouched, so every committed TWAP number stays
reproducible from its own source.
"""

import math

from abides_core.utils import str_to_ns


class TWAPCatchupAgent:
    """
    Deterministic schedule on outstanding shares: by the end of step k, at most
    parent_order_size * (1 - k / total_steps) should still be outstanding. The
    agent sends one child order whenever more than that is outstanding, and
    holds otherwise.

    On top of that it carries the same capacity-aware safety net as
    evaluation/manual_rules.py: if the number of child orders still needed is at
    least the number of steps still available, it sends regardless of the
    schedule. Without it a run of partial fills late in the window could leave
    the agent needing more orders than it has steps left.

    This agent does not learn anything. Its purpose is to give the RL agent a
    reference number it has to beat for a real reason.
    """

    # Action ids of the markets-execution-v0 environment, same as TWAPAgent.
    ACTION_MARKET_ORDER = 0
    ACTION_LIMIT_ORDER = 1
    ACTION_HOLD = 2

    # Read by the evaluation harness to decide what to pass to predict(). The
    # committed TWAPAgent takes an observation it ignores; this one needs the
    # outstanding quantity, which only the environment knows.
    NEEDS_REMAINING_SHARES = True

    def __init__(self, config, use_limit_orders=False):
        """
        Arguments:
            config:           the loaded project config (see common.env_factory).
            use_limit_orders: if False (default) each child order is a market
                              order, which pays the spread but does not rest.
                              Kept for signature parity with TWAPAgent; the
                              pre-registered runs use market orders, because a
                              resting limit order is how the Stage-1 cancel race
                              happened.
        """
        environment_config = config["environment"]

        self.parent_order_size = environment_config["parent_order_size"]
        self.order_fixed_size = environment_config["order_fixed_size"]

        # How many decision steps fit inside the execution window. Computed the
        # same way as in TWAPAgent, so the two agents cannot disagree about the
        # length of the window they are scheduling over.
        execution_window_ns = str_to_ns(environment_config["execution_window"])
        timestep_ns = str_to_ns(environment_config["timestep_duration"])
        self.total_steps = int(execution_window_ns / timestep_ns)

        self.trade_action = (
            self.ACTION_LIMIT_ORDER
            if use_limit_orders
            else self.ACTION_MARKET_ORDER
        )

        self.reset()

    def reset(self):
        """Clear the per-episode counter. Must be called before every episode."""
        self.step_index = 0

    def predict(self, remaining_shares):
        """
        Return the action for the current step.

        Arguments:
            remaining_shares: shares still outstanding, read from the
                environment before this step's order goes out. In the evaluation
                harness this is env.unwrapped.shares_remaining(), which is
                derived from the agent's holdings - the same quantity the
                environment computes done and reward from, and the only source
                that is fresh at step 1 of an episode. See
                common/execution_env.py's module docstring for why the metrics
                tracker cannot be used here.

        Unlike TWAPAgent.predict this takes a quantity rather than an
        observation. It is still blind to prices, spreads and depth.
        """
        self.step_index += 1

        if remaining_shares <= 0:
            # Nothing left to execute. Holding rather than sending matters: it is
            # what keeps the environment's clamp from having to suppress a
            # zero-size order, and it is what makes an overshoot impossible from
            # the agent's side.
            return self.ACTION_HOLD

        # Steps still available, counting this one. Matches the convention in
        # evaluation/manual_rules.py, where step is 0-indexed and
        # steps_left = total_steps - step.
        steps_left = self.total_steps - self.step_index + 1

        # Safety net, checked before the schedule so no schedule decision can
        # override it: ceil division, how many more child orders are needed.
        orders_still_needed = math.ceil(
            remaining_shares / self.order_fixed_size
        )
        if orders_still_needed >= steps_left:
            return self.trade_action

        # The time schedule, expressed on outstanding shares. At the end of step
        # k, a TWAP that started with parent_order_size should have
        # parent_order_size * (1 - k / total_steps) still to go, i.e.
        #
        #     remaining > parent_order_size * (total_steps - k) / total_steps
        #
        # Written multiplied out so that it is exact integer arithmetic. The
        # direct float form is NOT safe: at gate_e's step 8,
        # 5000 * (1 - 8 / 10) evaluates to 999.9999999999998 rather than 1000,
        # so an outstanding balance of exactly 1000 compares as behind schedule
        # and the agent sends a step early. That broke the equivalence claim on
        # gate_e and gate_d_varchild, and _self_test() caught it. TWAPAgent
        # avoids the same trap only incidentally, by multiplying before dividing.
        behind_schedule = (
            remaining_shares * self.total_steps
            > self.parent_order_size * (self.total_steps - self.step_index)
        )

        if behind_schedule:
            return self.trade_action

        return self.ACTION_HOLD


def _self_test():
    """
    Check the equivalence claim without starting a simulator.

    Two things are verified on each gate shape:

      1. With every child filling completely, this agent sends on exactly the
         same steps as baselines/twap_agent.py. This is the claim
         configs/twap_catchup_criteria.yaml stakes Run A on.
      2. With a scripted partial fill, this agent finishes the parent order and
         the committed agent does not - which is the whole point.

    Run with:  python -m baselines.twap_catchup_agent
    """
    from baselines.twap_agent import TWAPAgent

    action_names = {0: "MKT", 1: "LMT", 2: "---"}

    # (label, parent_order_size, order_fixed_size, total_steps)
    shapes = [
        ("gate_e           ", 5000, 1000, 10),
        ("gate_e candidate B", 3000, 1000, 10),
        ("gate_d_varchild  ", 500, 100, 10),
        ("gate_b (zero slack)", 1000, 100, 10),
    ]

    def config_for(parent, fixed, steps):
        # total_steps is derived as execution_window / timestep_duration, so the
        # window is written to produce the requested number of steps.
        return {
            "environment": {
                "parent_order_size": parent,
                "order_fixed_size": fixed,
                "execution_window": f"00:{steps:02d}:00",
                "timestep_duration": "60s",
            }
        }

    def run(agent, steps, fills):
        """Drive an agent through one scripted episode.

        fills[j] is how many shares the j-th order sent actually fills. A value
        shorter than the number of orders sent means "every later order fills
        completely".
        """
        agent.reset()
        remaining = agent.parent_order_size
        orders_sent = 0
        actions = []

        for _ in range(steps):
            if getattr(agent, "NEEDS_REMAINING_SHARES", False):
                action = agent.predict(remaining)
            else:
                action = agent.predict(None)
            actions.append(action_names[action])

            if action != agent.ACTION_HOLD:
                # The environment clamps each child to what is outstanding, see
                # common/execution_env.py.
                requested = min(agent.order_fixed_size, remaining)
                if orders_sent < len(fills):
                    filled = min(fills[orders_sent], requested)
                else:
                    filled = requested
                remaining -= filled
                orders_sent += 1

        return actions, remaining

    print("1. Equivalence when every child fills completely")
    print("   (committed TWAP vs catch-up TWAP, must be identical)")
    failures = 0
    for label, parent, fixed, steps in shapes:
        config = config_for(parent, fixed, steps)
        old_actions, old_left = run(TWAPAgent(config), steps, [])
        new_actions, new_left = run(TWAPCatchupAgent(config), steps, [])
        same = old_actions == new_actions and old_left == new_left
        failures += 0 if same else 1
        print(f"   {label}  {'OK   ' if same else 'DIFFER'}")
        print(f"      committed: {' '.join(old_actions)}  left={old_left}")
        print(f"      catch-up : {' '.join(new_actions)}  left={new_left}")

    print()
    print("2. Recovery when the first child fills only half")
    print("   (committed TWAP must end short, catch-up must finish)")
    for label, parent, fixed, steps in shapes:
        config = config_for(parent, fixed, steps)
        half = [fixed // 2]
        old_actions, old_left = run(TWAPAgent(config), steps, half)
        new_actions, new_left = run(TWAPCatchupAgent(config), steps, half)
        print(f"   {label}")
        print(f"      committed: {' '.join(old_actions)}  left={old_left}")
        print(f"      catch-up : {' '.join(new_actions)}  left={new_left}")
        if parent == fixed * steps:
            # gate_b has no slack at all, so nothing can recover there. Stated
            # rather than silently passing: a gate with zero slack cannot be
            # fixed by any schedule, only by resizing.
            print("      (zero slack by construction - neither can recover)")

    print()
    if failures:
        raise SystemExit(
            f"Equivalence check FAILED on {failures} gate shape(s). "
            "Run A in configs/twap_catchup_criteria.yaml would fail too."
        )
    print("Equivalence holds on all gate shapes tested.")


if __name__ == "__main__":
    _self_test()
