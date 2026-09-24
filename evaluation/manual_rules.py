"""
manual_rules.py

Pure decision-logic for the hand-written execution rules used in the
Day-3 "headroom screening" test (gate_b vs gate_d, no RL involved).

This file has ZERO dependency on ABIDES, gym, or your project's other
modules on purpose — it only knows about numbers (step, remaining shares,
spread). That means you can sanity-check it right now, on your own
machine, without running a single simulated market day. Just run:

    python manual_rules.py

and read the printed action sequences.

------------------------------------------------------------------------
BEFORE WIRING THIS INTO THE REAL EVALUATOR — CONFIRM, DON'T ASSUME
------------------------------------------------------------------------
The three constants right below are PLACEHOLDERS. The ABIDES-Gym paper
confirms the execution environment has exactly three discrete actions
("MARKET ORDER", "LIMIT ORDER", "DO NOTHING"), but it does not fix which
integer maps to which action in the actual gym action_space, and your
copy of abides/ is a patched fork, so the real mapping could differ from
any public example you find online.

Ask Claude Code to open the file inside abides_gym that implements
markets-execution-v0 and show you, verbatim, how action_space is defined
and how each integer is translated into an order. Then fix the three
lines below to match. Do not trust these numbers otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

# ---------------------------------------------------------------------
# TODO — CONFIRM AGAINST SOURCE BEFORE RUNNING ON THE REAL ENVIRONMENT.
# ---------------------------------------------------------------------
ACTION_MARKET = 0   # "MARKET ORDER"  — TODO confirm from source
ACTION_LIMIT = 1    # "LIMIT ORDER"   — TODO confirm from source
ACTION_HOLD = 2     # "DO NOTHING"    — TODO confirm from source


@dataclass
class RuleState:
    """
    Everything a rule needs in order to decide, for ONE step of ONE
    episode. The evaluator (evaluate_manual_rules_gates.py) is
    responsible for filling this in correctly every step, using the
    real observation/info coming back from env.step().
    """
    step: int                 # 0-indexed step number within the episode
    total_steps: int          # total number of decision steps this episode has
    remaining_shares: int     # shares still left to execute (>= 0)
    order_fixed_size: int     # size of ONE child order (fixed, can't change)
    spread: float             # best_ask - best_bid at this step
    spread_history: List[float] = field(default_factory=list)
    # spread_history = every spread OBSERVED BEFORE this step, oldest first.
    # Does NOT include the current step's spread.


class ManualRule:
    """
    Base class. Every rule inherits the SAME safety net automatically, so
    no individual rule can forget it and accidentally leave shares unsold.

    The safety net is capacity-aware, not "always force market in the
    last N steps": it checks, every single step, whether there is still
    enough room left to wait. The moment there is exactly one child-order
    worth of slack left (or less), it forces a market order — regardless
    of what the rule's own logic wants. On a gate with ZERO slack from
    the start (e.g. parent_order_size == order_fixed_size * total_steps,
    like gate_b), this means every rule collapses to "trade every step",
    identical to TWAP, starting from step 0. That is expected and
    correct, not a bug — a rule can only show its own judgement where the
    environment actually gives it room to.
    """

    name = "base_rule"

    def reset(self) -> None:
        """Called once at the start of every episode. Override if a rule
        needs to reset its own memory (none of the four below do)."""
        pass

    def _decide(self, state: RuleState) -> int:
        """Subclasses implement ONLY their own idea of what to do. They
        never need to worry about the safety net themselves — predict()
        below takes care of that before ever calling this."""
        raise NotImplementedError

    def predict(self, state: RuleState) -> int:
        if state.remaining_shares <= 0:
            # Nothing left to execute — always HOLD, no matter the rule.
            return ACTION_HOLD

        steps_left = state.total_steps - state.step
        # ceil division: how many more child orders are needed to finish
        trades_still_needed = -(-state.remaining_shares // state.order_fixed_size)

        if trades_still_needed >= steps_left:
            # No slack left at all: if we don't trade THIS step, the
            # parent order can no longer possibly finish inside the
            # execution window. Force it, overriding the rule's own logic.
            return ACTION_MARKET

        return self._decide(state)


class FrontLoaded(ManualRule):
    """Trade in the first `trade_steps` steps, then hold (subject to the
    safety net taking back over near the end if needed)."""
    name = "front_loaded"

    def __init__(self, trade_steps: int = 5):
        self.trade_steps = trade_steps

    def _decide(self, state: RuleState) -> int:
        if state.step < self.trade_steps:
            return ACTION_MARKET
        return ACTION_HOLD


class BackLoaded(ManualRule):
    """Hold for the first `hold_steps` steps, then trade."""
    name = "back_loaded"

    def __init__(self, hold_steps: int = 5):
        self.hold_steps = hold_steps

    def _decide(self, state: RuleState) -> int:
        if state.step < self.hold_steps:
            return ACTION_HOLD
        return ACTION_MARKET


class SpreadAware(ManualRule):
    """
    Trade only when the current spread is at or below the median of
    every spread observed so far THIS episode; otherwise wait for a
    better moment. On step 0 there is no history yet to compare against,
    so it defaults to trading (this only ever affects the very first
    step of every episode).
    """
    name = "spread_aware"

    def _decide(self, state: RuleState) -> int:
        if not state.spread_history:
            return ACTION_MARKET

        s = sorted(state.spread_history)
        mid = len(s) // 2
        median = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2

        return ACTION_MARKET if state.spread <= median else ACTION_HOLD


class PatientThenPanic(ManualRule):
    """Rest a limit order for the first `patient_steps` steps (hoping to
    get filled at a better price without crossing the spread), then
    switch to market orders for the rest."""
    name = "patient_then_panic"

    def __init__(self, patient_steps: int = 6):
        self.patient_steps = patient_steps

    def _decide(self, state: RuleState) -> int:
        if state.step < self.patient_steps:
            return ACTION_LIMIT
        return ACTION_MARKET


ALL_RULES: Dict[str, Type[ManualRule]] = {
    "front_loaded": FrontLoaded,
    "back_loaded": BackLoaded,
    "spread_aware": SpreadAware,
    "patient_then_panic": PatientThenPanic,
}


def make_rule(name: str) -> ManualRule:
    if name not in ALL_RULES:
        raise ValueError(f"Unknown rule '{name}'. Choices: {list(ALL_RULES)}")
    return ALL_RULES[name]()


# ---------------------------------------------------------------------
# Self-test: runs with NO dependency on ABIDES at all. This lets you
# eyeball what each rule actually does, on both gate shapes, before
# spending a single second of real simulation time.
#
#   python manual_rules.py
# ---------------------------------------------------------------------
def _self_test() -> None:
    action_names = {ACTION_MARKET: "MKT", ACTION_LIMIT: "LMT", ACTION_HOLD: "HLD"}

    # (label, parent_order_size, order_fixed_size, total_steps)
    scenarios = [
        ("gate_b  (parent=1000, fixed=100, steps=10 -> ZERO slack)", 1000, 100, 10),
        ("gate_d  (parent=500,  fixed=100, steps=10 -> some slack)", 500, 100, 10),
    ]

    # A fake, wide-swinging spread pattern just so spread_aware has
    # something non-trivial to react to in this mock run. Real spreads
    # will come from the actual simulated order book, not this list.
    fake_spreads = [3.0, 1.0, 4.0, 1.0, 2.0, 5.0, 1.0, 3.0, 1.0, 2.0]

    for label, parent, fixed, steps in scenarios:
        print(f"\n{label}")
        for rule_name in ALL_RULES:
            rule = make_rule(rule_name)
            rule.reset()
            remaining = parent
            spread_hist: List[float] = []
            actions: List[str] = []

            for step in range(steps):
                spread_now = fake_spreads[step % len(fake_spreads)]
                state = RuleState(
                    step=step,
                    total_steps=steps,
                    remaining_shares=remaining,
                    order_fixed_size=fixed,
                    spread=spread_now,
                    spread_history=list(spread_hist),
                )
                a = rule.predict(state)
                actions.append(action_names[a])
                if a == ACTION_MARKET:
                    remaining = max(0, remaining - fixed)
                spread_hist.append(spread_now)

            status = "OK" if remaining == 0 else f"UNEXECUTED={remaining}"
            print(f"  {rule_name:20s}: {' '.join(actions)}   [{status}]")


if __name__ == "__main__":
    _self_test()