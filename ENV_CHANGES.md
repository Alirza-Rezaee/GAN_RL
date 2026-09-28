# ENV_CHANGES.md — changes to the execution environment, made in PROJECT code

This file records deliberate changes to the behaviour of the ABIDES gym execution
environment that are implemented **in this project's own code**, by subclassing.

`abides/` is **not** modified by anything recorded here. That folder has its own
log, `PATCHES.md`, reserved for changes made inside the vendored simulator. As of
this entry, no such patch exists from the execution-cost work.

---

## 1. `variable_last_child` — child order size clamped to the outstanding quantity

- **Date:** 2026-09-28
- **Where:** `common/execution_env.py`, class `VariableLastChildExecutionEnv`
- **Selected by:** `env_variant: "variable_last_child"` under `environment:` in a
  config, resolved in `common/env_factory.py` (`ENV_VARIANTS`)
- **Gym id registered:** `markets-execution-variable_last_child-v0`
- **abides/ touched:** no

### What changed

Exactly one thing: the size of each child order. The stock environment always
sends `order_fixed_size`
([markets_execution_environment_v0.py:305-324](abides/abides-gym/abides_gym/envs/markets_execution_environment_v0.py#L305-L324)).
The variant sends `min(order_fixed_size, shares_remaining())`, and when nothing is
outstanding it sends `[{"type": "CCL_ALL"}]` instead of a new order.

Observation, reward, done condition and the info dict are inherited untouched.

### Why

`order_fixed_size` is fixed, so after any partial fill leaves fewer than
`order_fixed_size` shares outstanding, the next order **must** overshoot the parent
order. Stage 0 measured this on `gate_d`, seeds 100–299:

| rule | oversell episodes | total oversold shares |
|---|---|---|
| patient_then_panic | 51 / 200 | 2696 |
| front_loaded | 1 / 200 | 2 |
| spread_aware | 1 / 200 | 3 |

Every overshoot was smaller than `order_fixed_size` (worst: 93 shares) — the
signature of "partial fill, then a full-size order". Limit orders partial-fill
routinely because `CCL_ALL` precedes every action, so a resting order has one step
to fill. Market orders partial-fill on thin days; their unfilled remainder is
discarded rather than rested
([order_book.py:190-192](abides/abides-markets/abides_markets/order_book.py#L190-L192)).

This matters more for training than for reporting: `too_much_reward_update = -100`
per share and `not_enough_reward_update = -1000` per share, normalised by
`parent_order_size`, so a 81-share overshoot on `gate_d` costs −16.2 reward against
per-step slippage rewards of order 1. A PPO agent would spend its budget avoiding
the completion penalty instead of optimising execution cost.

### The trap this design avoids

The obvious source for "shares remaining" is
`self.custom_metrics_tracker.remaining_quantity`. It is **stale at step 1 of every
episode**:

- written only inside `raw_state_to_done`
  ([markets_execution_environment_v0.py:596-601](abides/abides-gym/abides_gym/envs/markets_execution_environment_v0.py#L596-L601));
- `reset()` never calls `raw_state_to_done` — [core_environment.py:49-102](abides/abides-gym/abides_gym/envs/core_environment.py#L49-L102)
  calls only `raw_state_to_state`, at line 99;
- the tracker is built once in `__init__` and never rebuilt
  ([markets_execution_environment_v0.py:133-135](abides/abides-gym/abides_gym/envs/markets_execution_environment_v0.py#L133-L135));
- `step()` calls the action mapping at [core_environment.py:141](abides/abides-gym/abides_gym/envs/core_environment.py#L141),
  **before** `raw_state_to_done` at line 151.

After a completed episode it reads `0`, so `min(order_fixed_size, 0) == 0`, and a
size-0 order is dropped with only a `warnings.warn`
([trading_agent.py:621-624](abides/abides-markets/abides_markets/agents/trading_agent.py#L621-L624)) —
an invisible HOLD on the first step of every episode.

The variant instead reads `self.gym_agent.get_holdings(...)`. That agent object is
rebuilt by `reset()` ([core_environment.py:70-79](abides/abides-gym/abides_gym/envs/core_environment.py#L70-L79)),
so it starts at 0 each episode, and it is the same quantity the environment's own
reward and done logic use.

### What it does NOT fix

A residual oversell race remains. Orders and cancels are delivered at
`current_time + computation_delay + latency`
([kernel.py:562-575](abides/abides-core/abides_core/kernel.py#L562-L575)), and rmsc04
spreads agents over a 3,866,660 m line
([utils/__init__.py:168-174](abides/abides-markets/abides_markets/utils/__init__.py#L168-L174)),
i.e. up to ~12.9 ms one way. A resting limit order can fill after the remaining
quantity was computed but before `CCL_ALL` lands. The clamp cannot close that
window; returning `CCL_ALL` when nothing is outstanding narrows it.

### Backwards compatibility

A config with no `env_variant` key resolves to `"stock"` and uses
`environment_config["id"]` unchanged. `gate_a/b/c/d.yaml` and
`phase1_baseline.yaml` are untouched, so every committed Stage-0 result stays
reproducible. `env_variant` is not in `ENV_PARAM_KEYS`, so it is never forwarded to
the environment constructor.

### Diagnostics added

`env.unwrapped.clamped_order_count` and `env.unwrapped.suppressed_order_count`,
both reset per episode. Recorded per episode by
`evaluation/evaluate_manual_rules_gates.py` as `clamped_orders` and
`suppressed_orders` (both `None` on the stock environment).

### Verification

Pre-registered in `configs/oversell_fix_criteria.yaml` before either run started.
