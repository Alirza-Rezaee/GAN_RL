"""
Execution-quality metrics used to compare agents.

The gym environment gives us a `reward`, which is convenient for training but
is not a standard finance metric and cannot be compared with published work.
This file converts what the environment reports into Implementation Shortfall
expressed in basis points (bps), which is the standard cost measure for order
execution.
"""

# 1 basis point = 0.01%. Execution costs are conventionally quoted in bps
# because they are tiny fractions of the price.
BPS = 10_000


def implementation_shortfall_bps(
    total_slippage_reward,
    entry_price,
    executed_quantity,
    parent_order_size,
):
    """
    Implementation Shortfall of one episode, in basis points.

    Definition used here: the average price actually obtained, compared with the
    arrival price (the mid price at the moment the parent order arrived).
    A POSITIVE value means a COST (we did worse than the arrival price);
    a NEGATIVE value means we did better than the arrival price.

    How we get there from the environment's numbers
    -----------------------------------------------
    Each step the environment reports
        slippage_reward = signed_pnl_of_that_step / parent_order_size
    where, for a SELL, signed_pnl = sum((fill_price - arrival_price) * quantity)
    over the child orders filled during that step (and the mirror image for a
    BUY). So summing slippage_reward over the episode and multiplying back by
    parent_order_size recovers the total signed PnL against the arrival price.

    Dividing that PnL by (arrival_price * executed_quantity) turns it into a
    relative price difference, and the sign is flipped so that cost is positive.

    Note: this measures the cost of the shares we DID execute. Shares that were
    never executed are reported separately as the fill rate, so the two numbers
    together describe the full outcome.

    Arguments:
        total_slippage_reward: sum of info["slippage_reward"] over the episode.
        entry_price:           arrival price (env.unwrapped.entry_price).
        executed_quantity:     shares actually executed in the episode.
        parent_order_size:     total shares the agent was asked to execute.

    Returns:
        Implementation Shortfall in bps, or None if nothing was executed
        (the metric is undefined with no fills).
    """
    if executed_quantity <= 0:
        return None

    # Undo the environment's normalisation to get the signed PnL in price units.
    signed_pnl = total_slippage_reward * parent_order_size

    # Value of the executed shares if they had all traded at the arrival price.
    benchmark_value = entry_price * executed_quantity

    # Flip the sign: PnL above the arrival price is a gain, so a negative cost.
    return -signed_pnl / benchmark_value * BPS


def price_advantage_bps(agent_is_bps, twap_is_bps):
    """
    Price Advantage of an agent over the TWAP baseline, in basis points.

    Since Implementation Shortfall is a cost, the agent is better when its cost
    is lower. A POSITIVE Price Advantage means the agent saved that many bps
    compared with TWAP on the same market day.

    Returns None if either shortfall is undefined.
    """
    if agent_is_bps is None or twap_is_bps is None:
        return None

    return twap_is_bps - agent_is_bps
