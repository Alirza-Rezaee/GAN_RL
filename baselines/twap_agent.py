class TWAPAgent:
    """Time-Weighted Average Price execution strategy."""

    def __init__(self, action=0):
        # In markets-execution-v0, action 0 is a market order.
        self.action = action

    def predict(self, observation):
        """Return the fixed market-order action for every step."""
        return self.action
