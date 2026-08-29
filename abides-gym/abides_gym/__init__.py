from gym.envs.registration import register

# ایمپورت اختیاری Ray برای جلوگیری از کرش
try:
    from ray.tune.registry import register_env
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

# ثبت محیط Daily Investor در Gym
register(
    id="markets-daily_investor-v0",
    entry_point="abides_gym.envs:SubGymMarketsDailyInvestorEnv_v0",
)

# ثبت محیط اجرای سفارش (Execution) در Gym
register(
    id="markets-execution-v0",
    entry_point="abides_gym.envs:SubGymMarketsExecutionEnv_v0",
)

if HAS_RAY:
    try:
        from abides_gym.envs import (
            SubGymMarketsDailyInvestorEnv_v0,
            SubGymMarketsExecutionEnv_v0,
        )
        register_env(
            "markets-daily_investor-v0",
            lambda config: SubGymMarketsDailyInvestorEnv_v0(**config),
        )
        register_env(
            "markets-execution-v0",
            lambda config: SubGymMarketsExecutionEnv_v0(**config),
        )
    except Exception:
        pass