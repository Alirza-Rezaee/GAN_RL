import gym
import abides_gym

print("--- Testing Daily Investor Environment ---")
try:
    # استفاده از مقادیر پیش‌فرض رشته‌ای خود کلاس
    env_investor = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
        first_interval="00:05:00",
        mkt_close="16:00:00",
        timestep_duration="60s",
    )
    env_investor.seed(0)
    state = env_investor.reset()
    print(f"Daily Investor State Shape: {state.shape}")
    
    state, reward, done, info = env_investor.step(0)
    print(f"Step 1 -> Reward: {reward} | Done: {done}")
    env_investor.close()
    print("Daily Investor Env: SUCCESS\n")
except Exception as e:
    print(f"Daily Investor Env Failed: {e}\n")


print("--- Testing Execution Environment ---")
try:
    env_exec = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
        first_interval="00:05:00",
        mkt_close="16:00:00",
        timestep_duration="60s",
    )
    env_exec.seed(0)
    state = env_exec.reset()
    print(f"Execution State Shape: {state.shape}")
    
    state, reward, done, info = env_exec.step(0)
    print(f"Step 1 -> Reward: {reward} | Done: {done}")
    env_exec.close()
    print("Execution Env: SUCCESS")
except Exception as e:
    print(f"Execution Env Failed: {e}")