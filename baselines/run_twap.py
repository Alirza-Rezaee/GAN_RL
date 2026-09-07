import gym
import abides_gym

from baselines.twap_agent import TWAPAgent


env = gym.make(
    "markets-execution-v0",
    background_config="rmsc04",
    order_fixed_size=100,
    parent_order_size=1000,
    execution_window="00:10:00",
    direction="SELL",
)

env.seed(0)
observation = env.reset()

agent = TWAPAgent()
total_reward = 0.0
step = 0

while True:
    action = agent.predict(observation)

    observation, reward, done, info = env.step(action)

    total_reward += reward
    step += 1

    print(
        f"step={step}, "
        f"action={action}, "
        f"reward={reward}, "
        f"done={done}"
    )

    if done:
        break

print("\nEpisode finished")
print("steps:", step)
print("total_reward:", total_reward)
print("final_info:", info)

env.close()