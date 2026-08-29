from abides_core import abides
from abides_markets.configs import rmsc04

# استفاده از مقادیر پیش‌فرض کامل بازار
config_state = rmsc04.build_config(seed=1, end_time="16:00:00")

end_state = abides.run(config_state)
print("Simulation completed successfully!")