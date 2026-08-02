from pricing_env import PricingEnvironment

env = PricingEnvironment()

obs, info = env.reset()

print("Environment initialized successfully!")
print(obs)