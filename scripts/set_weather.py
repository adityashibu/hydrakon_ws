import carla
import time

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()

# Start with clear weather
weather = world.get_weather()
weather.precipitation = 100.0
weather.precipitation_deposits = 100.0
world.set_weather(weather)
print("Starting with clear weather")
time.sleep(2)

# Gradually increase to heavy rain
for i in range(0, 100, 10):
    weather.precipitation = i
    weather.precipitation_deposits = i
    world.set_weather(weather)
    print(f"Rain intensity: {i}%")
    time.sleep(1)

print("Heavy rain should now be visible")
