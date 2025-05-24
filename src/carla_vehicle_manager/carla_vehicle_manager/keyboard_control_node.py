import carla
import time
import numpy as np
import cv2
import pygame

class VehicleController:
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.reverse = False

    def update(self, keys):
        # Reset controls
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        
        # Update based on key presses
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.throttle = 0.5
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.brake = 0.5
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.steer = -0.5
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.steer = 0.5
        if keys[pygame.K_r]:  # R key to toggle reverse
            self.reverse = not self.reverse

try:
    pygame.init()
    display_size = (1280, 720)  # ZED 2i resolution
    display = pygame.display.set_mode(
        (display_size[0], display_size[1]),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("ZED 2i Depth Visualization")

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    # Find existing vehicle instead of spawning new one
    vehicles = world.get_actors().filter('vehicle.*')
    if not vehicles:
        print("No vehicles found in the world! Please spawn a vehicle first.")
        exit(1)
    
    # Use the first vehicle found
    vehicle = vehicles[0]
    print(f"Using existing vehicle: {vehicle.type_id} (ID: {vehicle.id})")
    print(f"Vehicle location: {vehicle.get_location()}")

    # Set up ZED 2i-like depth camera
    blueprint_library = world.get_blueprint_library()
    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', '1280')
    depth_bp.set_attribute('image_size_y', '720')
    depth_bp.set_attribute('fov', '110')  # ZED 2i wide FOV

    camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
    depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)

    # Initialize vehicle controller
    controller = VehicleController()

    # Sensor data handler
    class SensorData:
        def __init__(self):
            self.depth_img = None

    sensor_data = SensorData()

    def process_depth(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        normalized = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return depth_colormap

    def depth_callback(image):
        sensor_data.depth_img = process_depth(image)

    depth_camera.listen(depth_callback)

    running = True
    clock = pygame.time.Clock()

    # Display controls
    print("Controls:")
    print("W/Up Arrow - Accelerate")
    print("S/Down Arrow - Brake")
    print("A/Left Arrow - Steer Left")
    print("D/Right Arrow - Steer Right")
    print("R - Toggle Reverse")
    print("ESC - Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Check if vehicle is still alive
        if not vehicle.is_alive:
            print("Vehicle is no longer alive! Exiting...")
            running = False
            continue

        # Update vehicle controls
        keys = pygame.key.get_pressed()
        controller.update(keys)
        
        # Apply control to vehicle
        control = carla.VehicleControl()
        control.throttle = controller.throttle
        control.steer = controller.steer
        control.brake = controller.brake
        control.reverse = controller.reverse
        vehicle.apply_control(control)

        if sensor_data.depth_img is not None:
            depth_surface = pygame.surfarray.make_surface(
                np.transpose(sensor_data.depth_img, (1, 0, 2)))
            display.blit(depth_surface, (0, 0))
            
            # Add debug info
            font = pygame.font.Font(None, 36)
            velocity = vehicle.get_velocity()
            speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            text = font.render(f'Speed: {speed:.1f} m/s', True, (255, 255, 255))
            display.blit(text, (10, 10))
            
            # Add vehicle info
            location = vehicle.get_location()
            info_text = font.render(f'Pos: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})', True, (255, 255, 255))
            display.blit(info_text, (10, 50))
            
            pygame.display.flip()

        clock.tick(20)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    pygame.quit()
    if 'depth_camera' in locals():
        try:
            depth_camera.destroy()
            print("Depth camera destroyed")
        except:
            pass
    # Don't destroy the vehicle since we didn't create it
    print("Controller shutdown complete")