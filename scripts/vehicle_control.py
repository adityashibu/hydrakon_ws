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
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.throttle = 0.5
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.brake = 0.5
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.steer = -0.5
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.steer = 0.5
        if keys[pygame.K_r]:  # toggle reverse
            self.reverse = not self.reverse

try:
    pygame.init()
    display_size = (1280, 720)
    display = pygame.display.set_mode(
        (display_size[0], display_size[1]),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("ZED 2i Depth Visualization")

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    # ==== Use existing vehicle ====
    vehicles = world.get_actors().filter('vehicle.*')
    if not vehicles:
        raise RuntimeError("No existing vehicles found in the world.")
    
    vehicle = vehicles[0]
    print(f"Using existing vehicle: {vehicle.type_id}, id: {vehicle.id}")
    # ==============================

    blueprint_library = world.get_blueprint_library()

    # Depth camera setup
    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', '1280')
    depth_bp.set_attribute('image_size_y', '720')
    depth_bp.set_attribute('fov', '110')

    camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
    depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)

    controller = VehicleController()

    class SensorData:
        def __init__(self):
            self.depth_img = None

    sensor_data = SensorData()

    def process_depth(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        controller.update(keys)

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

            font = pygame.font.Font(None, 36)
            velocity = vehicle.get_velocity()
            speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            text = font.render(f'Speed: {speed:.1f} m/s', True, (255, 255, 255))
            display.blit(text, (10, 10))

            pygame.display.flip()

        clock.tick(20)

finally:
    pygame.quit()
    if 'depth_camera' in locals():
        depth_camera.destroy()
