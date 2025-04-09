import pygame
import math

# Initialize pygame
pygame.init()

# Screen size and setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TurtleBot 2D Simulator")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# TurtleBot settings
robot_radius = 15
robot_x, robot_y = WIDTH // 2, HEIGHT // 2
robot_angle = 0  # In degrees (0 degrees points to the right)
speed = 2
turn_speed = 5

# Clock for framerate
clock = pygame.time.Clock()

def draw_robot(x, y, angle):
    """ Draw the TurtleBot at position (x, y) with angle. """
    # Convert angle to radians
    angle_rad = math.radians(angle)
    # Robot's front direction (for simulation of movement)
    line_length = robot_radius * 10
    line_end_x = x + math.cos(angle_rad) * line_length
    line_end_y = y - math.sin(angle_rad) * line_length

    pygame.draw.circle(screen, RED, (int(x), int(y)), robot_radius)
    pygame.draw.line(screen, BLACK, (x, y), (line_end_x, line_end_y), 3)

def move_robot(x, y, angle, forward=True):
    """ Move the robot in the direction it is facing. """
    angle_rad = math.radians(angle)
    if forward:
        x += speed * math.cos(angle_rad)
        y -= speed * math.sin(angle_rad)  # Subtract because screen y-axis is inverted
    else:
        x -= speed * math.cos(angle_rad)
        y += speed * math.sin(angle_rad)
    return x, y

def turn_robot(angle, clockwise=True):
    """ Turn the robot left or right. """
    if clockwise:
        angle += turn_speed
    else:
        angle -= turn_speed
    return angle % 360  # Keep the angle within 0-360 degrees

def main():
    global robot_x, robot_y, robot_angle

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:  # Move forward
            robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, forward=True)
        if keys[pygame.K_s]:  # Move backward
            robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, forward=False)
        if keys[pygame.K_d]:  # Turn left (counter-clockwise)
            robot_angle = turn_robot(robot_angle, clockwise=False)
        if keys[pygame.K_a]:  # Turn right (clockwise)
            robot_angle = turn_robot(robot_angle, clockwise=True)

        # Draw the robot
        draw_robot(robot_x, robot_y, robot_angle)

        # Update display
        pygame.display.flip()

        # Maintain 60 FPS
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
