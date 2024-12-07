import pygame
import numpy as np

# Parameter simulasi
WIDTH, HEIGHT = 800, 600
AGENT_COUNT = 50
MAX_SPEED = 2
PERCEPTION_RADIUS = 50

# Inisialisasi pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Swarm Intelligence Simulation")
clock = pygame.time.Clock()

# Kelas Agent
class Agent:
    def __init__(self):
        self.position = np.random.rand(2) * [WIDTH, HEIGHT]
        self.velocity = (np.random.rand(2) - 0.5) * MAX_SPEED

    def update(self, agents):
        acceleration = self.calculate_behavior(agents)
        self.velocity += acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = (self.velocity / speed) * MAX_SPEED
        self.position += self.velocity
        self.position %= [WIDTH, HEIGHT]

    def calculate_behavior(self, agents):
        cohesion = np.zeros(2)
        separation = np.zeros(2)
        alignment = np.zeros(2)
        total = 0

        for other in agents:
            if other == self:
                continue
            distance = np.linalg.norm(other.position - self.position)
            if distance < PERCEPTION_RADIUS:
                cohesion += other.position
                separation += (self.position - other.position) / (distance**2)
                alignment += other.velocity
                total += 1

        if total > 0:
            cohesion = (cohesion / total - self.position) * 0.01
            separation *= 0.05
            alignment = (alignment / total - self.velocity) * 0.05

        return cohesion + separation + alignment

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), self.position.astype(int), 3)

# Inisialisasi agen
agents = [Agent() for _ in range(AGENT_COUNT)]

# Loop utama
running = True
while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for agent in agents:
        agent.update(agents)
        agent.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
