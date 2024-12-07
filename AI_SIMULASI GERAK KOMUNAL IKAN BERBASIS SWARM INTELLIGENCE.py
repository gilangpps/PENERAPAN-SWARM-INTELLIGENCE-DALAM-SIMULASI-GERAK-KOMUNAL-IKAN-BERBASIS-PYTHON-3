import pygame
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from moviepy.editor import ImageSequenceClip

# Parameter simulasi
WIDTH, HEIGHT = 800, 600
DEFAULT_AGENT_COUNT = 50
DEFAULT_MAX_SPEED = 2
DEFAULT_PERCEPTION_RADIUS = 50
TARGET_RADIUS = 10
FPS = 30  # Frame per second untuk simulasi dan video

pygame.init()  # Inisialisasi pygame


class Agent:
    def __init__(self):
        self.position = np.random.rand(2) * [WIDTH, HEIGHT]
        self.velocity = (np.random.rand(2) - 0.5) * DEFAULT_MAX_SPEED

    def update(self, agents, targets):
        acceleration = self.calculate_behavior(agents)
        if targets:
            target_force = self.seek_nearest_target(targets) * 0.05
        else:
            target_force = np.zeros(2)
        self.velocity += acceleration + target_force
        speed = np.linalg.norm(self.velocity)
        if speed > DEFAULT_MAX_SPEED:
            self.velocity = (self.velocity / speed) * DEFAULT_MAX_SPEED
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
            if distance < DEFAULT_PERCEPTION_RADIUS:
                cohesion += other.position
                separation += (self.position - other.position) / (distance**2)
                alignment += other.velocity
                total += 1

        if total > 0:
            cohesion = (cohesion / total - self.position) * 0.01
            separation *= 0.05
            alignment = (alignment / total - self.velocity) * 0.05

        return cohesion + separation + alignment

    def seek_nearest_target(self, targets):
        nearest_target = min(targets, key=lambda t: np.linalg.norm(self.position - t))
        direction = nearest_target - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            return direction / distance
        return np.zeros(2)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), self.position.astype(int), 3)


def update_agents(agents, targets):
    for agent in agents:
        agent.update(agents, targets)


def draw_simulation(screen, agents, targets):
    screen.fill((0, 0, 0))
    for target in targets:
        pygame.draw.circle(screen, (255, 0, 0), target.astype(int), TARGET_RADIUS)
    for agent in agents:
        agent.draw(screen)
    pygame.display.flip()


def run_simulation(root, canvas, ax, add_target_callback):
    global DEFAULT_AGENT_COUNT, DEFAULT_MAX_SPEED, DEFAULT_PERCEPTION_RADIUS

    # Pygame setup
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Swarm Simulation")
    agents = [Agent() for _ in range(DEFAULT_AGENT_COUNT)]
    targets = [np.random.rand(2) * [WIDTH, HEIGHT]]
    frames = []  # List untuk menyimpan frame animasi

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Klik kiri
                    mouse_pos = np.array(event.pos)
                    targets.append(mouse_pos)

        # Update agents
        update_agents(agents, targets)

        # Draw agents and targets
        draw_simulation(screen, agents, targets)

        # Rekam frame
        frame = pygame.surfarray.array3d(screen)
        frames.append(np.rot90(frame, k=3))  # Rotasi agar orientasi video benar

        # Update Matplotlib graph
        distances = [np.linalg.norm(agent.position - t) for t in targets for agent in agents]
        ax.clear()
        ax.hist(distances, bins=10, color='blue', alpha=0.7)
        ax.set_title("Distribusi Jarak Agen ke Target")
        ax.set_xlabel("Jarak")
        ax.set_ylabel("Jumlah Agen")
        canvas.draw()

        root.update()
        clock.tick(FPS)

    # Simpan animasi sebagai video
    save_video(frames)


def save_video(frames):
    print("Menyimpan video...")
    clip = ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile("swarm_simulation.mp4", codec="libx264")
    print("Video berhasil disimpan sebagai 'swarm_simulation.mp4'.")


# GUI Setup
root = tk.Tk()
root.title("Swarm Simulation")

# Frame untuk kontrol
frame = tk.Frame(root)
frame.pack(side=tk.LEFT, padx=10, pady=10)

ttk.Label(frame, text="Jumlah Agen").pack()
agent_count_slider = ttk.Scale(frame, from_=10, to=200, orient="horizontal")
agent_count_slider.set(DEFAULT_AGENT_COUNT)
agent_count_slider.pack()

ttk.Label(frame, text="Kecepatan Maksimum").pack()
max_speed_slider = ttk.Scale(frame, from_=0.5, to=5, orient="horizontal")
max_speed_slider.set(DEFAULT_MAX_SPEED)
max_speed_slider.pack()

ttk.Label(frame, text="Radius Persepsi").pack()
perception_radius_slider = ttk.Scale(frame, from_=10, to=200, orient="horizontal")
perception_radius_slider.set(DEFAULT_PERCEPTION_RADIUS)
perception_radius_slider.pack()

# Grafik Analitik
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.RIGHT, padx=10, pady=10)

# Callback untuk menambahkan target
def add_target():
    new_target = np.random.rand(2) * [WIDTH, HEIGHT]
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=new_target.astype(int)))

ttk.Button(frame, text="Tambah Target", command=add_target).pack()

# Jalankan simulasi
run_simulation(root, canvas, ax, add_target)

root.mainloop()
