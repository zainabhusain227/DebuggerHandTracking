import cv2
import mediapipe as mp
import pygame
import random
import math
import numpy as np

pygame.init()
pygame.mixer.init()

cap = cv2.VideoCapture(0)
ret, sample_frame = cap.read()
cv2.flip(sample_frame, 1, sample_frame)

height, width, _ = sample_frame.shape
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Hand Tracking and Fireworks")

# Load assets
bug_image = pygame.image.load("whitebug.png").convert_alpha()
error_image = pygame.image.load("error_message.png").convert_alpha()
explosion_sound = pygame.mixer.Sound("fireworks_sound.wav")
game_over_sound = pygame.mixer.Sound("windows-error-sound.wav")

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

fireworks = []

font = pygame.font.Font(None, 36)
game_over_font = pygame.font.Font(None, 72)
score_font = pygame.font.Font(None, 48)
bit_font = pygame.font.Font(None, 18)  # Smaller font for 1s and 0s

last_firework_time = pygame.time.get_ticks()
firework_interval = 1000
missed_fireworks = 0
score = 0

def create_firework():
    return {
        'color': (random.randint(0, 255), 255, random.randint(0, 255)),
        'radius': random.randint(10, 30),
        'speed': random.randint(5, 10),
        'burst_particles': [],
        'exploded': False,
        'x': width - random.randint(0, width),
        'y': height,
        'rect': pygame.Rect(0, 0, 0, 0),
    }

def create_particle(x, y):
    return {
        'x': x,
        'y': y,
        'text': random.choice(['0', '1']),
        'color': (0, random.randint(180, 255), 0),
        'speed': random.uniform(1, 5),
        'angle': random.uniform(0, 2 * math.pi),
    }

running = True
game_over = False
played_game_over_sound = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 128))

    if not game_over:
        current_time = pygame.time.get_ticks()
        if current_time - last_firework_time >= firework_interval:
            fireworks.append(create_firework())
            last_firework_time = current_time
            firework_interval = max(50, firework_interval - 5)

    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmark_points = []
            for id, lm in enumerate(handLms.landmark):
                cx = width - int(lm.x * width)
                cy = int(lm.y * height)
                landmark_points.append((cx, cy))

                if id == 8:
                    pygame.draw.circle(screen, (255, 255, 255, 128), (cx, cy), 10)

                    for firework in fireworks:
                        if not firework['exploded']:
                            firework['rect'] = pygame.Rect(firework['x'] - firework['radius'],
                                                           firework['y'] - firework['radius'],
                                                           2 * firework['radius'], 2 * firework['radius'])
                            if firework['rect'].colliderect((cx - 25, cy - 25, 50, 50)):
                                firework['burst_particles'] = [create_particle(firework['x'], firework['y']) for _ in range(50)]
                                firework['exploded'] = True
                                score += 1
                                explosion_sound.play()

            hand_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            landmark_color = (255, 255, 255, 128)
            line_color = (255, 255, 255, 100)

            for pt in landmark_points:
                pygame.draw.circle(hand_surface, landmark_color, pt, 5)

            for connection in mpHands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    pygame.draw.line(hand_surface, line_color,
                                     landmark_points[start_idx],
                                     landmark_points[end_idx], 2)

            screen.blit(hand_surface, (0, 0))

    for firework in fireworks.copy():
        if not firework['exploded']:
            scaled_bug = pygame.transform.scale(bug_image, (2 * firework['radius'], 2 * firework['radius']))
            screen.blit(scaled_bug, (firework['x'] - firework['radius'], firework['y'] - firework['radius']))
            firework['y'] -= firework['speed']

            if firework['y'] <= 0:
                missed_fireworks += 1
                fireworks.remove(firework)
                if missed_fireworks >= 3:
                    game_over = True

        for particle in firework['burst_particles']:
            text_surface = bit_font.render(particle['text'], True, particle['color'])
            screen.blit(text_surface, (int(particle['x']), int(particle['y'])))
            particle['x'] += particle['speed'] * math.cos(particle['angle'])
            particle['y'] += firework['speed'] * math.sin(particle['angle'])

        firework['burst_particles'] = [
            p for p in firework['burst_particles']
            if 0 <= p['x'] <= width and 0 <= p['y'] <= height
        ]

        if firework['exploded'] and all(p['y'] < 0 for p in firework['burst_particles']):
            fireworks.remove(firework)

    text_missed = font.render(f"Missed Bugs: {missed_fireworks}", True, (255, 255, 255))
    screen.blit(text_missed, (10, 10))

    text_score = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(text_score, (10, 50))

    if game_over:
        if not played_game_over_sound:
            game_over_sound.play()
            played_game_over_sound = True

        error_rect = error_image.get_rect(center=(width // 2, height // 2))
        screen.blit(error_image, error_rect)

        score_text = score_font.render(f"Score: {score}", True, (0, 0, 128))
        screen.blit(score_text, (width // 2 - score_text.get_width() // 2,
                                 error_rect.bottom + 10))

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()
