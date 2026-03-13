"""
Play Parkour manually with arrow keys.
Run from project root: python run/run_play.py
"""
import sys
import yaml
from pathlib import Path

import matplotlib.cm as _cm
import pygame

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.parkour_env import ParkourEnv, Action


def load_config():
    with open(PROJECT_ROOT / "configs" / "env.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    env_config = load_config()
    env = ParkourEnv(env_config)

    pygame.init()
    cell_size = 56
    rows, cols = env.rows, env.cols
    width = cols * cell_size
    height = rows * cell_size
    header_h = 50
    screen = pygame.display.set_mode((width, height + header_h))
    pygame.display.set_caption("Parkour — стрелки: ход, R: рестарт, Q/ESC: выход")

    font = pygame.font.Font(None, 28)
    font_big = pygame.font.Font(None, 42)

    # Colors matching YlGnBu colormap used in landscape visualizations
    _cmap = _cm.get_cmap("YlGnBu")
    h_min = int(env.height_map.min())
    h_max = int(env.height_map.max())

    def height_color(h):
        t = (h - h_min) / max(h_max - h_min, 1)
        r, g, b, _ = _cmap(t)
        return (int(r * 255), int(g * 255), int(b * 255))

    key_to_action = {
        pygame.K_UP: Action.UP,
        pygame.K_DOWN: Action.DOWN,
        pygame.K_LEFT: Action.LEFT,
        pygame.K_RIGHT: Action.RIGHT,
    }

    def get_start_state():
        return (0, 0, env.hp_start)

    state = get_start_state()
    done = False
    total_reward = 0

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                    break
                if event.key == pygame.K_r:
                    state = get_start_state()
                    done = False
                    total_reward = 0
                    continue
                if not done and event.key in key_to_action:
                    action = key_to_action[event.key]
                    next_state, reward, done, dead = env.step(state, action)
                    total_reward += reward
                    state = next_state

        # Draw
        screen.fill((40, 40, 40))

        for i in range(rows):
            for j in range(cols):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                h = int(env.height_map[i, j])
                pygame.draw.rect(screen, height_color(h), rect)
                pygame.draw.rect(screen, (60, 60, 60), rect, 1)
                text = font.render(str(h), True, (255, 255, 255))
                screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

        # Agent (semi-transparent so height text is visible underneath)
        i, j, hp = state
        cx = j * cell_size + cell_size // 2
        cy = i * cell_size + cell_size // 2
        agent_surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
        pygame.draw.circle(
            agent_surf,
            (255, 200, 80, 150),
            (cell_size // 2, cell_size // 2),
            cell_size // 3,
        )
        pygame.draw.circle(
            agent_surf,
            (200, 150, 0, 220),
            (cell_size // 2, cell_size // 2),
            cell_size // 3,
            2,
        )
        screen.blit(agent_surf, (j * cell_size, i * cell_size))

        # Goal
        gj, gi = cols - 1, rows - 1
        gx = gj * cell_size + cell_size // 2
        gy = gi * cell_size + cell_size // 2
        pygame.draw.rect(screen, (0, 200, 100), (gj * cell_size + 4, gi * cell_size + 4, cell_size - 8, cell_size - 8), 3)

        # Header: HP and status
        header_rect = pygame.Rect(0, height, width, header_h)
        pygame.draw.rect(screen, (50, 50, 50), header_rect)
        hp_text = font.render(f"HP: {hp}  |  Награда: {total_reward}", True, (220, 220, 220))
        screen.blit(hp_text, (12, height + 14))

        if done:
            overlay = pygame.Surface((width, height + header_h))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            if hp <= 0:
                msg = font_big.render("Смерть", True, (255, 80, 80))
            else:
                msg = font_big.render("Победа!", True, (80, 255, 120))
            r = msg.get_rect(center=(width // 2, (height + header_h) // 2))
            screen.blit(msg, r)
            hint = font.render("R — рестарт   Q/ESC — выход", True, (180, 180, 180))
            screen.blit(hint, (width // 2 - hint.get_width() // 2, r.bottom + 12))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
