"""蓝点追踪红点：测试 region_visible_target_seek 塑形函数（逻辑 + pygame 可视化）。"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Literal, Sequence

import pygame

from envs.reward.shaping import (
    region_visible_target_seek_score,
    region_visible_target_seek_shaping,
)
from envs.reward.utils import (
    Point,
    ShapingOutputRange,
    nearest_target,
    nearest_target_in_region,
    point_distance,
)


@dataclass
class MovingTarget:
    x: float
    y: float
    vx: float
    vy: float


_ACTION_DELTA: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, -1.0),
    2: (0.0, 1.0),
    3: (-1.0, 0.0),
    4: (1.0, 0.0),
}


class TargetSeekGame:
    WIDTH = 640
    HEIGHT = 480
    FPS = 60
    NUM_ACTIONS = 5

    PLAYER_RADIUS = 10
    TARGET_RADIUS = 8
    PLAYER_SPEED = 4.0
    TARGET_SPEED = 2.2
    TARGET_JITTER_INTERVAL = 90

    PLAYER_COLOR = (60, 120, 255)
    TARGET_COLOR = (220, 50, 50)
    NEAREST_LINE_COLOR = (255, 200, 80)
    IDEAL_RING_COLOR = (100, 200, 100)
    REGION_FILL = (40, 80, 40, 60)
    REGION_BORDER = (80, 160, 80)
    BG_COLOR = (24, 24, 32)
    TEXT_COLOR = (230, 230, 230)
    REWARD_POS = (80, 220, 80)
    REWARD_NEG = (220, 80, 80)

    @classmethod
    def game_region(cls) -> list[Point]:
        """游戏区域：整个正方形框。"""
        return [
            (0.0, 0.0),
            (float(cls.WIDTH), 0.0),
            (float(cls.WIDTH), float(cls.HEIGHT)),
            (0.0, float(cls.HEIGHT)),
        ]

    def __init__(
        self,
        render_mode: Literal["human", "rgb_array"] | None = None,
        *,
        num_targets: int = 1,
        targets_moving: bool = True,
        target_distance: float | None = None,
        max_deviation: float = 640.0,
        gamma: float = 1.0,
        output_range: ShapingOutputRange = ShapingOutputRange.UNIT,
        region: Sequence[Point] | None = None,
        seed: int | None = None,
    ):
        self.render_mode = render_mode
        self.num_targets = max(1, num_targets)
        self.max_deviation = max_deviation
        self.gamma = gamma
        self.output_range = output_range
        self._region = list(region) if region is not None else None
        self._rng = random.Random(seed)

        if target_distance is None:
            self.target_distance = 0.0
        else:
            self.target_distance = target_distance

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.title_font: pygame.font.Font | None = None

        self.is_running = True
        self.use_region = self._region is not None
        self.targets_moving = targets_moving
        self.targets: list[MovingTarget] = []
        self._prev_targets: list[Point] = []
        self.observer: Point = (self.WIDTH / 2, self.HEIGHT / 2)
        self._prev_observer: Point = self.observer
        self.phi = 0.0
        self.step_reward = 0.0
        self.dist_to_nearest = 0.0
        self.nearest_index: int | None = None
        self._frame = 0

        if render_mode is not None:
            self._init_pygame()

        self.reset(seed=seed)

    def _init_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Target Seek Shaping Demo")
            self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 14)
        self.title_font = pygame.font.SysFont("menlo,consolas,monospace", 16, bold=True)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        margin = 40
        self.observer = (self.WIDTH / 2, self.HEIGHT / 2)
        self._prev_observer = self.observer
        self.targets = [self._spawn_target(margin) for _ in range(self.num_targets)]
        self._frame = 0
        self._update_metrics(reset=True)

    def step_action(self, action_id: int) -> None:
        dx, dy = _ACTION_DELTA.get(int(action_id), (0.0, 0.0))
        self._apply_movement(dx * self.PLAYER_SPEED, dy * self.PLAYER_SPEED)
        self._update_targets()

    def step_delta(self, dx: float, dy: float) -> None:
        self._apply_movement(dx, dy)
        self._update_targets()
        self._update_metrics()

    def target_points(self) -> list[Point]:
        return [(t.x, t.y) for t in self.targets]

    def _spawn_target(self, margin: float) -> MovingTarget:
        x = self._rng.uniform(margin, self.WIDTH - margin)
        y = self._rng.uniform(margin, self.HEIGHT - margin)
        angle = self._rng.uniform(0.0, 2.0 * math.pi)
        speed = self._rng.uniform(self.TARGET_SPEED * 0.6, self.TARGET_SPEED * 1.2)
        return MovingTarget(
            x=x,
            y=y,
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed,
        )

    def _active_region(self) -> list[Point] | None:
        if not self.use_region:
            return None
        if self._region is not None:
            return self._region
        return [
            (0.0, 0.0),
            (self.WIDTH / 2, 0.0),
            (self.WIDTH / 2, self.HEIGHT),
            (0.0, self.HEIGHT),
        ]

    def _resolve_nearest_index(self, points: Sequence[Point]) -> int | None:
        region = self._active_region()
        if not points:
            return None
        if region is None:
            nearest = nearest_target(self.observer, points)
        else:
            nearest = nearest_target_in_region(self.observer, points, region)
        if nearest is None:
            return None
        for i, pt in enumerate(points):
            if pt == nearest:
                return i
        return None

    def _update_metrics(self, *, reset: bool = False) -> None:
        region = self._active_region()
        targets_post = self.target_points()
        self.nearest_index = self._resolve_nearest_index(targets_post)
        if self.nearest_index is not None:
            tx, ty = targets_post[self.nearest_index]
            ox, oy = self.observer
            self.dist_to_nearest = point_distance(ox, oy, tx, ty)
        else:
            self.dist_to_nearest = float("inf")

        self.phi = region_visible_target_seek_score(
            self.observer,
            targets_post,
            self.target_distance,
            self.max_deviation,
            self.output_range,
            region,
        )

        if reset:
            self.step_reward = 0.0
            self._prev_observer = self.observer
            self._prev_targets = targets_post
            return

        self.step_reward = region_visible_target_seek_shaping(
            self._prev_observer,
            self.observer,
            self._prev_targets,
            targets_post,
            self.target_distance,
            self.max_deviation,
            self.gamma,
            self.output_range,
            region,
        )
        self._prev_observer = self.observer
        self._prev_targets = targets_post

    def _apply_movement(self, dx: float, dy: float) -> None:
        if not dx and not dy:
            return
        ox, oy = self.observer
        nx = max(self.PLAYER_RADIUS, min(self.WIDTH - self.PLAYER_RADIUS, ox + dx))
        ny = max(self.PLAYER_RADIUS, min(self.HEIGHT - self.PLAYER_RADIUS, oy + dy))
        self.observer = (nx, ny)

    def _update_targets(self) -> None:
        self._frame += 1
        if not self.targets_moving:
            return

        margin = self.TARGET_RADIUS + 2
        for target in self.targets:
            target.x += target.vx
            target.y += target.vy

            if target.x < margin:
                target.x = margin
                target.vx = abs(target.vx)
            elif target.x > self.WIDTH - margin:
                target.x = self.WIDTH - margin
                target.vx = -abs(target.vx)

            if target.y < margin:
                target.y = margin
                target.vy = abs(target.vy)
            elif target.y > self.HEIGHT - margin:
                target.y = self.HEIGHT - margin
                target.vy = -abs(target.vy)

        if self._frame % self.TARGET_JITTER_INTERVAL == 0:
            for target in self.targets:
                angle = self._rng.uniform(-0.6, 0.6)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                vx, vy = target.vx, target.vy
                target.vx = vx * cos_a - vy * sin_a
                target.vy = vx * sin_a + vy * cos_a
                speed = math.hypot(target.vx, target.vy)
                if speed > 0:
                    target.vx = target.vx / speed * self.TARGET_SPEED
                    target.vy = target.vy / speed * self.TARGET_SPEED

    def step(self) -> None:
        keys = pygame.key.get_pressed()
        dx = dy = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= self.PLAYER_SPEED
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += self.PLAYER_SPEED
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= self.PLAYER_SPEED
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += self.PLAYER_SPEED
        self.step_delta(dx, dy)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_n:
                    self.targets.append(self._spawn_target(40))
                    self._update_metrics(reset=True)
                elif event.key == pygame.K_t:
                    self.use_region = not self.use_region
                    self._update_metrics(reset=True)
                elif event.key == pygame.K_m:
                    self.targets_moving = not self.targets_moving
                    self._update_metrics(reset=True)

    def draw(self) -> pygame.Surface:
        if self.font is None:
            self._init_pygame()
        assert self.font is not None and self.title_font is not None

        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.BG_COLOR)

        self._draw_region(surface)
        self._draw_ideal_ring(surface)
        self._draw_nearest_line(surface)

        for i, target in enumerate(self.targets):
            is_nearest = i == self.nearest_index
            color = (255, 120, 120) if is_nearest else self.TARGET_COLOR
            radius = self.TARGET_RADIUS + (2 if is_nearest else 0)
            pygame.draw.circle(surface, color, (int(target.x), int(target.y)), radius)

        ox, oy = self.observer
        pygame.draw.circle(surface, self.PLAYER_COLOR, (int(ox), int(oy)), self.PLAYER_RADIUS)
        pygame.draw.circle(surface, (200, 220, 255), (int(ox), int(oy)), self.PLAYER_RADIUS, 2)

        self._draw_hud(surface)

        if self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        return surface

    def _draw_region(self, surface: pygame.Surface) -> None:
        region = self._active_region()
        if region is None:
            return
        pts = [(int(x), int(y)) for x, y in region]
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(overlay, self.REGION_FILL, pts)
        surface.blit(overlay, (0, 0))
        pygame.draw.polygon(surface, self.REGION_BORDER, pts, 2)

    def _draw_ideal_ring(self, surface: pygame.Surface) -> None:
        if self.nearest_index is None or self.target_distance <= 0:
            return
        target = self.targets[self.nearest_index]
        pygame.draw.circle(
            surface,
            self.IDEAL_RING_COLOR,
            (int(target.x), int(target.y)),
            int(self.target_distance),
            1,
        )

    def _draw_nearest_line(self, surface: pygame.Surface) -> None:
        if self.nearest_index is None:
            return
        ox, oy = self.observer
        target = self.targets[self.nearest_index]
        pygame.draw.line(
            surface,
            self.NEAREST_LINE_COLOR,
            (int(ox), int(oy)),
            (int(target.x), int(target.y)),
            2,
        )

    def _draw_hud(self, surface: pygame.Surface) -> None:
        assert self.font is not None and self.title_font is not None
        deviation = abs(self.dist_to_nearest - self.target_distance)
        reward_color = self.REWARD_POS if self.step_reward >= 0 else self.REWARD_NEG

        lines = [
            ("Target Seek Shaping Demo", self.title_font, self.TEXT_COLOR),
            ("", self.font, self.TEXT_COLOR),
            (f"ideal distance d: {self.target_distance:.1f}px", self.font, self.TEXT_COLOR),
            (f"Region: {'on (left half)' if self.use_region else 'off (global)'}", self.font, self.TEXT_COLOR),
            (f"Target move: {'on' if self.targets_moving else 'off (static)'}", self.font, self.TEXT_COLOR),
            (f"Targets: {len(self.targets)}", self.font, self.TEXT_COLOR),
            ("", self.font, self.TEXT_COLOR),
            (f"dist to nearest: {self.dist_to_nearest:.1f}px", self.font, self.TEXT_COLOR),
            (f"ideal distance:  {self.target_distance:.1f}px", self.font, self.TEXT_COLOR),
            (f"deviation:       {deviation:.1f}px / {self.max_deviation:.0f}px", self.font, self.TEXT_COLOR),
            (f"Φ (potential):   {self.phi:.3f}", self.font, self.TEXT_COLOR),
            (f"F (step reward): {self.step_reward:+.3f}", self.font, reward_color),
            ("", self.font, self.TEXT_COLOR),
            ("WASD/Arrows move  |  R reset  |  N add target", self.font, (160, 160, 170)),
            ("T region  |  M target move  |  ESC quit", self.font, (160, 160, 170)),
        ]

        y = 12
        for text, font, color in lines:
            if not text:
                y += 6
                continue
            rendered = font.render(text, True, color)
            surface.blit(rendered, (12, y))
            y += rendered.get_height() + 2

        if self.nearest_index is None and self.use_region:
            warn = self.font.render("No target in region!", True, (255, 180, 80))
            surface.blit(warn, (12, self.HEIGHT - 28))

    def run(self) -> None:
        if self.render_mode != "human":
            raise ValueError("run() requires render_mode='human'")
        while self.is_running:
            assert self.clock is not None
            self.clock.tick(self.FPS)
            self._handle_events()
            if not self.is_running:
                break
            self.step()
            self.draw()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    TargetSeekGame(render_mode="human", seed=42).run()
