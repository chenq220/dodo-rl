"""蓝点在随机六边形目标区域内活动：测试 region_activity 塑形函数（逻辑 + pygame 可视化）。"""

from __future__ import annotations

import math
import random
from typing import Literal, Sequence

import pygame

from envs.reward.shaping import region_activity_score, region_activity_shaping
from envs.reward.utils import (
    Point,
    ShapingOutputRange,
    pbrs_shaping,
    point_in_region,
    point_to_region_boundary_distance,
)

# 四向移动，无停止动作
_ACTION_DELTA: dict[int, tuple[float, float]] = {
    0: (0.0, -1.0),
    1: (0.0, 1.0),
    2: (-1.0, 0.0),
    3: (1.0, 0.0),
}


class RegionActivityGame:
    WIDTH = 640
    HEIGHT = 480
    FPS = 60
    NUM_ACTIONS = 4

    PLAYER_RADIUS = 10
    PLAYER_SPEED = 4.0
    REGION_SPEED = 2.0

    PLAYER_COLOR = (60, 120, 255)
    HEX_FILL = (40, 100, 60, 70)
    HEX_BORDER = (80, 200, 100)
    BG_COLOR = (24, 24, 32)
    TEXT_COLOR = (230, 230, 230)
    REWARD_POS = (80, 220, 80)
    REWARD_NEG = (220, 80, 80)

    _SCREEN_AREA = float(WIDTH * HEIGHT)
    _MAX_HEX_AREA = _SCREEN_AREA / 4.0
    MAX_HEX_RADIUS = math.sqrt(_MAX_HEX_AREA * 2.0 / (3.0 * math.sqrt(3.0)))
    MIN_HEX_RADIUS = 36.0

    def __init__(
        self,
        render_mode: Literal["human", "rgb_array"] | None = None,
        *,
        region_moving: bool = False,
        leave_decay_coef: float = 0.01,
        gamma: float = 1.0,
        output_range: ShapingOutputRange = ShapingOutputRange.UNIT,
        seed: int | None = None,
    ):
        self.render_mode = render_mode
        self.region_moving = region_moving
        self.leave_decay_coef = leave_decay_coef
        self.gamma = gamma
        self.output_range = output_range
        self._rng = random.Random(seed)

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.title_font: pygame.font.Font | None = None

        self.is_running = True
        self.observer: Point = (self.WIDTH / 2, self.HEIGHT / 2)
        self._prev_observer: Point = self.observer
        self.hex_center: Point = (self.WIDTH / 2, self.HEIGHT / 2)
        self.hex_radius = self.MIN_HEX_RADIUS
        self.hex_rotation = 0.0
        self.hex_velocity: Point = (0.0, 0.0)
        self.hex_vertices: list[Point] = []
        self._prev_hex_vertices: list[Point] = []
        self.phi = 0.0
        self.step_reward = 0.0
        self.inside_hex = False
        self.dist_outside = 0.0
        self._frame = 0
        self._step_count = 0

        if render_mode is not None:
            self._init_pygame()

        self.reset(seed=seed)

    @staticmethod
    def build_hex_vertices(cx: float, cy: float, radius: float, rotation: float = 0.0) -> list[Point]:
        return [
            (
                cx + radius * math.cos(rotation + i * math.pi / 3.0),
                cy + radius * math.sin(rotation + i * math.pi / 3.0),
            )
            for i in range(6)
        ]

    @classmethod
    def hex_area(cls, radius: float) -> float:
        return (3.0 * math.sqrt(3.0) / 2.0) * radius * radius

    def _init_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Region Activity Shaping Demo")
            self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 14)
        self.title_font = pygame.font.SysFont("menlo,consolas,monospace", 16, bold=True)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        margin = self.PLAYER_RADIUS + 4
        self.observer = (
            self._rng.uniform(margin, self.WIDTH - margin),
            self._rng.uniform(margin, self.HEIGHT - margin),
        )
        self._prev_observer = self.observer
        self._spawn_hexagon(margin)
        self._frame = 0
        self._step_count = 0
        self._update_metrics(reset=True)

    def step_action(self, action_id: int) -> None:
        action_id = int(action_id) % self.NUM_ACTIONS
        dx, dy = _ACTION_DELTA[action_id]
        self._apply_movement(dx * self.PLAYER_SPEED, dy * self.PLAYER_SPEED)
        self._update_region()
        self._update_metrics()

    def step_delta(self, dx: float, dy: float) -> None:
        if dx or dy:
            self._apply_movement(dx, dy)
        self._update_region()
        self._update_metrics()

    def _spawn_hexagon(self, margin: float) -> None:
        for _ in range(64):
            radius = self._rng.uniform(self.MIN_HEX_RADIUS, self.MAX_HEX_RADIUS)
            rotation = self._rng.uniform(0.0, math.pi / 3.0)
            cx = self._rng.uniform(margin + radius, self.WIDTH - margin - radius)
            cy = self._rng.uniform(margin + radius, self.HEIGHT - margin - radius)
            vertices = self.build_hex_vertices(cx, cy, radius, rotation)
            if not self._vertices_in_bounds(vertices, margin):
                continue
            if self.hex_area(radius) > self._MAX_HEX_AREA + 1e-6:
                continue
            self.hex_center = (cx, cy)
            self.hex_radius = radius
            self.hex_rotation = rotation
            self.hex_vertices = vertices
            if self.region_moving:
                angle = self._rng.uniform(0.0, 2.0 * math.pi)
                speed = self._rng.uniform(self.REGION_SPEED * 0.6, self.REGION_SPEED * 1.2)
                self.hex_velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            else:
                self.hex_velocity = (0.0, 0.0)
            return
        # fallback：居中中等六边形
        self.hex_radius = min(self.MAX_HEX_RADIUS, 80.0)
        self.hex_rotation = 0.0
        self.hex_center = (self.WIDTH / 2, self.HEIGHT / 2)
        self.hex_vertices = self.build_hex_vertices(
            *self.hex_center, self.hex_radius, self.hex_rotation,
        )
        self.hex_velocity = (0.0, 0.0)

    @classmethod
    def _vertices_in_bounds(cls, vertices: Sequence[Point], margin: float) -> bool:
        for x, y in vertices:
            if x < margin or x > cls.WIDTH - margin or y < margin or y > cls.HEIGHT - margin:
                return False
        return True

    def _rebuild_hex_vertices(self) -> None:
        self.hex_vertices = self.build_hex_vertices(
            self.hex_center[0],
            self.hex_center[1],
            self.hex_radius,
            self.hex_rotation,
        )

    def _update_region(self) -> None:
        self._frame += 1
        if not self.region_moving:
            return

        margin = self.PLAYER_RADIUS + 4
        cx, cy = self.hex_center
        vx, vy = self.hex_velocity
        cx += vx
        cy += vy

        vertices = self.build_hex_vertices(cx, cy, self.hex_radius, self.hex_rotation)
        if not self._vertices_in_bounds(vertices, margin):
            if cx - self.hex_radius < margin or cx + self.hex_radius > self.WIDTH - margin:
                vx = -vx
            if cy - self.hex_radius < margin or cy + self.hex_radius > self.HEIGHT - margin:
                vy = -vy
            cx, cy = self.hex_center
            cx += vx
            cy += vy
            vertices = self.build_hex_vertices(cx, cy, self.hex_radius, self.hex_rotation)
            if not self._vertices_in_bounds(vertices, margin):
                cx, cy = self.hex_center

        self.hex_center = (cx, cy)
        self.hex_velocity = (vx, vy)
        self._rebuild_hex_vertices()

    def _apply_movement(self, dx: float, dy: float) -> None:
        ox, oy = self.observer
        nx = max(self.PLAYER_RADIUS, min(self.WIDTH - self.PLAYER_RADIUS, ox + dx))
        ny = max(self.PLAYER_RADIUS, min(self.HEIGHT - self.PLAYER_RADIUS, oy + dy))
        self.observer = (nx, ny)

    def _activity_shaping_step(
        self,
        pre_xy: Point,
        post_xy: Point,
        pre_region: Sequence[Point],
        post_region: Sequence[Point],
    ) -> float:
        """与 RegionActivityEnv._activity_shaping 相同：移动框时需 pre/post 两套 region。"""
        if pre_region is post_region or pre_region == post_region:
            return region_activity_shaping(
                pre_xy,
                post_xy,
                post_region,
                self.leave_decay_coef,
                self.gamma,
                self.output_range,
            )
        phi_pre = region_activity_score(
            pre_xy[0], pre_xy[1], pre_region, self.leave_decay_coef, self.output_range,
        )
        phi_post = region_activity_score(
            post_xy[0], post_xy[1], post_region, self.leave_decay_coef, self.output_range,
        )
        return pbrs_shaping(phi_pre, phi_post, self.gamma, self.output_range)

    def _update_metrics(self, *, reset: bool = False) -> None:
        ox, oy = self.observer
        self.inside_hex = point_in_region(ox, oy, self.hex_vertices)
        self.dist_outside = (
            0.0 if self.inside_hex else point_to_region_boundary_distance(ox, oy, self.hex_vertices)
        )
        self.phi = region_activity_score(
            ox, oy, self.hex_vertices, self.leave_decay_coef, self.output_range,
        )

        if reset:
            self.step_reward = 0.0
            self._prev_observer = self.observer
            self._prev_hex_vertices = list(self.hex_vertices)
            return

        self._step_count += 1
        self.step_reward = self._activity_shaping_step(
            self._prev_observer,
            self.observer,
            self._prev_hex_vertices,
            self.hex_vertices,
        )
        self._prev_observer = self.observer
        self._prev_hex_vertices = list(self.hex_vertices)

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
        if dx or dy:
            self.step_delta(dx, dy)
        else:
            self._update_region()
            self._update_metrics()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_m:
                    self.region_moving = not self.region_moving
                    if self.region_moving and self.hex_velocity == (0.0, 0.0):
                        angle = self._rng.uniform(0.0, 2.0 * math.pi)
                        speed = self._rng.uniform(self.REGION_SPEED * 0.6, self.REGION_SPEED * 1.2)
                        self.hex_velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
                    elif not self.region_moving:
                        self.hex_velocity = (0.0, 0.0)
                    self._update_metrics(reset=True)

    def draw(self) -> pygame.Surface:
        if self.font is None:
            self._init_pygame()
        assert self.font is not None and self.title_font is not None

        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.BG_COLOR)
        self._draw_hex(surface)

        ox, oy = self.observer
        pygame.draw.circle(surface, self.PLAYER_COLOR, (int(ox), int(oy)), self.PLAYER_RADIUS)
        pygame.draw.circle(surface, (200, 220, 255), (int(ox), int(oy)), self.PLAYER_RADIUS, 2)

        self._draw_hud(surface)

        if self.render_mode == "human" and self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        return surface

    def _draw_hex(self, surface: pygame.Surface) -> None:
        if len(self.hex_vertices) < 3:
            return
        points = [(int(x), int(y)) for x, y in self.hex_vertices]
        fill = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(fill, self.HEX_FILL, points)
        surface.blit(fill, (0, 0))
        pygame.draw.polygon(surface, self.HEX_BORDER, points, 2)

    def _draw_hud(self, surface: pygame.Surface) -> None:
        assert self.font is not None and self.title_font is not None
        reward_color = self.REWARD_POS if self.step_reward >= 0 else self.REWARD_NEG
        area_pct = self.hex_area(self.hex_radius) / self._SCREEN_AREA * 100.0

        lines = [
            ("Region Activity Shaping Demo", self.title_font, self.TEXT_COLOR),
            ("", self.font, self.TEXT_COLOR),
            (f"step: {self._step_count}", self.font, self.TEXT_COLOR),
            (f"Hex move: {'on' if self.region_moving else 'off (static)'}", self.font, self.TEXT_COLOR),
            (f"Hex radius: {self.hex_radius:.0f}px  area: {area_pct:.1f}% screen", self.font, self.TEXT_COLOR),
            (f"inside hex: {'yes' if self.inside_hex else 'no'}", self.font, self.TEXT_COLOR),
            (f"dist outside: {self.dist_outside:.1f}px", self.font, self.TEXT_COLOR),
            (f"Φ (potential):   {self.phi:.3f}", self.font, self.TEXT_COLOR),
            (f"F (step reward): {self.step_reward:+.3f}", self.font, reward_color),
            ("", self.font, self.TEXT_COLOR),
            ("WASD/Arrows move (no stop)  |  M hex move  |  R reset  |  ESC quit", self.font, (160, 160, 170)),
        ]

        y = 12
        for text, font, color in lines:
            if not text:
                y += 6
                continue
            rendered = font.render(text, True, color)
            surface.blit(rendered, (12, y))
            y += rendered.get_height() + 2

    def run(self) -> None:
        if self.render_mode != "human":
            raise ValueError("run() requires render_mode='human'")
        while self.is_running:
            assert self.clock is not None
            self.clock.tick(self.FPS)
            self._handle_events()
            self.step()
            self.draw()


if __name__ == "__main__":
    RegionActivityGame(render_mode="human", seed=42).run()
