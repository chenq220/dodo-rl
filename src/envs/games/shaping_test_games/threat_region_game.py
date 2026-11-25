"""蓝点从北侧绕开中央威胁区域抵达南侧引导点：测试 region_threat 塑形函数（逻辑 + pygame 可视化）。"""

from __future__ import annotations

import math
import random
from typing import Literal, Sequence

import pygame

from envs.reward.shaping import (
    region_threat_score,
    region_threat_shaping,
    region_threat_transition_score,
)
from envs.reward.utils import (
    Point,
    ShapingOutputRange,
    pbrs_shaping,
    point_in_region,
    point_to_region_boundary_distance,
    segment_intersects_region,
)

# 四向移动，无停止动作
_ACTION_DELTA: dict[int, tuple[float, float]] = {
    0: (0.0, -1.0),
    1: (0.0, 1.0),
    2: (-1.0, 0.0),
    3: (1.0, 0.0),
}


class ThreatRegionGame:
    WIDTH = 640
    HEIGHT = 480
    FPS = 60
    NUM_ACTIONS = 4

    PLAYER_RADIUS = 10
    GOAL_RADIUS = 8
    PLAYER_SPEED = 4.0
    THREAT_SPEED = 2.0  # 观测归一化参考；配置 threat_speed 默认亦取此值
    THREAT_WIDTH = WIDTH / 3.0
    THREAT_HEIGHT = 24.0

    PLAYER_COLOR = (60, 120, 255)
    GOAL_COLOR = (220, 50, 50)
    THREAT_FILL = (180, 40, 40, 90)
    THREAT_BORDER = (255, 80, 80)
    GUIDE_LINE_COLOR = (120, 120, 140)
    BG_COLOR = (24, 24, 32)
    TEXT_COLOR = (230, 230, 230)
    REWARD_POS = (80, 220, 80)
    REWARD_NEG = (220, 80, 80)

    @classmethod
    def spawn_point(cls) -> Point:
        margin = cls.PLAYER_RADIUS + 4
        return (cls.WIDTH / 2.0, float(margin))

    @classmethod
    def goal_point(cls) -> Point:
        margin = cls.GOAL_RADIUS + 4
        return (cls.WIDTH / 2.0, cls.HEIGHT - float(margin))

    @staticmethod
    def build_threat_rect(cx: float, cy: float, width: float, height: float) -> list[Point]:
        hw = width / 2.0
        hh = height / 2.0
        return [
            (cx - hw, cy - hh),
            (cx + hw, cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh),
        ]

    def __init__(
        self,
        render_mode: Literal["human", "rgb_array"] | None = None,
        *,
        threat_speed: float = 0.0,
        approach_decay_coef: float = 1.0,
        gamma: float = 1.0,
        output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
        seed: int | None = None,
    ):
        self.render_mode = render_mode
        self.threat_speed = float(threat_speed)
        self._configured_threat_speed = self.threat_speed
        self.approach_decay_coef = approach_decay_coef
        self.gamma = gamma
        self.output_range = output_range
        self._rng = random.Random(seed)

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.title_font: pygame.font.Font | None = None

        self.is_running = True
        self.observer: Point = self.spawn_point()
        self._prev_observer: Point = self.observer
        self.goal: Point = self.goal_point()
        self.threat_center: Point = (self.WIDTH / 2.0, self.HEIGHT / 2.0)
        self.threat_velocity: Point = (0.0, 0.0)
        self.threat_vertices: list[Point] = []
        self._prev_threat_vertices: list[Point] = []
        self.phi = 0.0
        self.step_reward = 0.0
        self.in_threat = False
        self.crossed_threat = False
        self.dist_outside = 0.0
        self._frame = 0
        self._step_count = 0

        if render_mode is not None:
            self._init_pygame()

        self.reset(seed=seed)

    def _init_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Threat Region Shaping Demo")
            self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 14)
        self.title_font = pygame.font.SysFont("menlo,consolas,monospace", 16, bold=True)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        self.observer = self.spawn_point()
        self._prev_observer = self.observer
        self.goal = self.goal_point()
        self._reset_threat()
        self._frame = 0
        self._step_count = 0
        self._update_metrics(reset=True)

    def _reset_threat(self) -> None:
        self.threat_center = (self.WIDTH / 2.0, self.HEIGHT / 2.0)
        self._rebuild_threat_vertices()
        if self.threat_speed == 0.0:
            self.threat_velocity = (0.0, 0.0)
        else:
            direction = 1.0 if self.threat_speed > 0 else -1.0
            self.threat_velocity = (direction * abs(self.threat_speed), 0.0)

    def _rebuild_threat_vertices(self) -> None:
        self.threat_vertices = self.build_threat_rect(
            self.threat_center[0],
            self.threat_center[1],
            self.THREAT_WIDTH,
            self.THREAT_HEIGHT,
        )

    def step_action(self, action_id: int) -> None:
        action_id = int(action_id) % self.NUM_ACTIONS
        dx, dy = _ACTION_DELTA[action_id]
        self._apply_movement(dx * self.PLAYER_SPEED, dy * self.PLAYER_SPEED)
        self._update_threat()
        self._update_metrics()

    def step_delta(self, dx: float, dy: float) -> None:
        if dx or dy:
            self._apply_movement(dx, dy)
        self._update_threat()
        self._update_metrics()

    def _update_threat(self) -> None:
        self._frame += 1
        if self.threat_speed == 0.0:
            return

        margin = 4.0
        cx, cy = self.threat_center
        vx, _ = self.threat_velocity
        cx += vx
        half_w = self.THREAT_WIDTH / 2.0
        if cx - half_w < margin:
            cx = margin + half_w
            vx = abs(vx)
        elif cx + half_w > self.WIDTH - margin:
            cx = self.WIDTH - margin - half_w
            vx = -abs(vx)
        self.threat_center = (cx, cy)
        self.threat_velocity = (vx, 0.0)
        self._rebuild_threat_vertices()

    def _apply_movement(self, dx: float, dy: float) -> None:
        ox, oy = self.observer
        nx = max(self.PLAYER_RADIUS, min(self.WIDTH - self.PLAYER_RADIUS, ox + dx))
        ny = max(self.PLAYER_RADIUS, min(self.HEIGHT - self.PLAYER_RADIUS, oy + dy))
        self.observer = (nx, ny)

    def _threat_shaping_step(
        self,
        pre_xy: Point,
        post_xy: Point,
        pre_region: Sequence[Point],
        post_region: Sequence[Point],
    ) -> float:
        """
        region_threat_shaping 只接受单个 region；移动威胁区时步前/步后矩形不同，
        需分别用 pre_region / post_region 计 Φ，见 ThreatRegionEnv._threat_shaping。
        """
        targets_pre = [pre_xy]
        targets_post = [post_xy]
        if pre_region == post_region:
            return region_threat_shaping(
                targets_pre,
                targets_post,
                post_region,
                self.approach_decay_coef,
                self.gamma,
                self.output_range,
            )
        phi_pre = region_threat_score(
            targets_pre, pre_region, self.approach_decay_coef, self.output_range,
        )
        phi_post = region_threat_transition_score(
            targets_pre, targets_post, post_region, self.approach_decay_coef, self.output_range,
        )
        return pbrs_shaping(phi_pre, phi_post, self.gamma, self.output_range)

    def _update_metrics(self, *, reset: bool = False) -> None:
        ox, oy = self.observer
        self.in_threat = point_in_region(ox, oy, self.threat_vertices)
        self.crossed_threat = segment_intersects_region(
            self._prev_observer, self.observer, self.threat_vertices,
        )
        self.dist_outside = (
            0.0
            if self.in_threat
            else point_to_region_boundary_distance(ox, oy, self.threat_vertices)
        )
        self.phi = region_threat_score(
            [self.observer],
            self.threat_vertices,
            self.approach_decay_coef,
            self.output_range,
        )

        if reset:
            self.step_reward = 0.0
            self._prev_observer = self.observer
            self._prev_threat_vertices = list(self.threat_vertices)
            return

        self._step_count += 1
        self.step_reward = self._threat_shaping_step(
            self._prev_observer,
            self.observer,
            self._prev_threat_vertices,
            self.threat_vertices,
        )
        self._prev_observer = self.observer
        self._prev_threat_vertices = list(self.threat_vertices)

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
            self._update_threat()
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
                    if self.threat_speed == 0.0:
                        self.threat_speed = self._configured_threat_speed
                    else:
                        self.threat_speed = 0.0
                    self._reset_threat()
                    self._update_metrics(reset=True)

    def draw(self) -> pygame.Surface:
        if self.font is None:
            self._init_pygame()
        assert self.font is not None and self.title_font is not None

        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.BG_COLOR)
        self._draw_guide_line(surface)
        self._draw_threat(surface)

        gx, gy = self.goal
        pygame.draw.circle(surface, self.GOAL_COLOR, (int(gx), int(gy)), self.GOAL_RADIUS)
        pygame.draw.circle(surface, (255, 120, 120), (int(gx), int(gy)), self.GOAL_RADIUS, 2)

        ox, oy = self.observer
        pygame.draw.circle(surface, self.PLAYER_COLOR, (int(ox), int(oy)), self.PLAYER_RADIUS)
        pygame.draw.circle(surface, (200, 220, 255), (int(ox), int(oy)), self.PLAYER_RADIUS, 2)

        self._draw_hud(surface)

        if self.render_mode == "human" and self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        return surface

    def _draw_guide_line(self, surface: pygame.Surface) -> None:
        ox, oy = self.observer
        gx, gy = self.goal
        pygame.draw.line(surface, self.GUIDE_LINE_COLOR, (int(ox), int(oy)), (int(gx), int(gy)), 1)

    def _draw_threat(self, surface: pygame.Surface) -> None:
        if len(self.threat_vertices) < 3:
            return
        points = [(int(x), int(y)) for x, y in self.threat_vertices]
        fill = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(fill, self.THREAT_FILL, points)
        surface.blit(fill, (0, 0))
        pygame.draw.polygon(surface, self.THREAT_BORDER, points, 2)

    def _draw_hud(self, surface: pygame.Surface) -> None:
        assert self.font is not None and self.title_font is not None
        reward_color = self.REWARD_POS if self.step_reward >= 0 else self.REWARD_NEG

        lines = [
            ("Threat Region Shaping Demo", self.title_font, self.TEXT_COLOR),
            ("", self.font, self.TEXT_COLOR),
            (f"step: {self._step_count}", self.font, self.TEXT_COLOR),
            ("red goal: guide only (not in reward)", self.font, (180, 140, 140)),
            (f"threat_speed: {self.threat_speed:+.1f}px/step (0=static)", self.font, self.TEXT_COLOR),
            (f"in threat: {'yes' if self.in_threat else 'no'}", self.font, self.TEXT_COLOR),
            (f"crossed segment: {'yes' if self.crossed_threat else 'no'}", self.font, self.TEXT_COLOR),
            (f"dist outside: {self.dist_outside:.1f}px", self.font, self.TEXT_COLOR),
            (f"Φ (potential):   {self.phi:.3f}", self.font, self.TEXT_COLOR),
            (f"F (step reward): {self.step_reward:+.3f}", self.font, reward_color),
            ("", self.font, self.TEXT_COLOR),
            ("WASD/Arrows move  |  M toggle threat  |  R reset  |  ESC quit", self.font, (160, 160, 170)),
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
    ThreatRegionGame(render_mode="human", seed=42).run()
