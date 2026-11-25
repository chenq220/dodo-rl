"""蓝球连线选择目标：验证离散目标选择塑形函数。"""

from __future__ import annotations

import random
from typing import Literal, Sequence

import pygame

from envs.reward.shaping import target_selection_shaping
from envs.reward.utils import ShapingOutputRange


class TargetSelectionGame:
    """上方蓝球通过连线选择下方十个目标球中的一个。"""

    WIDTH = 800
    HEIGHT = 420
    FPS = 60
    NUM_TARGETS = 10
    NUM_SHAPING_TARGETS = 4

    CONTROLLER_POS = (WIDTH // 2, 92)
    TARGET_Y = 265
    CONTROLLER_RADIUS = 18
    TARGET_RADIUS = 20

    BG_COLOR = (24, 24, 32)
    TEXT_COLOR = (230, 230, 235)
    CONTROLLER_COLOR = (65, 130, 245)
    SHAPING_COLOR = (242, 196, 58)
    PRIMARY_COLOR = (225, 68, 68)
    OTHER_COLOR = (120, 124, 136)
    LINE_COLOR = (105, 170, 255)

    def __init__(
        self,
        render_mode: Literal["human", "rgb_array"] | None = None,
        *,
        primary_mode: Literal["never", "random"] = "random",
        primary_probability: float = 0.5,
        output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
        seed: int | None = None,
    ):
        self._validate_primary_options(primary_mode, primary_probability)
        self.render_mode = render_mode
        self.primary_mode = primary_mode
        self.primary_probability = float(primary_probability)
        self.output_range = output_range
        self._rng = random.Random(seed)

        self.shaping_target_indices: tuple[int, ...] = ()
        self.primary_target_indices: tuple[int, ...] = ()
        self.last_shaping_target_indices: tuple[int, ...] = ()
        self.last_primary_target_indices: tuple[int, ...] = ()
        self.selected_target: int | None = None
        self.last_target_kind = "none"
        self.last_primary_reward = 0.0
        self.last_shaping_reward = self._neutral_shaping_reward()
        self.last_total_reward = self.last_shaping_reward
        self.frame_index = 0
        self.primary_hits = 0
        self.primary_opportunities = 0
        self.shaping_hits = 0
        self.shaping_opportunities = 0

        self.is_running = True
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.title_font: pygame.font.Font | None = None
        if render_mode is not None:
            self._init_pygame()
        self.reset(seed=seed)

    @classmethod
    def target_positions(cls) -> list[tuple[int, int]]:
        margin = 58
        spacing = (cls.WIDTH - 2 * margin) / (cls.NUM_TARGETS - 1)
        return [(round(margin + index * spacing), cls.TARGET_Y) for index in range(cls.NUM_TARGETS)]

    @staticmethod
    def _validate_primary_options(primary_mode: str, primary_probability: float) -> None:
        if primary_mode not in ("never", "random"):
            raise ValueError(f"unsupported primary_mode={primary_mode!r}")
        if not 0.0 <= float(primary_probability) <= 1.0:
            raise ValueError("primary_probability must be in [0, 1]")

    def _neutral_shaping_reward(self) -> float:
        return 0.5 if self.output_range == ShapingOutputRange.DELTA_UNIT else 0.0

    def _init_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Target Selection Shaping Demo")
            self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 14)
        self.title_font = pygame.font.SysFont("menlo,consolas,monospace", 18, bold=True)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        self.selected_target = None
        self.last_shaping_target_indices = ()
        self.last_primary_target_indices = ()
        self.last_target_kind = "none"
        self.last_primary_reward = 0.0
        self.last_shaping_reward = self._neutral_shaping_reward()
        self.last_total_reward = self.last_shaping_reward
        self.frame_index = 0
        self.primary_hits = self.primary_opportunities = 0
        self.shaping_hits = self.shaping_opportunities = 0
        self._sample_target_roles()

    def _sample_target_roles(self) -> None:
        all_targets = tuple(range(self.NUM_TARGETS))
        self.shaping_target_indices = tuple(
            sorted(self._rng.sample(all_targets, self.NUM_SHAPING_TARGETS))
        )
        can_spawn_primary = (
            self.primary_mode == "random" and self._rng.random() < self.primary_probability
        )
        if can_spawn_primary:
            candidates = [
                target for target in all_targets if target not in self.shaping_target_indices
            ]
            self.primary_target_indices = (self._rng.choice(candidates),)
        else:
            self.primary_target_indices = ()

    def set_primary_mode(self, primary_mode: Literal["never", "random"]) -> None:
        self._validate_primary_options(primary_mode, self.primary_probability)
        self.primary_mode = primary_mode
        self.reset()

    def step_action(self, target_index: int) -> float:
        """连接蓝球与目标球，结算本步奖励，再随机下一步的目标颜色。"""
        target_index = int(target_index)
        if not 0 <= target_index < self.NUM_TARGETS:
            raise ValueError(f"target_index must be in [0, {self.NUM_TARGETS - 1}]")

        shaping_targets = self.shaping_target_indices
        primary_targets = self.primary_target_indices
        # 渲染必须保留策略作出本次选择时看到的颜色；结算后会立即采样下一状态。
        self.last_shaping_target_indices = shaping_targets
        self.last_primary_target_indices = primary_targets
        self.selected_target = target_index
        if target_index in primary_targets:
            self.last_target_kind = "primary"
        elif target_index in shaping_targets:
            self.last_target_kind = "shaping"
        else:
            self.last_target_kind = "other"

        self.last_primary_reward = float(target_index in primary_targets)
        self.last_shaping_reward = target_selection_shaping(
            target_index,
            shaping_targets,
            tuple(range(self.NUM_TARGETS)),
            primary_target_indices=primary_targets,
            available_target_indices=tuple(range(self.NUM_TARGETS)),
            output_range=self.output_range,
        )
        self.last_total_reward = self.last_primary_reward + self.last_shaping_reward

        if primary_targets:
            self.primary_opportunities += 1
            self.primary_hits += int(target_index in primary_targets)
        else:
            self.shaping_opportunities += 1
            self.shaping_hits += int(target_index in shaping_targets)

        self.frame_index += 1
        self._sample_target_roles()
        return self.last_total_reward

    def target_kind(self, target_index: int) -> str:
        if target_index in self.primary_target_indices:
            return "primary"
        if target_index in self.shaping_target_indices:
            return "shaping"
        return "other"

    def _target_color(
        self,
        target_index: int,
        shaping_targets: Sequence[int],
        primary_targets: Sequence[int],
    ) -> tuple[int, int, int]:
        if target_index in primary_targets:
            kind = "primary"
        elif target_index in shaping_targets:
            kind = "shaping"
        else:
            kind = "other"
        if kind == "primary":
            return self.PRIMARY_COLOR
        if kind == "shaping":
            return self.SHAPING_COLOR
        return self.OTHER_COLOR

    def draw(self, *, show_last_decision: bool = False) -> pygame.Surface:
        if self.font is None:
            self._init_pygame()
        assert self.font is not None and self.title_font is not None
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.BG_COLOR)

        title = self.title_font.render("Blue Ball Target Selection", True, self.TEXT_COLOR)
        surface.blit(title, title.get_rect(center=(self.WIDTH // 2, 24)))
        mode = self.font.render(
            f"primary: {self.primary_mode.upper()}  |  M toggle  R reset  ESC quit",
            True,
            (185, 188, 200),
        )
        surface.blit(mode, mode.get_rect(center=(self.WIDTH // 2, 49)))

        positions = self.target_positions()
        if show_last_decision and self.selected_target is not None:
            display_shaping_targets = self.last_shaping_target_indices
            display_primary_targets = self.last_primary_target_indices
        else:
            display_shaping_targets = self.shaping_target_indices
            display_primary_targets = self.primary_target_indices
        if self.selected_target is not None:
            pygame.draw.line(
                surface,
                self.LINE_COLOR,
                self.CONTROLLER_POS,
                positions[self.selected_target],
                4,
            )

        pygame.draw.circle(
            surface,
            self.CONTROLLER_COLOR,
            self.CONTROLLER_POS,
            self.CONTROLLER_RADIUS,
        )
        pygame.draw.circle(
            surface,
            (190, 215, 255),
            self.CONTROLLER_POS,
            self.CONTROLLER_RADIUS,
            2,
        )

        for target_index, position in enumerate(positions):
            pygame.draw.circle(
                surface,
                self._target_color(
                    target_index,
                    display_shaping_targets,
                    display_primary_targets,
                ),
                position,
                self.TARGET_RADIUS,
            )
            pygame.draw.circle(surface, self.TEXT_COLOR, position, self.TARGET_RADIUS, 1)
            label = self.font.render(str(target_index + 1), True, (20, 20, 24))
            surface.blit(label, label.get_rect(center=position))

        primary_rate = self.primary_hits / max(self.primary_opportunities, 1)
        shaping_rate = self.shaping_hits / max(self.shaping_opportunities, 1)
        lines = [
            "yellow = shaping target   red = primary target   gray = other",
            (
                f"last={self.selected_target!s:>4} ({self.last_target_kind})  "
                f"main={self.last_primary_reward:+.1f}  "
                f"shape={self.last_shaping_reward:+.1f}  total={self.last_total_reward:+.1f}"
            ),
            (
                f"primary hit={primary_rate:.1%} ({self.primary_hits}/{self.primary_opportunities})  "
                f"absence shaping hit={shaping_rate:.1%} "
                f"({self.shaping_hits}/{self.shaping_opportunities})"
            ),
            "Press 1-9 or 0 (target 10) to connect the blue ball.",
        ]
        for row, text in enumerate(lines):
            rendered = self.font.render(text, True, self.TEXT_COLOR)
            surface.blit(rendered, (34, 320 + row * 22))

        if self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        return surface

    @staticmethod
    def _key_to_target(key: int) -> int | None:
        if pygame.K_1 <= key <= pygame.K_9:
            return key - pygame.K_1
        if key == pygame.K_0:
            return 9
        return None

    def run(self) -> None:
        if self.render_mode != "human":
            raise ValueError("run() requires render_mode='human'")
        while self.is_running:
            assert self.clock is not None
            self.clock.tick(self.FPS)
            for event in pygame.event.get():
                escape = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                if event.type == pygame.QUIT or escape:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    next_mode = "never" if self.primary_mode == "random" else "random"
                    self.set_primary_mode(next_mode)
                elif event.type == pygame.KEYDOWN:
                    target = self._key_to_target(event.key)
                    if target is not None:
                        self.step_action(target)
            self.draw()
        pygame.quit()


if __name__ == "__main__":
    TargetSelectionGame(render_mode="human", seed=42).run()
