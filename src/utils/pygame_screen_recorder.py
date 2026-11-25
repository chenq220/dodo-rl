from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import ResourceDir
from scheduler.artifact import ensure_output_parent


class RenderGifRecorder:
    """
    Capture render frames and save them as a GIF.

    The recorder accepts a render callable such as ``env.render`` or a concrete
    frame. Supported frame values cover PettingZoo render outputs commonly used
    in this project: numpy arrays, PIL images, lists/tuples of frames, and
    pygame surfaces.
    """

    def __init__(
            self,
            output_dir: str | Path = ResourceDir.RECORDINGS.value,
            fps: int = 15,
            frame_stride: int = 1,
            max_frames: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.fps = max(int(fps), 1)
        self.frame_stride = max(int(frame_stride), 1)
        self.max_frames = None if max_frames is None else max(int(max_frames), 1)
        self._frames: list[Image.Image] = []
        self._capture_count = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def start(self) -> None:
        self._frames = []
        self._capture_count = 0
        self._active = True

    def capture(self, render_or_frame: Callable[[], Any] | Any) -> None:
        if not self._active:
            return
        self._capture_count += 1
        if (self._capture_count - 1) % self.frame_stride != 0:
            return
        if self._frame_limit_reached():
            return

        frame = render_or_frame() if callable(render_or_frame) else render_or_frame
        if frame is None:
            frame = self._pygame_display_surface()
        for image in self._to_images(frame):
            if self._frame_limit_reached():
                break
            self._frames.append(image)

    def finish(self, filename: str) -> Optional[Path]:
        if not self._active:
            return None
        self._active = False
        if not self._frames:
            return None

        path = self.output_dir / filename
        if path.suffix.lower() != ".gif":
            path = path.with_suffix(".gif")
        ensure_output_parent(path)

        duration_ms = max(int(1000 / self.fps), 1)
        first, *rest = self._frames
        first.save(
            path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        self._frames = []
        return path

    def discard(self) -> None:
        self._frames = []
        self._capture_count = 0
        self._active = False

    def _frame_limit_reached(self) -> bool:
        return self.max_frames is not None and len(self._frames) >= self.max_frames

    @classmethod
    def _to_images(cls, frame: Any) -> list[Image.Image]:
        if frame is None:
            return []
        if isinstance(frame, (list, tuple)):
            images: list[Image.Image] = []
            for item in frame:
                images.extend(cls._to_images(item))
            return images
        image = cls._to_image(frame)
        return [] if image is None else [image]

    @staticmethod
    def _to_image(frame: Any) -> Optional[Image.Image]:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, str):
            return RenderGifRecorder._text_to_image(frame)

        if hasattr(frame, "get_size") and hasattr(frame, "get_view"):
            import pygame

            array = pygame.surfarray.array3d(frame)
            array = np.transpose(array, (1, 0, 2))
            return Image.fromarray(RenderGifRecorder._as_uint8(array), mode="RGB")

        array = np.asarray(frame)
        if array.ndim == 2:
            return Image.fromarray(RenderGifRecorder._as_uint8(array), mode="L").convert("RGB")
        if array.ndim != 3:
            return None
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[2] == 4:
            return Image.fromarray(RenderGifRecorder._as_uint8(array), mode="RGBA").convert("RGB")
        if array.shape[2] >= 3:
            return Image.fromarray(RenderGifRecorder._as_uint8(array[:, :, :3]), mode="RGB")
        return None

    @staticmethod
    def _as_uint8(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array
        if array.dtype == np.bool_:
            return array.astype(np.uint8) * 255
        if np.issubdtype(array.dtype, np.floating):
            finite = array[np.isfinite(array)]
            scale = 255.0 if finite.size == 0 or float(np.nanmax(finite)) <= 1.0 else 1.0
            return np.clip(array * scale, 0, 255).astype(np.uint8)
        return np.clip(array, 0, 255).astype(np.uint8)

    @staticmethod
    def _text_to_image(text: str) -> Image.Image:
        font = ImageFont.load_default()
        lines = text.splitlines() or [""]
        padding = 8
        line_height = 14
        width = max((len(line) for line in lines), default=1) * 8 + padding * 2
        height = max(len(lines), 1) * line_height + padding * 2
        image = Image.new("RGB", (max(width, 64), max(height, 32)), "white")
        draw = ImageDraw.Draw(image)
        y = padding
        for line in lines:
            draw.text((padding, y), line, fill="black", font=font)
            y += line_height
        return image

    @staticmethod
    def _pygame_display_surface() -> Any:
        try:
            import pygame
        except ImportError:
            return None
        if not pygame.get_init():
            return None
        return pygame.display.get_surface()


PygameEpisodeRecorder = RenderGifRecorder
