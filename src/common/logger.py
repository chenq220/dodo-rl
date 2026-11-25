import logging
import os
from pathlib import Path
import sys
from logging.handlers import RotatingFileHandler
from urllib.parse import quote

from common.log_files import SharedLogFiles

from config import LogConfig, LogHandlerKind
from utils.task_id import task_id

CONF = LogConfig()
_CONSOLE_FORMATTER = logging.Formatter(CONF.console_log_format)
_FILE_FORMATTER = logging.Formatter(CONF.file_log_format)
_CONSOLE_HANDLER = None
_FILE_HANDLER = None
_MANAGED_LOGGER_NAMES: set[str] = set()
_DODORL_HANDLER_KIND = "_dodorl_handler_kind"
_LOG_ROLE_ID = ""


class TaskIdFilter(logging.Filter):
    """为每条日志记录补充全局 task_id 字段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.task_id = str(task_id)
        return True


_TASK_ID_FILTER = TaskIdFilter()


def log_file_path(
        role_id: str | None = None,
        log_task: str | None = None,
        *,
        configured: Path | None = None,
) -> Path:
    """按 log_dir / stem / task_id / role_id 解析活跃日志路径，不依赖本进程 handler。"""
    source = configured if configured is not None else Path(CONF.log_dir) / CONF.log_file_name
    source = source.expanduser()
    if not source.is_absolute():
        source = Path.cwd() / source
    tid = quote(str(log_task if log_task is not None else task_id), safe="-_.")
    prefix = f"{role_id}_" if role_id else ""
    name = f"{prefix}{source.stem}_{tid}{source.suffix or '.log'}"
    return source.with_name(name)


def resolve_role_log_files(role_id: str, log_task: str | None = None) -> list[Path]:
    """该 role_id 的已轮转分片 + 活跃文件；文件尚不存在时返回空列表。"""
    active = log_file_path(role_id, log_task)
    if not active.parent.exists():
        return []
    with SharedLogFiles(active).locked() as files:
        return files.files()


def set_logger_role_id(role_id: str | None) -> None:
    """在首次写文件前注入 role_id，更新已创建但 delay 的 file handler 路径。"""
    global _LOG_ROLE_ID
    _LOG_ROLE_ID = "" if role_id is None else str(role_id)
    handler = _FILE_HANDLER
    if handler is not None:
        handler._refresh_role_path()


class SafeRotatingFileHandler(RotatingFileHandler):
    """同任务多进程共写一个文件，使用文件锁协调完整日志行和轮转。"""

    def __init__(self, filename, **kwargs):
        self._configured_path = Path(filename).expanduser().resolve()
        self._log_task = str(task_id)
        self._registered = False
        self._retired = False
        kwargs["delay"] = True
        super().__init__(self._task_path(), **kwargs)

    def _task_path(self) -> str:
        return str(log_file_path(_LOG_ROLE_ID, self._log_task, configured=self._configured_path))

    def _refresh_role_path(self) -> None:
        self.baseFilename = self._task_path()
        self._registered = False

    def _prepare_file(self):
        from scheduler.artifact import ensure_output_parent

        ensure_output_parent(self.baseFilename)

    def _close_stream(self):
        if self.stream is not None:
            self.flush()
            self.stream.close()
            self.stream = None

    def _rollover_locked(self, files: SharedLogFiles) -> str | None:
        if not files.path.exists() or not files.path.stat().st_size:
            return None
        destination = files.reserve_part()
        os.replace(files.path, destination)
        return str(destination)

    def doRollover(self):
        self.acquire()
        try:
            self._prepare_file()
            with SharedLogFiles(self.baseFilename).locked() as files:
                self._close_stream()
                part = self._rollover_locked(files)
                with self._open():
                    pass
            self._registered = True
        finally:
            self.release()

    def emit(self, record):
        if self._retired:
            return
        try:
            if self._log_task != str(task_id):
                self._log_task = str(task_id)
                self.baseFilename = self._task_path()
                self._registered = False
            self._prepare_file()
            message = self.format(record) + self.terminator
            part = None
            with SharedLogFiles(self.baseFilename).locked() as files:
                size = files.path.stat().st_size if files.path.exists() else 0
                if self.maxBytes > 0 and size and size + len(message.encode(self.encoding or "utf-8")) >= self.maxBytes:
                    part = self._rollover_locked(files)
                try:
                    self.stream = self._open()
                    self.stream.write(message)
                    self.flush()
                finally:
                    # 持锁期间完成一整条记录，随后释放句柄；别的进程可安全轮转。
                    self._close_stream()
            self._registered = True
        except Exception:
            self.handleError(record)


def retire_log_directory(directory: Path) -> None:
    """任务目录清理前停止其中的文件日志，后续关停日志仍可输出到控制台。"""
    handler = _FILE_HANDLER
    if handler is None or not Path(handler.baseFilename).resolve().is_relative_to(directory):
        return
    handler.acquire()
    try:
        handler._retired = True
        handler._close_stream()
    finally:
        handler.release()


def _mark_handler_kind(handler: logging.Handler, kind: str) -> logging.Handler:
    setattr(handler, _DODORL_HANDLER_KIND, kind)
    return handler


def _resolve_log_level(level_name: str) -> int:
    return getattr(logging, str(level_name).upper(), logging.INFO)


def _shared_console_handler() -> logging.Handler:
    global _CONSOLE_HANDLER
    if _CONSOLE_HANDLER is None:
        _CONSOLE_HANDLER = _mark_handler_kind(logging.StreamHandler(sys.stdout), LogHandlerKind.CONSOLE.value)
        _CONSOLE_HANDLER.setFormatter(_CONSOLE_FORMATTER)
        _CONSOLE_HANDLER.addFilter(_TASK_ID_FILTER)
    return _CONSOLE_HANDLER


def _shared_file_handler() -> logging.Handler:
    global _FILE_HANDLER
    if _FILE_HANDLER is None:
        file_path = os.path.join(CONF.log_dir, CONF.log_file_name)
        _FILE_HANDLER = _mark_handler_kind(
            SafeRotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,
                # 分片由产物归档管理，不能在任务结束前循环覆盖早期日志。
                backupCount=0,
                encoding='utf-8',
                # TOML 日志配置可能在模块导入后才加载；避免默认 rl.log
                # 仅因初始化 handler 就被创建为空文件。
                delay=True,
            ),
            LogHandlerKind.FILE.value,
        )
        _FILE_HANDLER.setFormatter(_FILE_FORMATTER)
        _FILE_HANDLER.addFilter(_TASK_ID_FILTER)
    return _FILE_HANDLER


def _reset_shared_handlers() -> None:
    global _CONSOLE_HANDLER, _FILE_HANDLER
    old_handlers = (_CONSOLE_HANDLER, _FILE_HANDLER)
    _CONSOLE_HANDLER = None
    _FILE_HANDLER = None
    for handler in old_handlers:
        if handler is not None:
            handler.close()


def _enabled_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = []
    if LogHandlerKind.CONSOLE.value in CONF.enabled_handlers:
        handlers.append(_shared_console_handler())
    if LogHandlerKind.FILE.value in CONF.enabled_handlers:
        handlers.append(_shared_file_handler())
    return handlers


def _remove_dodorl_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _DODORL_HANDLER_KIND, None):
            logger.removeHandler(handler)


def _apply_config_to_logger(logger: logging.Logger) -> None:
    _remove_dodorl_handlers(logger)
    logger.setLevel(_resolve_log_level(CONF.log_level))
    logger.propagate = False
    for handler in _enabled_handlers():
        if handler not in logger.handlers:
            logger.addHandler(handler)


def configure_logger(config: LogConfig) -> None:
    """
    用 LogConfig 更新全局日志行为；TomlTrainingConfig 加载 [log_config] 后调用。

    已创建的 init_logger 实例会同步增删 Handler 与日志级别。
    """
    global CONF, _CONSOLE_FORMATTER, _FILE_FORMATTER
    CONF = config
    _CONSOLE_FORMATTER = logging.Formatter(CONF.console_log_format)
    _FILE_FORMATTER = logging.Formatter(CONF.file_log_format)
    _reset_shared_handlers()
    for logger_name in list(_MANAGED_LOGGER_NAMES):
        logger = logging.getLogger(logger_name)
        if isinstance(logger, logging.Logger):
            _apply_config_to_logger(logger)


class LoggerWrapper:
    """包装原始 logger 以支持带 tag 的 debug"""

    def __init__(self, logger: logging.Logger, config: LogConfig):
        self._logger = logger
        self._config = config
        self._stats = {}

    def debug(self, msg: str, tag: str = None):
        if tag and tag in self._config.log_debug_enable:
            self._logger.info(f"[debug-{tag}] {msg}")
        elif not tag:
            self._logger.debug(msg)

    def log_stat(self, key: str, value, t_env: int):
        """兼容EPyMARL learner的统计接口。"""
        self._stats[key] = value
        self._logger.debug(f"[stat] t={t_env} {key}={value}")

    def pop_stats(self):
        stats = dict(self._stats)
        self._stats.clear()
        return stats

    # 将其他方法代理到原始 logger
    def __getattr__(self, name):
        return getattr(self._logger, name)


def init_logger(name: str) -> LoggerWrapper:
    logger = logging.getLogger(name)
    _MANAGED_LOGGER_NAMES.add(name)
    _apply_config_to_logger(logger)
    return LoggerWrapper(logger, CONF)


_STARTUP_PROCESS_LOG = init_logger("startup_process")


def log_startup_process(
        role: str,
        config_file: str | os.PathLike[str],
        role_id: str | None = None,
) -> None:
    """输出可检索且不包含配置正文的统一角色启动日志。"""
    effective_role_id = role_id if role_id is not None else role
    set_logger_role_id(effective_role_id)
    message = (
        f"Startup process: role={role} "
        f"config_file={Path(config_file).expanduser().resolve()} "
        f"task_id={task_id}"
    )
    if role_id is not None:
        message += f" role_id={role_id}"
    _STARTUP_PROCESS_LOG.info(message)
