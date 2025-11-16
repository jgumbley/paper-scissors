from __future__ import annotations

import logging
import os
from typing import Any

from rich._null_file import NullFile
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import Traceback
from rich.traceback import install as install_rich_traceback

console = Console()
install_rich_traceback(console=console, width=120, show_locals=False)

ACTION_LEVEL = logging.INFO + 5
logging.addLevelName(ACTION_LEVEL, "ACTION")


def _logger_action(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(ACTION_LEVEL):
        self._log(ACTION_LEVEL, message, args, **kwargs)


logging.Logger.action = _logger_action  # type: ignore[attr-defined]


class PawlRichHandler(RichHandler):
    """Custom handler that tones down INFO and highlights ACTION events."""

    INFO_LEVEL_STYLE = "grey70"
    INFO_MESSAGE_STYLE = "grey70"
    ACTION_LEVEL_STYLE = "bold bright_white on grey11"
    ACTION_ROW_STYLE = "bright_white on grey11"

    def get_level_text(self, record: logging.LogRecord) -> Text:  # type: ignore[override]
        level_name = record.levelname
        level_key = level_name.lower()
        if level_key == "info":
            style = self.INFO_LEVEL_STYLE
        elif level_key == "action":
            style = self.ACTION_LEVEL_STYLE
        else:
            style = f"logging.level.{level_key}"
        return Text(level_name.ljust(8), style=style)

    def render_message(self, record: logging.LogRecord, message: str):
        renderable = super().render_message(record, message)
        if isinstance(renderable, Text) and record.levelname.lower() == "info":
            renderable.stylize(self.INFO_MESSAGE_STYLE)
        return renderable

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        message = self.format(record)
        traceback = None
        if self.rich_tracebacks and record.exc_info and record.exc_info != (None, None, None):
            exc_type, exc_value, exc_traceback = record.exc_info
            assert exc_type is not None
            assert exc_value is not None
            traceback = Traceback.from_exception(
                exc_type,
                exc_value,
                exc_traceback,
                width=self.tracebacks_width,
                code_width=self.tracebacks_code_width,
                extra_lines=self.tracebacks_extra_lines,
                theme=self.tracebacks_theme,
                word_wrap=self.tracebacks_word_wrap,
                show_locals=self.tracebacks_show_locals,
                locals_max_length=self.locals_max_length,
                locals_max_string=self.locals_max_string,
                suppress=self.tracebacks_suppress,
                max_frames=self.tracebacks_max_frames,
            )
            message = record.getMessage()
            if self.formatter:
                record.message = record.getMessage()
                formatter = self.formatter
                if hasattr(formatter, "usesTime") and formatter.usesTime():
                    record.asctime = formatter.formatTime(record, formatter.datefmt)
                message = formatter.formatMessage(record)

        message_renderable = self.render_message(record, message)
        log_renderable = self.render(
            record=record,
            traceback=traceback,
            message_renderable=message_renderable,
        )
        if isinstance(self.console.file, NullFile):
            self.handleError(record)
            return

        row_style = self.ACTION_ROW_STYLE if record.levelname.lower() == "action" else None
        try:
            self.console.print(log_renderable, style=row_style)
        except Exception:
            self.handleError(record)


def setup_logging() -> None:
    if getattr(setup_logging, "_configured", False):
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = PawlRichHandler(
        console=console,
        show_time=True,
        rich_tracebacks=True,
        markup=True,
        show_level=True,
        show_path=False,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )
    setattr(setup_logging, "_configured", True)
