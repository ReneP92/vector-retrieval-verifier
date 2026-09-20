import logging
from collections.abc import Mapping, MutableMapping
from typing import Any, Self, TextIO

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class TelemetryLogger(logging.LoggerAdapter):
    """Repository-wide logger with optional structured context."""

    def __init__(
        self,
        name: str,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(logging.getLogger(name), dict(context or {}))

    @classmethod
    def configure(
        cls,
        level: int | str = logging.INFO,
        *,
        log_format: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
        stream: TextIO | None = None,
        force: bool = False,
    ) -> None:
        """Configure the root logger once at an application entry point."""
        options: dict[str, object] = {
            "level": level,
            "format": log_format,
            "datefmt": date_format,
            "force": force,
        }
        if stream is not None:
            options["stream"] = stream
        logging.basicConfig(**options)

    def bind(self, **context: object) -> Self:
        """Return a logger carrying this context on every record."""
        return type(self)(self.logger.name, {**self.extra, **context})

    def process(
        self,
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        call_context = kwargs.get("extra")
        if call_context is not None and not isinstance(call_context, Mapping):
            raise TypeError("logging extra must be a mapping")
        kwargs["extra"] = {**self.extra, **(call_context or {})}
        return msg, kwargs
