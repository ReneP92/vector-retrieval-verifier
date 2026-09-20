import io
import logging

from common.telemetry import TelemetryLogger


def test_configures_and_writes_a_named_log() -> None:
    output = io.StringIO()
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    try:
        TelemetryLogger.configure(
            level="INFO",
            log_format="%(levelname)s %(name)s %(message)s",
            stream=output,
            force=True,
        )

        TelemetryLogger("app.retrieval").info("search completed")

        assert output.getvalue() == "INFO app.retrieval search completed\n"
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)


def test_bound_and_per_call_context_are_added_to_log_records() -> None:
    records: list[logging.LogRecord] = []

    class RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = TelemetryLogger("tests.telemetry.context").bind(request_id="request-1")
    logger.logger.handlers = [RecordingHandler()]
    logger.logger.propagate = False
    logger.logger.setLevel(logging.INFO)

    logger.info("search completed", extra={"duration_ms": 12})

    assert vars(records[0])["request_id"] == "request-1"
    assert vars(records[0])["duration_ms"] == 12
