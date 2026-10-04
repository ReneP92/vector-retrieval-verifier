from typing import Annotated

import typer

from app.config import QrelSplit
from app.domain.models import RetrievalStrategy
from common.telemetry import TelemetryLogger
from eval.eval import run_evaluation
from eval.models import EvaluationMode

TelemetryLogger.configure(level="INFO")
logger = TelemetryLogger(__name__)

app = typer.Typer(help="RAG retrieval evaluation pipeline.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """RAG retrieval evaluation pipeline."""


@app.command()
def run(
    strategy: Annotated[
        list[RetrievalStrategy] | None,
        typer.Option(
            "--strategy", "-s", help="Strategy to evaluate (repeatable). Default: all available."
        ),
    ] = None,
    mode: Annotated[EvaluationMode, typer.Option("--mode", "-m")] = EvaluationMode.QRELS,
    depth: Annotated[
        int, typer.Option("--depth", "-d", help="Retrieval depth (>= max recall cutoff).")
    ] = 100,
    sample: Annotated[
        int | None, typer.Option("--sample", help="Evaluate only the first N queries.")
    ] = None,
    split: Annotated[
        QrelSplit, typer.Option("--split", help="Qrels split. Tune on dev. Report on test.")
    ] = QrelSplit.TEST,
) -> None:
    """Run the evaluation pipeline, print the JSON report to stdout, and write it to var/eval/."""
    try:
        report = run_evaluation(mode, strategy or None, depth=depth, sample=sample, split=split)
    except (NotImplementedError, RuntimeError) as error:
        logger.error(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(report.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
