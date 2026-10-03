import typer

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
    strategy: list[RetrievalStrategy] = typer.Option(
        None, "--strategy", "-s", help="Strategy to evaluate (repeatable). Default: all available."
    ),
    mode: EvaluationMode = typer.Option(EvaluationMode.DETERMINISTIC, "--mode", "-m"),
    depth: int = typer.Option(100, "--depth", "-d", help="Retrieval depth (>= max recall cutoff)."),
    sample: int | None = typer.Option(None, "--sample", help="Evaluate only the first N queries."),    
)-> None:
    """Run the evaluation pipeline and write a report to var/eval/."""
    report = run_evaluation(
        mode,
        strategy or None, 
        depth=depth,
        sample=sample
    )
    typer.echo(report.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
