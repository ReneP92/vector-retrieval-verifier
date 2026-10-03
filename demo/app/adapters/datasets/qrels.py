import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExcludedPositiveQrel:
    query_id: str
    document_id: str
    score: int


def find_excluded_positive_qrels(
    qrels_path: Path,
    excluded_document_ids: list[str],
) -> list[ExcludedPositiveQrel]:
    if not qrels_path.is_file() or not excluded_document_ids:
        return []

    excluded = set(excluded_document_ids)
    matches: list[ExcludedPositiveQrel] = []
    with qrels_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            document_id = str(row["corpus-id"])
            score = int(row["score"])
            if document_id in excluded and score > 0:
                matches.append(
                    ExcludedPositiveQrel(
                        query_id=str(row["query-id"]),
                        document_id=document_id,
                        score=score,
                    )
                )
    return matches


def load_qrels(qrels_path: Path) -> dict[str, set[str]]:
    """Map each quyery id to the set of its positively-judged document ids."""
    relevant: dict[str, set[str]] = defaultdict(set)
    with qrels_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if int(row["score"]) > 0:
                relevant[str(row["query-id"])].add(str(row["corpus-id"]))
    return dict(relevant)
