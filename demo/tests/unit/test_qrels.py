from pathlib import Path

from app.adapters.datasets.qrels import find_excluded_positive_qrels


def test_reports_positive_qrels_for_excluded_documents(tmp_path: Path) -> None:
    qrels = tmp_path / "test.tsv"
    qrels.write_text(
        "query-id\tcorpus-id\tscore\nq1\tempty-positive\t1\nq2\tempty-negative\t0\nq3\tusable\t2\n"
    )

    matches = find_excluded_positive_qrels(
        qrels,
        ["empty-positive", "empty-negative"],
    )

    assert len(matches) == 1
    assert matches[0].query_id == "q1"
    assert matches[0].document_id == "empty-positive"
