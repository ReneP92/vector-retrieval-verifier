# FiQA Dataset Layout

`scripts.download_fiqa` extracts the official BEIR FiQA-2018 archive to
`var/datasets/fiqa/`:

```text
var/datasets/fiqa/
  corpus.jsonl      57,638 answer passages
  queries.jsonl      6,648 questions (all splits)
  qrels/
    train.tsv        5,500 queries, 14,166 judgments
    dev.tsv            500 queries,  1,238 judgments
    test.tsv           648 queries,  1,706 judgments
```

## How the Files Link

The files are linked by ID. Each qrels row is a join between one query and one
relevant passage:

```text
queries.jsonl          qrels/test.tsv                 corpus.jsonl
{"_id": "8", ...}  <-  8    566392    1   ->   {"_id": "566392", ...}
                       8    65404     1   ->   {"_id": "65404", ...}
```

- `corpus.jsonl`: one passage per line with `_id`, `title` (always empty in
  FiQA) and `text`.
- `queries.jsonl`: one question per line with `_id` and `text`.
- `qrels/*.tsv`: a header row, then `query-id`, `corpus-id` and `score`
  separated by tabs.

Query IDs and corpus IDs are separate namespaces. Query `8` and passage `8`
have nothing to do with each other.

## Splits

`queries.jsonl` holds the questions for every split, and the corpus is shared
across all of them. A query belongs to a split only because it appears in that
split's qrels file. Evaluation loads the test qrels (`Settings.qrels_path`) and
keeps only the queries they reference.

- `train`: for training models. Unused, since nothing in the pipeline is
  trained.
- `dev`: for tuning settings such as `rrf_k` and `candidate_k`.
- `test`: for reported scores only. Never tune against it.

## Relevance

- Every score is `1`, so relevance is binary.
- Test queries have about 2.6 relevant passages each. The labels come from the
  answers in the original FiQA question threads.
- Any query-passage pair without a qrels row counts as not relevant, even
  though nobody reviewed it. Retrieved passages that answer the question but
  were never labeled therefore lower the scores.
- 38 passages have no title or text. The indexes exclude them, and one of them
  has a positive test judgment (see the [README](../README.md)).
