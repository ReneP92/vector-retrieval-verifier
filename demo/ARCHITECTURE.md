# Demo Architecture

The demo is intentionally structured like a production retrieval service while
using local persisted indexes. Interactive RAG and future offline evaluation
must execute the same `RetrievalPipeline`.

```text
FastAPI or future BEIR replay job
                |
                v
        RetrievalPipeline
          /           \
       BM25           Dense
    sparse ranking  vector ranking
          \           /
           \         /
              RRF
               |
        optional Cohere
           reranker
               |
               v
      ranked SearchHit objects
          /           \
 retrieval trace   answer generator
```

BM25 and dense strategies can return their rankings directly. Hybrid strategies
send both rankings through RRF, and only `rrf_rerank` sends the fused candidates
through Cohere.

## Boundaries

- `domain/` defines provider-neutral documents, hits, traces, and protocols.
- `services/` owns retrieval orchestration and RAG generation policy.
- `adapters/` integrates BEIR, BM25S, OpenAI, Cohere, and persisted artifacts.
- `web/` translates HTTP and form requests into application-service calls.
- `scripts/` performs explicit, repeatable dataset and index preparation.

Qrels never enter the online retrieval path. When evaluation is added, a replay
job will read FiQA queries, call `RetrievalPipeline.search`, and write TREC run
files plus an experiment manifest. `rag_retrieval_evaluator` will compare those
immutable runs with qrels outside this application.

## Production Mapping

| Demo component | Production replacement |
| --- | --- |
| BEIR JSONL repository | Versioned corpus service or object-store snapshot |
| BM25S index | OpenSearch or Elasticsearch adapter |
| Memory-mapped dense matrix | Vector database adapter |
| File response cache | Shared content-addressed cache |
| In-process FastAPI app | Horizontally scaled query service |
| Local replay process | Scheduled or CI evaluation worker |

The application service and result contracts should remain unchanged when an
adapter is replaced.

## Reproducibility

Every retrieval result records the corpus SHA-256, strategy, stage timings, and
retrieval parameters. Persisted indexes include manifests tying them to the
corpus and model. Future run manifests should additionally record source commit,
provider model versions, query split, random seed, and complete configuration.

Index manifests also record the effective-corpus policy. FiQA's whitespace-only
passages are excluded from every retrieval index while remaining unchanged in
the source snapshot. Positive qrels that reference excluded passages are
reported as dataset-quality warnings and must be surfaced by future evaluation
reports.
