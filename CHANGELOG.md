# Changelog

## [0.1.0] - 2026-03-09

### Added

- 7 MCP tools: `learn`, `recall`, `save_perception`, `forget`, `update`, `start_session`, `end_session`
- Python API: `from robotmem import learn, recall, save_perception, ...`
- BM25 + vector hybrid search with RRF fusion
- Structured context filtering (`context_filter`)
- Spatial nearest-neighbor sorting (`spatial_sort`)
- Session-based memory consolidation with Jaccard dedup
- Proactive recall on `end_session`
- CJK support via jieba tokenizer
- Web management UI (`robotmem web`)
- Health check endpoint (`/api/doctor`)
- FetchPush simulation experiment (+25% success rate)
- Cross-environment transfer experiment (FetchPush -> FetchSlide)
- 576 tests, 87% coverage
