# Tasks: REER × DSPy × MLX Social Posting Pack

**Input**: Design documents from `/specs/001-reer_mlx/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```
1. Load plan.md from feature directory
   → Extract: Python 3.11+, DSPy, MLX, JSONL storage
2. Load optional design documents:
   → data-model.md: 6 entities (REER Trace, Post Candidate, Timeline Entry, Strategy Pattern, LM Provider Config, Performance Metrics)
   → contracts/: 3 schemas (traces.schema.json, candidate.schema.json, timeline.schema.json)
   → research.md: Technical decisions for DSPy patterns, MLX integration, provider routing
3. Generate tasks by category:
   → Setup: Python project, dependencies, linting
   → Tests: 3 contract tests, 5 integration tests
   → Core: 10 core modules, 6 plugin modules, 6 CLI scripts
   → Integration: JSONL storage, provider routing, DSPy pipeline
   → Polish: unit tests, performance optimization, documentation
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001-T040)
6. Validate task completeness
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: Repository root with core/, plugins/, dspy_program/, social/, scripts/, tests/
- Paths shown below follow the single project structure from plan.md

## Phase 3.1: Setup

- [ ] T001 Create project structure per implementation plan (core/, plugins/, dspy_program/, social/, scripts/, schemas/, tests/, tools/, docs/, data/)
- [ ] T002 Initialize Python 3.11+ project with pyproject.toml and requirements.txt
- [ ] T003 [P] Configure ruff and black for linting/formatting
- [ ] T004 [P] Create .env.example with provider API key templates
- [ ] T005 [P] Set up Git hooks for pre-commit testing

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (Schema Validation)

- [ ] T006 [P] Contract test for traces.schema.json in tests/contract/test_trace_schema.py
- [ ] T007 [P] Contract test for candidate.schema.json in tests/contract/test_candidate_schema.py
- [ ] T008 [P] Contract test for timeline.schema.json in tests/contract/test_timeline_schema.py

### Integration Tests (End-to-End Flows)

- [ ] T009 [P] Integration test for X analytics import → normalization in tests/integration/test_data_collection.py
- [ ] T010 [P] Integration test for REER strategy extraction in tests/integration/test_reer_mining.py
- [ ] T011 [P] Integration test for content generation pipeline in tests/integration/test_pipeline.py
- [ ] T012 [P] Integration test for provider switching (mlx::/dspy::) in tests/integration/test_provider_switching.py
- [ ] T013 [P] Integration test for GEPA optimization flow in tests/integration/test_gepa_tuning.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Core Modules (Trace & Strategy Management)

- [ ] T014 [P] REER trace store (append-only JSONL) in core/trace_store.py
- [ ] T015 [P] Trajectory synthesizer for strategy extraction in core/trajectory_synthesizer.py
- [ ] T016 [P] Candidate scorer with perplexity calculation in core/candidate_scorer.py
- [ ] T017 [P] DSPy GEPA runner in dspy_program/gepa_runner.py

### Plugin Modules (Provider Adapters)

- [ ] T018 [P] MLX language model adapter in plugins/mlx_lm.py
- [ ] T019 [P] DSPy language model adapter in plugins/dspy_lm.py
- [ ] T020 [P] Scoring heuristics module in plugins/heuristics.py
- [ ] T021 [P] LM registry for provider routing (mlx::/dspy::/dummy::) in plugins/lm_registry.py

### DSPy Pipeline Modules

- [ ] T022 [P] Main DSPy pipeline orchestrator in dspy_program/pipeline.py
- [ ] T023 [P] REER search wrapper module in dspy_program/reer_module.py
- [ ] T024 [P] KPI evaluator for performance metrics in dspy_program/evaluator.py

### Social Media Modules

- [ ] T025 [P] X analytics normalizer in social/collectors/x_normalize.py
- [ ] T026 [P] Social-specific DSPy modules (IdeateSignature, ComposeSignature) in social/dspy_modules.py
- [ ] T027 [P] KPI metrics calculator in social/kpis.py
- [ ] T028 [P] Content policies template in social/templates/policies.md

### CLI Scripts

- [ ] T029 Data collection CLI in scripts/social_collect.py
- [ ] T030 REER mining CLI in scripts/social_reer.py
- [ ] T031 GEPA tuning CLI in scripts/social_gepa.py (DSPy GEPA)
- [ ] T032 Pipeline execution CLI in scripts/social_run.py
- [ ] T033 Evaluation CLI in scripts/social_eval.py
- [ ] T034 MLX model management CLI in scripts/cli_mlx.py

### Schema Validation Tool

- [ ] T035 [P] JSON schema validator utility in tools/schema_check.py

## Phase 3.4: Integration

- [ ] T036 Wire TraceStore to REER mining pipeline
- [ ] T037 Connect LM registry to all CLI scripts
- [ ] T038 Integrate DSPy pipeline with provider routing
- [ ] T039 Set up rate limiting with exponential backoff
- [ ] T040 Configure structured logging across all modules

## Phase 3.5: Polish

- [ ] T041 [P] Unit tests for trace_store operations in tests/unit/test_trace_store.py
- [ ] T042 [P] Unit tests for candidate_scorer perplexity in tests/unit/test_scorer.py
- [ ] T043 [P] Unit tests for provider routing in tests/unit/test_lm_registry.py
- [ ] T044 Performance optimization for MLX inference (<50ms per score)
- [ ] T045 [P] Generate API documentation in docs/api.md
- [ ] T046 [P] Create REER methodology guide in docs/reer.md
- [ ] T047 [P] Write architecture overview in docs/architecture.md
- [ ] T048 Run quickstart.md validation end-to-end
- [ ] T049 Memory profiling and optimization
- [ ] T050 Final linting and code cleanup

## Dependencies

- Setup (T001-T005) blocks everything
- Tests (T006-T013) before implementation (T014-T035)
- Core modules (T014-T017) before plugins (T018-T021)
- Plugins before DSPy modules (T022-T024)
- All modules before CLI scripts (T029-T034)
- CLI scripts before integration (T036-T040)
- Everything before polish (T041-T050)

## Parallel Execution Examples

### Batch 1: All Contract Tests (after setup)

```bash
# Launch T006-T008 together:
Task: "Contract test for traces.schema.json in tests/contract/test_trace_schema.py"
Task: "Contract test for candidate.schema.json in tests/contract/test_candidate_schema.py"
Task: "Contract test for timeline.schema.json in tests/contract/test_timeline_schema.py"
```

### Batch 2: All Integration Tests

```bash
# Launch T009-T013 together:
Task: "Integration test for X analytics import in tests/integration/test_data_collection.py"
Task: "Integration test for REER strategy extraction in tests/integration/test_reer_mining.py"
Task: "Integration test for content generation pipeline in tests/integration/test_pipeline.py"
Task: "Integration test for provider switching in tests/integration/test_provider_switching.py"
Task: "Integration test for DSPy GEPA optimization in tests/integration/test_gepa_tuning.py"
```

### Batch 3: Core Modules

```bash
# Launch T014-T017 together:
Task: "REER trace store in core/trace_store.py"
Task: "Trajectory synthesizer in core/trajectory_synthesizer.py"
Task: "Candidate scorer in core/candidate_scorer.py"
Task: "DSPy GEPA runner in dspy_program/gepa_runner.py"
```

### Batch 4: Plugin Modules

```bash
# Launch T018-T021 together:
Task: "MLX language model adapter in plugins/mlx_lm.py"
Task: "DSPy language model adapter in plugins/dspy_lm.py"
Task: "Scoring heuristics in plugins/heuristics.py"
Task: "LM registry in plugins/lm_registry.py"
```

## Notes

- **TDD Enforcement**: Tests T006-T013 MUST fail before implementing T014-T035
- **Provider Switching**: Core feature - test thoroughly (T012)
- **MLX Priority**: Optimize for Apple Silicon performance
- **JSONL Format**: All data storage uses append-only JSONL
- **Schema Versioning**: Include version field in all schemas
- **Rate Limiting**: Critical for provider APIs (T039)

## Validation Checklist

*GATE: Verified before task execution*

- [X] All 3 contract schemas have corresponding tests (T006-T008)
- [X] All 6 entities from data-model.md have implementation tasks
- [X] All tests (T006-T013) come before implementation (T014-T035)
- [X] Parallel tasks ([P]) modify different files
- [X] Each task specifies exact file path
- [X] No [P] task modifies same file as another [P] task
- [X] CLI scripts depend on core/plugin modules
- [X] Integration tests cover all user stories from quickstart.md

## Estimated Timeline

- **Setup**: 1 hour (T001-T005)
- **Tests**: 2-3 hours (T006-T013)
- **Core Implementation**: 8-10 hours (T014-T035)
- **Integration**: 2-3 hours (T036-T040)
- **Polish**: 3-4 hours (T041-T050)
- **Total**: ~20-25 hours of focused development

---

*Based on Constitution v2.1.1 - Tasks follow TDD with RED-GREEN-Refactor cycle*
