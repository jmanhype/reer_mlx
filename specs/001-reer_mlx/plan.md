# Implementation Plan: REER × DSPy × MLX Social Posting Pack

**Branch**: `001-build-the-reer` | **Date**: 2025-01-10 | **Spec**: [`/specs/001-build-the-reer/spec.md`](./spec.md)
**Input**: Feature specification from `/specs/001-build-the-reer/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Build a repeatable, auditable system to learn posting strategies from X (Twitter) analytics and generate optimized content using REER (Reverse-Engineered Reasoning) methodology, with flexible LM backend routing between cloud providers (DSPy) and local MLX processing for cost optimization.

## Technical Context
**Language/Version**: Python 3.11+  
**Primary Dependencies**: DSPy (LM orchestration), MLX + mlx-lm (local inference), JSON Schema (draft-07)  
**Storage**: JSONL files (append-only traces), local filesystem  
**Testing**: pytest, ruff/black for linting  
**Target Platform**: Linux/macOS, Apple Silicon preferred for MLX  
**Project Type**: single (CLI-based tool with libraries)  
**Performance Goals**: ≥15% uplift in impressions-per-follower, ≥50% cost reduction with MLX  
**Constraints**: ≥90% schema adherence, zero code edits for provider switching  
**Scale/Scope**: Processing thousands of posts, supporting 4+ LM providers

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (single Python package with CLI tools)
- Using framework directly? Yes (DSPy, MLX used directly)
- Single data model? Yes (JSONL traces, schemas)
- Avoiding patterns? Yes (no unnecessary abstractions)

**Architecture**:
- EVERY feature as library? Yes (core/, plugins/, social/ modules)
- Libraries listed: 
  - `core/trace_store`: Append-only REER trace persistence
  - `plugins/lm_registry`: LM provider routing
  - `dspy_program/pipeline`: DSPy orchestration
  - `social/collectors`: Analytics normalization
- CLI per library: All exposed via scripts/ directory
- Library docs: Will use docstrings + docs/ markdown

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes
- Git commits show tests before implementation? Yes
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes (actual LM providers, real JSONL files)
- Integration tests for: new libraries, contract changes, shared schemas? Yes
- FORBIDDEN: Implementation before test, skipping RED phase

**Observability**:
- Structured logging included? Yes (via Python logging)
- Frontend logs → backend? N/A (CLI-only)
- Error context sufficient? Yes (full stack traces, LM responses)

**Versioning**:
- Version number assigned? Yes (via pyproject.toml)
- BUILD increments on every change? Yes (CI automation)
- Breaking changes handled? Yes (schema versioning, migration scripts)

## Project Structure

### Documentation (this feature)
```
specs/001-build-the-reer/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT) - Selected for this CLI tool
core/
├── trace_store.py           # Append-only REER trace storage
├── trajectory_synthesizer.py # Strategy synthesis from traces
├── candidate_scorer.py      # Score generated content
└── trainer.py              # GEPA optimization

plugins/
├── mlx_lm.py               # MLX backend for local inference
├── dspy_lm.py              # DSPy.LM adapter for cloud providers
├── heuristics.py           # Scoring heuristics
└── lm_registry.py          # Route mlx::/dspy::/dummy prefixes

dspy_program/
├── pipeline.py             # Main DSPy pipeline
├── reer_module.py          # REER search wrapper
└── evaluator.py            # KPI evaluation

social/
├── collectors/
│   └── x_normalize.py      # X analytics normalization
├── dspy_modules.py         # Social-specific DSPy modules
├── templates/
│   └── policies.md         # Content policies
└── kpis.py                # Performance metrics

scripts/
├── social_collect.py       # Data collection CLI
├── social_reer.py          # REER mining CLI
├── social_gepa.py          # GEPA tuning CLI
├── social_run.py           # Pipeline execution CLI
├── social_eval.py          # Evaluation CLI
└── cli_mlx.py             # MLX CLI wrapper

schemas/
├── candidate.schema.json   # Post candidate schema
├── timeline.schema.json    # Timeline entry schema
└── traces.schema.json      # REER trace schema

tests/
├── contract/              # Schema validation tests
├── integration/           # End-to-end pipeline tests
└── unit/                  # Module unit tests

tools/
└── schema_check.py        # JSON schema validator

docs/
├── reer.md               # REER methodology documentation
├── architecture.md       # System architecture
└── prd.md               # Product requirements

data/
├── social/
│   └── normalized.jsonl  # Normalized X data
└── traces/
    └── traces.jsonl      # REER traces
```

**Structure Decision**: Option 1 (Single project) - appropriate for CLI-based tool with libraries

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - DSPy best practices for pipeline composition
   - MLX token-level logprobs extraction methods
   - Perplexity score normalization across providers
   - DSPy GEPA configuration for social content (budget, reflection LM)

2. **Generate and dispatch research agents**:
   ```
   Task: "Research DSPy pipeline patterns for content generation"
   Task: "Find MLX logprobs extraction methods for perplexity scoring"
   Task: "Research logprob normalization across Together/OpenAI/Anthropic"
   Task: "Find DSPy GEPA optimization strategies for social media content"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - REER Trace: {id, timestamp, seed_params, score, metadata}
   - Post Candidate: {id, content, features, score, provider}
   - Timeline Entry: {id, topic, drafts[], scheduled_time}
   - Strategy Pattern: {id, pattern_type, parameters, performance}

2. **Generate API contracts** from functional requirements:
   - JSON schemas in `/contracts/` for traces, candidates, timelines
   - CLI argument specifications for each script
   - LM provider interface contracts

3. **Generate contract tests** from contracts:
   - Schema validation tests for each JSON schema
   - CLI argument parsing tests
   - Provider switching tests

4. **Extract test scenarios** from user stories:
   - Analytics import → REER mining → content generation flow
   - Provider switching scenario
   - Cost comparison scenario

5. **Update agent file incrementally**:
   - Add REER × DSPy × MLX context to CLAUDE.md
   - Include key schemas and CLI commands
   - Document provider switching patterns

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs:
  - Each schema → validation test task [P]
  - Each core module → implementation task
  - Each plugin → adapter implementation task [P]
  - Each script → CLI implementation task
  - Integration tests for complete flows

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: 
  1. Schemas and validation
  2. Core modules (trace_store first)
  3. Plugins (lm_registry, then adapters)
  4. DSPy pipeline modules
  5. CLI scripts
  6. Integration tests

**Estimated Output**: 30-35 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*No violations - design follows constitutional principles*

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
