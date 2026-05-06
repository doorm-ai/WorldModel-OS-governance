# DOORM WorldModel-OS: A Governance-First Architecture for Auditable Agentic Reasoning — Hybrid State-Space, Structurally-Primed Discrete Anchors, and Anti-Scale Training Discipline

**v0.2 · 2026-05-06**
**Status:** Position & system paper, pre-empirical. No empirical claims are made; falsification criteria are specified in §11.
**License:** CC BY 4.0

---

## Authors

**Tong Li**¹\*  ·  **Tong Tong**¹  ·  **Wu Lihua**²

¹ DOORM AI PTE. LTD. (UEN 202441729W), Singapore
² Independent Researcher
\* Corresponding author: service@doorm.ai

*Note on author order.* The author order on this paper reflects **relative contribution to the work**, not academic seniority. Affiliations are listed independently of contribution rank.

---

> **Open / closed disclosure notice.** This paper publishes the *architecture*, *methodology*, *contracts*, and *operational positioning* of the DOORM WorldModel-OS platform and its companion Finetune Workbench. These correspond to our L1 (platform skeleton) and L2 (seven cognitive increments, training methodology) tiers, which are AGPL-3.0-or-later open source or defensively published here. **We do not disclose**: (a) production LoRA weights; (b) clinical-grade adapter variants; (c) partner-specific medical data; (d) commercial licensing terms; (e) secret material of our device-authentication deployment. These constitute L3 and are held under separate commercial / clinical governance.

---

## Abstract

We present **WorldModel-OS**, a hybrid local / cloud operating system for auditable agentic world models, and its separate training workbench **Finetune Workbench**. The design rejects two dominant assumptions in contemporary large-model practice: (i) that capability is monotone in parameter count ("scale is all you need"), and (ii) that alignment can be bolted onto a post-training checkpoint. Instead, we factor the system into seven cognitive increments, grounded in a **discrete state space** built from an 8-mode base prior, a 64-state composite obtained by Cartesian self-pairing of the base modes, and a 6-position ordinal extension yielding 384 anchor points; we enforce safety through a four-level rollback contract embedded in every inference trace. Training uses a five-class grounded-citation data pipeline with three hard gates (citation, consistency, compliance), a 3%-degradation bucketed-evaluation rollback protocol, and an immutable `adapter_version` handshake between platform and workbench. We motivate the design through medical-adjacent applications — Traditional Chinese Medicine (TCM) syndrome differentiation and rehabilitation robotics — where per-decision auditability and refusal-by-default beat benchmark-scale gains. We outline deployment as a hybrid client (local LLM via Ollama plus cloud-side knowledge, safety, and memory microservices) bound by ED25519 device identity. **No empirical results are claimed in this paper**; we specify the reproducibility envelope, three hard release gates, an explicit non-coverage list, cross-domain migration cost estimates, and falsification criteria against which our positions can be refuted. Section §11 contains the open / closed disclosure boundary.

**Keywords.** world model; LoRA fine-tuning; auditable inference; clinical safety; discrete state space; structural prior; anti-scale training; Traditional Chinese Medicine decision support; reproducibility; defensive disclosure; falsifiability.

---

## 1. Introduction

The post-2022 large-model industry has converged on a single hill-climbing strategy: more tokens, more parameters, more compute, broader RLHF. Within that frame, alternatives are framed as distractions. We argue this frame is incomplete for a specific class of deployments — safety-critical, audit-bearing, culturally-situated, and compute-constrained — and that a small but disciplined system can out-deliver scale on problems the industry is actively failing to solve.

This paper documents the architectural commitments of DOORM WorldModel-OS, a system we are building to occupy exactly that gap. We take three positions, the first of which decomposes into four sub-claims (4-anti).

**P1 (Anti-scale thesis, decomposed).** For a class of problems defined by *reproducibility on consumer GPUs*, *per-decision auditability*, and *cultural / linguistic particularity* (here: classical Chinese health and rehabilitation traditions), improvements do not primarily come from more parameters. We commit to four operational refusals, jointly:

1. **Anti-token.** We do not treat life and medicine as flat token / symbol sequences to be fitted; we build *state–mechanism–feedback* structure first.
2. **Anti-scale.** "Bigger" is not the only answer; under correct structure, smaller-scale models obtain stronger interpretability and transferability.
3. **Anti-stacking.** We do not trade data and parameters for capability; we *organize* complexity through structural modeling.
4. **Anti-compute-monopoly.** Critical capabilities should not depend on a few institutions' supercomputers; we ship a reproducible, deployable, broadly accessible health-intelligence stack.

These are not anti-science; they are anti "pure-engineering-stack" as the only allowed ladder.

**P2 (Two-project separation).** We split the system into two peer projects connected by a narrow, contractual interface: a platform `WorldModel-OS` that owns inference, state, safety, audit, and knowledge; and a training workbench `Finetune Workbench` that produces LoRA adapters and *never* imports the platform's code. The only cross-project handshake is an `adapter_version` string conforming to a frozen regex.

**P3 (Defensive disclosure with commercial retention).** The platform, the training methodology, and the cognitive-increment framework are disclosed (AGPL-3.0 + this paper). Production weights fine-tuned on regulated medical corpora are retained commercially and served cloud-side only, never distributed as quantized open weights. This paper delineates which claims are L1 / L2 (public) and which are L3 (held).

### 1.1 Scope and Non-Coverage (Boundary)

To pre-empt the most common review question — "what does this system *not* claim universality over?" — we make scope explicit before any technical content. The following are **out of scope** for v0.1 / v0.2 and form a hard non-coverage list (cross-referenced to internal strategy and threat-model documents):

- **No surgical robotics; no last-mile delivery / logistics robotics.** Our application focus is TCM clinical decision support and rehabilitation, not autonomous physical intervention.
- **No claim of artificial general intelligence (AGI) and no positioning as a "GPT alternative".** We are a *disciplined subset*, not a universal substitute.
- **No fortune-telling or astrological interpretation** (see §9.1 hard rule).
- **No closed-form mathematical disclosure of all seven cognitive increments before defensive publication.** This paper discloses the framework and the engineering anchors; the full mathematical form of each increment is defensively published incrementally per quarterly schedule.
- **No clinical data outside the controlled training environment.** Production adapter weights trained on regulated corpora are retained and served cloud-side only.
- **No domains other than health (TCM + rehabilitation + pulse) for v0.2.** Government, finance, meteorology, autonomous driving, environmental, and IoT applications are not present in this stack and are not claimed. Cross-domain extension is deferred (see §13.2 migration recipe and the commercial governance roadmap).

These exclusions are deliberate. They define the *useful sense* in which our universality claim is made: high abstraction, narrow first-application, with a published recipe for later extension (§13).

We make four contributions:

1. A **seven-cognitive-increment** decomposition (§5) that is neither scale-extrapolation nor end-to-end perception-grounding; we claim it is a useful axis of progress for audit-bearing agentic systems.
2. A **discrete state space with structurally-primed anchors** (§4): an 8-mode base prior, a 64 = 8×8 paired-state composite, and a 384 = 64×6 ordinal anchor scheme, equipped with a learnable transition prior matrix initialized from a domain-relevant ordering rather than uniform.
3. A **five-class, three-gate training pipeline** (§6) that emits grounded citations `[corpus:Lx-y]` in every answer and blocks samples at the citation / consistency / compliance boundary *before* they enter the training pool.
4. A **four-level (L0–L4) safety and audit contract** (§7) attached to every inference span, with an automatic bucketed-evaluation rollback rule (any bucket × any metric degrades > 3% ⇒ previous adapter is retained).

We are not the first to argue for auditable AI, nor for structural priors, nor for culturally-grounded NLP. What, to our knowledge, is novel is the *combination*: a commercial-grade system that fuses a culturally-grounded combinatorial structure with modern parameter-efficient fine-tuning, under an explicit anti-scale positioning, with a contractual separation between platform and workbench.

> **Methodological note.** The bet that a rigorously formalized version of a tradition-grounded structure can outperform end-to-end approaches in a niche domain is, methodologically, structurally analogous to the Belousov–Zhabotinsky case in chemistry, where a phenomenon previously dismissed as anomalous was eventually accepted via mechanistic reformulation (FKN mechanism, Oregonator model). We do not depend on this analogy for any technical claim; we acknowledge it only as motivation for entertaining a non-mainstream methodological prior.

---

## 2. Related Work

**Parameter-efficient fine-tuning.** Low-Rank Adaptation (LoRA) [Hu et al., 2021] and QLoRA [Dettmers et al., 2023] enable adaptation of billion-parameter models with a small fraction of trainable parameters. We build on QLoRA with 4-bit NF4 quantization, with the v0 reference base set to Llama-3.2-3B-Instruct [Meta AI, 2024], following the Hugging Face PEFT ecosystem [Mangrulkar et al., 2022]. **The workbench is base-model agnostic**: any decoder-class LLM that respects the workbench's data-format and `adapter_version` handshake may serve as the base — Llama-3.2-3B-Instruct is the v0 reference, not a binding requirement. Our technical novelty is *not* in the optimizer or the choice of base. It is in what we gate and what we retain as a contract across versions.

**World models in RL and planning.** From Ha and Schmidhuber (2018) through Dreamer-v3 (Hafner et al., 2023), the "world model" label typically denotes a latent dynamics predictor trained on ego-centric trajectories. We repurpose the label at a higher level of abstraction: a *compositional state space* that mediates between inputs and skill dispatch, and whose dynamics are *audited* rather than purely learned end-to-end.

**Auditable and trustworthy AI.** Model cards [Mitchell et al., 2019], datasheets [Gebru et al., 2021], and deployment ledgers (e.g., Hugging Face's model registry) treat audit as a release artifact. We go further: audit is a *runtime contract* — every inference span carries `trace_id`, input / output hashes, `adapter_version`, and a monotone `safety_level` field, and every trace is replayable from the append-only log.

**Computational humanities and structural priors.** A substantial literature applies formal methods to classical Chinese textual corpora, ranging from symbolic combinatorics to recent neural NLP for classical Chinese. We do not extend this literature scholastically. We use the *combinatorial skeleton* of one such corpus as an *engineering prior* — its 8-mode base × 8-mode pairing × 6-line extension yields 384 discrete anchor points which serve as citation anchors and state-space coordinates. The historical or interpretive merits of the corpus's commentary are orthogonal to whether the structure is *useful* as a prior in our audit-bearing setting.

**Retrieval-augmented generation (RAG).** We use retrieval sparingly. Our `HexaRAG` component now ships vector-embedding retrieval keyed on `BGE-zh-v1.5` (`RETRIEVER_VERSION = "v1-bge-zh-2026-04-26"`) with a graceful fallback to a deterministic blake2b stream when `sentence-transformers` is unavailable — the API contract is preserved either way; only semantic quality degrades. We distinguish between *retrieval for grounding* (citation coverage in training, §6) and *retrieval for inference-time evidence* (cloud-side knowledge service). Both use the same 384-anchor index as the anchor vocabulary.

**Medical AI safety.** Within clinical and near-clinical deployments, refusal-by-default and evidence-linked generation are near-consensus requirements (FDA, 2024 draft guidance on AI/ML-enabled medical devices; EU AI Act high-risk provisions). We adopt both explicitly as training gates (§6.3 compliance gate) and runtime safety levels (§7.1). Unlike purely prompt-level alignment, our contract is enforced across three tiers: data, runtime, and audit.

---

## 3. System Architecture

### 3.1 Two-project separation

We maintain two peer repositories:

- **`WorldModel-OS`** (platform). Seven microservices, one gateway, a shared `core` of state encoders and audit primitives, two domain modules (health / medical-adjacent, and a planned robotics module).
- **`Finetune Workbench`** (workbench). Data-generation pipeline, three quality gates, a training runner (ECS GPU only), and a bucketed evaluator. **No import** of the platform's Python modules is permitted; CI enforces this with a grep-based lint.

**Hard boundaries (enforced by lint tests).**

1. The platform must not `import finetune.*`.
2. The platform must run without the workbench present (base model inference is the minimum).
3. The source corpora (two classical texts the user holds IP on) live in the platform; the workbench consumes them via symbolic links only.
4. Pre-training is a **non-goal** (PRD-01 §2.2). All adaptation is external, via LoRA.

**The only handshake.** A single frozen regex:

```
ADAPTER_VERSION_PATTERN = r"^v\d+-[0-9a-f]{8}-[0-9a-f]{8}$"
```

Both sides have a byte-level lint test pinning this literal. Format: `v{N}-{data_hash[:8]}-{code_hash[:8]}` where `data_hash` is the SHA-256 prefix of the training JSONL and `code_hash` is `git rev-parse --short HEAD`.

This one string binds every inference trace to the exact dataset and code that produced the adapter — sufficient, in conjunction with open methodology and an auditable log, for third-party reproduction within the release gate (§11).

### 3.2 Microservices

The platform exposes seven services behind a FastAPI gateway:

| Service | Primary contract | Sprint-2 realism |
|---|---|---|
| `perception` | `POST /embed` (text → d=256 vector) | Deterministic pseudo-embedding (blake2b stream); swap to BGE-zh or Jina-v3 scheduled. |
| `world_model` | `POST /observe` → state; `/step`, `/rollout` | Real: 8×8 transition prior, session state machine. |
| `planner` | `POST /plan` (skill graph DAG) | Real: six built-in skills, topological sort. |
| `executor` | `POST /dispatch`, `GET /history/{id}` | Real: safety-ceiling truncation. |
| `event_bus` | `publish/consume`, bounded deque (10k/topic) | Real: in-process pub/sub. |
| `safety` | `POST /assess` (rule engine) | Real: 4-level rollback, 28 rule cases. |
| `memory` | structural / contextual / episodic (PRD-01 §6); a **strategic** mid-tier (migratable action-experience) is scheduled for v0.2 | Real: three-layer DTO; strategic tier in design. |

Two additional API surfaces sit *beside* the microservice layer:

- `POST /state/anchor_infer` — exposes the 384-anchor index directly (the public API uses `state.anchor_id`; internal code naming is documented in the repository README).
- `POST /health/encode` — Mind Encoder (text + PHQ-9 + HRV / HR → 8-mode probability distribution).

Critical to this paper's auditability claims: `GET /audit/snapshot` — a read API over the append-only trace log, filterable by `operation`, `adapter_version`, and `safety_level`.

### 3.3 Solver and the cross-project seam

A dedicated `solver` microservice hot-swaps LoRA adapters via an LRU cache (capacity configurable, default 4). Requests carry `X-Adapter-Version`; unknown versions fall back to the base model rather than silently misrouting. This is the only service in the platform that loads workbench outputs, and it does so via a filesystem volume — never a Python import.

---

## 4. State Space: Structurally-Primed Discrete Anchors

### 4.1 Motivation

Language models infer in a continuous latent space whose coordinates are not meaningful to any human practitioner. This is fine for scale-limited chat, and catastrophic for auditable medical decision support: "the model said so" is not a defensible warrant. We need an *anchored* state space — a finite, human-readable, semantically contentful set of coordinates — such that every inference can be tied back to at least one anchor.

Concretely, we represent state as a five-tuple — **continuous** (sensor / numerical / waveform), **discrete** (categorical / structural-anchor index), **relation graph** (entity-relation edges), **evidence genealogy** (source citations with version pointers), and **uncertainty** (per-coordinate confidence). None of these five projects cleanly into a single 1D token stream; flattening them is what produces the "language hallucination" failure mode in physical-world deployments.

A culturally-grounded combinatorial corpus from the Chinese tradition gives us such an anchored space at near-zero embedding cost.

### 4.2 The 8 base modes and the 64 paired-state composite

Let the base modes be ordered binary triples over a binary primitive (denoted `{a, b}` with semantics dual-mode), yielding |2³| = 8 base states. We label them as 8 abstract indices `m ∈ {0, ..., 7}` (denoted **BM₀ … BM₇**), each carrying a Five-Phase polarity tag (metal / wood / water / fire / earth, with yang/yin polarity where applicable) for downstream clinical mapping; the abstract index `m` is the canonical mathematical handle, and the Five-Phase tag is purely a clinical-domain readout. The 64-state paired composite is the Cartesian self-pairing 8 × 8 → 64, and admits two canonical orderings: a **canonical (intrinsic)** ordering and a **deployed (applied)** ordering, related by a fixed permutation. Both orderings are realized as bijective encodings, with a property-based test (Python `hypothesis`) over the full 64-element domain pinning down their equivalence classes.

### 4.3 The 384-anchor ordinal scheme

Each of the 64 paired states carries 6 ordinal positions, indexed bottom-to-top from 1 to 6. Total discrete anchor points: 64 × 6 = 384. We treat each anchor as:

- A **state-space coordinate**. The world model's discrete observation `state.anchor_id ∈ {1, ..., 384}`.
- A **citation anchor**. Every training-sample answer must include at least one token of the form `[corpus:Lx-y]` where `Lx-y` is a line-range in one of the two source corpora; the line-range is indexed by the anchor.
- A **retrieval key**. `HexaRAG` keys its keyword index by `(paired_state, ordinal)`.

### 4.4 Transition prior

The 8 × 8 transition prior matrix `T[i, j] = P(next_mode = j | current_mode = i)` is initialized from an ordering prior derived from a structured cultural corpus, and frozen as `configs/transition_v1.yaml` at release time. Versioning is explicit: `transition_vN` cannot silently replace `transition_vM`. This is a deliberately opinionated prior. We are *not* claiming the source ordering is empirically optimal. We are claiming that (a) a non-trivial prior is better than uniform, (b) a *culturally-grounded* prior is legible to users and auditors, and (c) the prior's specific claims are *learnable around* given sufficient data (§6).

---

## 5. Seven Cognitive Increments

Rather than claim a monolithic "AGI path" or a single scaling law, we decompose the cognitive capability surface into seven increments. Each is independently shippable, independently ablatable, and maps onto a specific module or contract. We label them §13.2.1–§13.2.7 in our internal operating manual and summarize here.

| # | Increment | Mapped to |
|---|---|---|
| 1 | **Grounded citation as a first-class output token** | §6 (citation gate) + §4.3 (anchor vocabulary) |
| 2 | **Discrete, auditable state over continuous latents** | §4 (mode / anchor state) + `world_model` service |
| 3 | **Compositional skill graph over free-form agentic loops** | `planner` service (skill DAG) |
| 4 | **Refusal as a typed output, not a sampling accident** | §7.1 safety level L0–L4 + A4 adversarial training (§6.1) |
| 5 | **Cross-version data–adapter–metric binding** | `adapter_version` + `data_hash` + bucketed eval (§6.4) |
| 6 | **Append-only, replayable trace as the ground truth** | §7.2 audit contract |
| 7 | **Structural prior as engineering, not metaphysics** | §4.4 transition prior + §2 disclaimer |

We argue these seven are both *necessary* and *sufficient* for an audit-bearing LLM deployment in a regulated domain. The sufficiency claim is strong and we do not defend it here empirically; we defend it *operationally* — each increment removes a specific class of failure mode we have observed in scale-first deployments.

**What this is not.** It is not a claim to AGI, not a replacement for scaling, and not a general-purpose cognitive architecture. It is a *disciplined subset* for applications where auditability and refusal-correctness dominate capability.

---

## 6. Training Pipeline: Five Classes, Three Gates, Bucketed Rollback

### 6.1 Five generator classes

Training samples come from five explicit classes, each with a separate generator module and a frozen target quota:

| Class | Share | Semantics |
|---|---|---|
| **A1 Extraction** | 60% | Ground-truth QA from corpora. Answer must cite `[corpus:Lx-y]`. |
| **A2 Paraphrase** | 20% | A1 with the *question* rewritten; answer / citation / mode-anchor all inherited. |
| **A3 Multi-hop** | 15% | 2–3 chunk reasoning (causal or contrast). Citations set-unioned. |
| **A4 Adversarial** | 5% | Induced-overreach questions paired with *compliant refusal* answers. Subtypes: induce / boundary / mystical. |
| **A5 Schema** | 5% | Strict JSON output (e.g., `state.anchor_infer` or `safety.assess`) with embedded citations. |

All five accept a dependency-injected `Teacher` callable. Local development uses a stub teacher (zero cost, pipeline-correctness only); production runs inject Anthropic Claude or OpenAI GPT. The dispatcher selects per-class teachers in stub mode and shares a single teacher in real mode.

### 6.2 Three hard gates

Any sample admitted to the training pool *must* pass all three:

1. **Citation gate.** Answer contains `[corpus:Lx-y]`; the line-range resolves in the chunker index.
2. **Consistency gate.** Teacher multi-samples on the same prompt; cosine-similarity variance (sentence-transformer) + Jaccard variance of citations below threshold. In stub mode, the gate returns `PASS` with a recorded stub reason.
3. **Compliance gate.** No un-negated occurrence of {diagnose / prescribe / cure / guarantee heal / miracle effect}. Negation windowing is explicit (operating manual §6.2.1).

A4 (adversarial) has an inverted compliance expectation: a sample that *does not* refuse to the induced overreach fails. This is enforced by flipping the gate polarity per-class.

A5 (schema) has an additional *structural gate* earlier in the pipeline (strict JSON parse + field presence), below the three-gate tier.

Failed samples are *not* discarded. They enter a `<out>.rejected.jsonl` file with full `gate_results`, and every verify event lands in `<out>.audit.jsonl` for diff-based methodology iteration.

### 6.3 The adapter–dataset binding

Adapters are named:

```
outputs/llama3.2-3b-qlora-v{N}-{data_hash[:8]}-{code_hash[:8]}/
  adapter_config.json
  adapter_model.safetensors
  meta.json
```

`meta.json` includes `data_hash`, `code_hash`, `pass_rate_by_class`, the teacher backend used, and a pointer to the dataset meta that generated it. Given an `adapter_version`, one can mechanically locate the dataset meta, replay the generation (with the same seed and stub teacher for hermetic reproduction, or the same real teacher and prompts for noisy reproduction), and re-train.

### 6.4 Bucketed evaluation and the 3% rollback

A single aggregate score is an averaging device that hides regressions. We evaluate per-bucket (class × difficulty × corpus-source) and apply:

> **Rollback rule.** If any bucket × any reported metric (pass rate, citation coverage, refusal correctness) degrades > 3% relative to the previous released adapter, version `N` is not released; version `N-1` is retained.

The threshold is deliberately tight. The rationale is that in regulated deployment, the cost of a silent regression to one sub-population is larger than the marginal benefit of a headline improvement. This commitment maps to the classical principle of *non-regression*: rather no progress than regression.

---

## 7. Safety and Audit Contract

### 7.1 Four-level safety

Every inference span is tagged with a monotone safety level:

- **L0** — nominal. No concern flags raised.
- **L1** — advisory. Mild ambiguity or minor compliance risk; proceed with a visible caveat.
- **L2** — constrained. Risk requires bounded answer form (e.g., schema-only, citation-required, no free-form elaboration).
- **L3** — refusal. Answer is a typed refusal object; reason code is logged.
- **L4** — escalation. Full rollback of the current dispatch; event published to the event bus; the session's state is preserved but the turn is void.

Safety level can only *increase* across a span. Downstream modules see an envelope, not the raw output, whenever level ≥ L2.

**Two-track architecture (non-self-persuasion property).** The level-raising path is implemented as a track *independent* of the capability-maximization path. The capability track produces a candidate output; the boundary track (§7.4) evaluates that candidate against an explicit rule corpus and may unilaterally raise the safety level or substitute a typed refusal. Crucially, *the same model is not asked to talk itself out of an objection it just produced*: the boundary track's verdict is binding, and the capability track has no privileged reasoning channel through which to overturn it. We call this the *non-self-persuasion* property; it is the contractual analog of the bicameral two-chamber principle in legislative practice.

### 7.2 Audit contract

Every service endpoint **must**:

1. Accept `X-Trace-Id` and `X-Parent-Span-Id` from the request header; pass through to `record_span`.
2. Hash inputs and outputs (`input_hash`, `output_hash`); never persist raw text in the audit log.
3. If the solver was invoked, propagate `adapter_version` (validated against `ADAPTER_VERSION_PATTERN`).
4. Monotonically raise `safety_level` on any rule hit.
5. Name operations as `<service>.<op>` for filterable snapshot queries.

Violations are PR-blocking (CI enforces via a schema lint).

### 7.3 Redactor (PRD-04)

A compliance redactor removes PHI / PII (Chinese mobile-number regex, identity-card number, email, bank card, IP, URL, exact date → month granularity) from any `extra` payload before it hits the audit store. The redactor is invocable both as a middleware (`record_span(redact_extra=True)`) and as a standalone API (`POST /research/redact`), with a preview variant that counts matches without modifying the payload.

### 7.4 Refusal-as-Contract: Boundary Gate and Refusal Object

The L3 (refusal) and L4 (escalation) levels are not free-form generations with a "sorry, I cannot" template. They are *typed objects* enforced by an upstream **Boundary Gate**.

**Boundary Gate — five-dimensional pre-action check.** Before any action-bearing dispatch (`executor.dispatch`, `solver.infer` in agentic mode, any external-effect tool call), the candidate action is evaluated along five orthogonal dimensions:

1. **Permission** — does the calling principal have authority for this action?
2. **Environment** — is the action consistent with the current spatiotemporal / operational context?
3. **Personal safety** — does the action risk human harm under any plausible execution?
4. **Equipment safety** — does the action risk damage to the device or coupled hardware?
5. **Task constraint** — does the action remain within the bounds of the originally accepted task?

Failure on any dimension routes the dispatch to a **typed refusal object**:

```
RefusalObject {
  reason_code: <enum>,          // why we refused
  trigger_clause: <string>,     // which gate clause / rule was hit
  alternative_action: <action | null>,  // a safe substitute, if any
  escalation_path: <recipient>, // who/where to route if user disputes
}
```

This four-tuple — *reason / trigger / alternative / escalation* — is the contract surface. A refusal that does not populate all four is itself a contract violation and is rejected by the schema lint at PR time. The point is operational: refusal is *not* a silent strike. It is a cooperative move that hands the conversation forward to a defensible alternative or an escalation channel, with sufficient context for an auditor to understand what happened.

**Default output schema (informational dispatches).** Independently of refusals, every non-refusal answer in agentic / decision-support mode is constrained to a five-field structure: **conclusion + confidence + evidence chain + boundary judgment + next action** (do / don't / escalate). This is enforced at training time as the A5 schema class (§6.1) and at runtime as a wrapper around the solver output. It is the runtime counterpart of the audit contract (§7.2): an answer that omits any of the five fields cannot be admitted to the audit log.

---

## 8. Canonical Form (Subject Layer) and Applied Projection (Deployment Layer): a Meta-Architectural Discipline

The discrete state space (§4) and the seven cognitive increments (§5) sit on top of an explicit two-layer commitment that we treat as a system-level engineering discipline rather than a metaphysical claim.

**Subject layer (canonical form, intrinsic primary).** The formal cognitive structure — the binary primitives, the 8 base modes, the 64 paired-state composite, the 384 ordinal anchors, and the canonical↔deployed permutation — is the *subject*: one, invariant, the cognitive form to which every application must trace back. In code it lives in the core state-space module and the frozen `configs/transition_v1.yaml`.

**Projection layer (applied form, deployed).** Every domain module — health (TCM syndrome differentiation, rehabilitation, voice interaction, pulse acquisition), education, mental-health computing, robotics — is a *projection* of the subject onto a specific spatiotemporal / use context. Projections are many; they may differ across domains; but they must all back-trace to the same subject.

This separation is operationalized as a CI-enforceable contract:

- **Three-question admission gate.** Before any new feature, DTO, or hard rule enters the codebase, three questions must be answered: (i) which subject-layer structure does it correspond to? (ii) can it be back-traced via canonical-text citations? (iii) if not — admission is **denied**.
- **Module-level docstring requirement.** Every `domains/*` module's top-level docstring must declare *"this module is a projection of <subject-layer structure> onto <applied context>"*; absence triggers a CI fail.
- **DTO discipline.** New DTOs annotate their subject-layer correspondence (e.g., `SixMeridianState` corresponds to the canonical-form three-yang / three-yin partition; see §9.3).
- **Conflict policy.** *Projections may conflict* (different domain methodologies are allowed); *the subject layer must be conflict-free*, since two domains tracing to incompatible roots would invalidate the cross-domain audit guarantee.
- **Adapter discipline.** A LoRA adapter may *extend* the subject's reach into a new projection (deployed capability); it must *not* rewrite the subject's values. Training answers must back-trace to subject-layer citations (`[corpus:Lx-y]`); rejection by the citation / compliance gates (§6.2) is the runtime enforcement of this rule.

**Why this is engineering, not metaphysics.** Without this discipline, accumulated projections drift. Health applications drift toward non-falsifiable interpretation; education applications drift toward generic chat; the audit log degrades into a log of opinions. With it, the citation gate (§6.2), the bucketed-evaluation rollback (§6.4), and the audit replay (§7.2) all become *cross-domain* invariants — a TCM application's audit trail becomes meaningfully comparable to a robotics application's trail, because both share subject-layer coordinates. This is what makes the 384-anchor vocabulary *useful* rather than decorative.

---

## 9. Domain Realization: a Five-Stage Reasoning Chain for TCM Syndrome Differentiation

We describe the first non-trivial projection of the subject layer (§8) into a clinical domain: **Traditional Chinese Medicine (TCM) syndrome differentiation**. This is the engineering scaffold against which the domain-specific TCM adapter (its own `adapter_version` line, §6.3) will be trained from the new generator class introduced in §9.7, and against which the L3 boundary (§11) is drawn.

The chain has five stages; *the order is not commutative*; *the paired-state anchor (§4.2) is the unique root — every later stage routes through it*. A pipeline that skips the anchor step is a contract violation enforced at the orchestrator level (§9.9).

**Implementation status (v0.1.1).** The chain is realized as `domains/health/tcm_pipeline.py` (the `differentiate()` orchestrator) and exposed via `POST /tcm/differentiate` mounted on the FastAPI gateway (§9.9). Stages 1–5 execute in this exact order; the anchor-binding stage (§9.2) raises a typed `ValueError` on failure and the orchestrator halts stages 3–5 with safety level L2 ("hard brake"). Every dispatch is wrapped in an audit span with `redact_extra=True`, so any `birth_year_month` field is reduced to month granularity *before* the trace log.

### 9.1 Stage 1 — Spatiotemporal Context (cosmological baseline; pure-function, formula-driven, no learning)

The first stage is a **pure-function** spatiotemporal context engine. Given `(timestamp, lat, lon, optional birth_year_month)` it produces:

- **`GeoContext`** — `{ temperature, humidity, climate, political / economic / cultural }` derived from `(lat, lon)`.
- **`WuyunLiuqi`** — `{ jieqi (the 24 solar terms with exact moments), year / month / day / hour ganzhi, sitian, zaiquan, zhuqi, keqi, zhuyun, keyun }` computed from the lunar–stem-branch calendar. We pin `WUYUN_LIUQI_VERSION = "v0-draft-2026-04-26"`. Any change to the ganzhi algorithm, the jieqi-moment table, the sitian / zaiquan table, or the main / guest-qi table requires a version bump and a snapshot regression test. Implementation depends on `lunar_python` or `sxtwl`; the sitian–zaiquan and zhu / ke-qi tables are hand-encoded from canonical commentaries and frozen pending TCM-advisor review.

The aggregate is **`SpacetimeContext { geo, time }`** — *deterministic, auditable, rollback-able*. Epistemologically it is **background context** (analogous to a weather forecast), **not a diagnostic step** — it never emits a diagnosis, never claims causation, never modifies user state.

**Hard rule (anti-fortune-telling).** *Wǔ Yùn Liù Qì is a cosmological baseline, not a fortune predictor.* The output layer must not produce statements such as "your luck this year", "Tai Sui clash", "auspicious / inauspicious", "Bazi / fate diction". The compliance gate (§6.2) blocks such samples *before* training; the safety contract (§7.1) raises L3 refusal at runtime if generation strays.

**Privacy.** `birth_year_month` is **optional**; if supplied, only `birth_year_ganzhi` and `birth_month_ganzhi` are derived (deliberately *not* day or hour, which together with year and month would constitute a complete natal chart). The redactor (§7.3) reduces date precision to month granularity *before* the audit log.

### 9.2 Stage 2 — Anchor Binding (the unique root)

`SpacetimeContext + linguistic input + (optional) prosodic input` → 8-mode canonical state → 8-mode applied state → 384-anchor coordinate. This stage is the **only** authorized writer to `state.anchor_id`; no later stage may bind independently. The Stage-1 output enters Stage 2 *as conditioning context*, not as a substitute anchor.

### 9.3 Stage 3 — Six-Meridian Differentiation (六经辨证)

Three-Yang (太阳 / 阳明 / 少阳) and Three-Yin (太阴 / 少阴 / 厥阴). The paired-state anchor is mapped to the six meridians via a frozen mapping table (`LIUJING_MAP_VERSION = "v0-draft-2026-04-25"`):

| Base mode (Five-Phase polarity) | Primary | Secondary |
|---|---|---|
| BM₀ (metal-yang) | Yangming | Taiyang |
| BM₁ (metal-yin) | Yangming | Taiyin |
| BM₂ (fire) | Shaoyin | Yangming |
| BM₃ (wood-yang) | Jueyin | Shaoyang |
| BM₄ (wood-yin) | Shaoyang | Jueyin |
| BM₅ (water) | Shaoyin | Taiyang |
| BM₆ (earth-yang) | Taiyin | Yangming |
| BM₇ (earth-yin) | Taiyin | Yangming |

Any modification requires a version bump and a snapshot regression test, **and** TCM-advisor review before L3 freeze.

The Stage-1 `time.zhuqi / keqi` fields serve as the *cosmological baseline*; Stage 3 reports the **deviation** of the patient's six-meridian state from that baseline (the *Inner Canon* doctrine: "harmonious with heaven-earth → orthotic; deviant → pathological").

### 9.4 Stage 4 — Eight-Principle Differentiation (八纲辨证)

Yin / Yang × Exterior / Interior × Cold / Heat × Deficiency / Excess; each axis lives in `[-1, +1]`. This is a coarse classification stage downstream of Stages 2–3.

### 9.5 Stage 5 — Theory · Method · Formula · Drug (理法方药)

Eight Therapeutic Methods (八法) → monarch / minister / assistant / courier (君臣佐使) role assignment → per-drug *generation / growth / transformation / harvest / storage* (生 / 长 / 化 / 收 / 藏) annotation.

Seasonal-modulating drugs (`当令药`) and channel-conducting drugs (`引经药`) are modulated by Stage-1's `time.sitian / zaiquan`. The 引经药 field is **hint-only**: `fang.yinjing` may be empty, but `yinjing_hint: str | None` must be written; an empty hint triggers an output-layer educational suggestion ("consult a TCM physician about possible channel-conducting drugs"). Single-herb formulas, acute-condition formulas, and external-application formulas are exempt from the hint requirement.

### 9.6 Hard constraints (nine, CI-enforced)

1. Output is *educational / inferential*, never *prescriptive*. Specific dosages (gram / liang / qian) are blocked; the L3 boundary forces dosage-bearing answers to "consult corpus reference" form.
2. Every herb in every formula carries `citations: [CorpusRef]`; absence routes the sample to the negative-example pool.
3. **Anchor-first.** `SpacetimeContext` to "理法方药" *must* route through `state.anchor_id`. Wǔ Yùn Liù Qì at Stage 1 does *not* preempt the anchor at Stage 2.
4. The platform does **not** maintain a self-built TCM herb database; herb information is sourced from canonical-text citations only.
5. 引经药 is hint-only (no FAIL); `yinjing_hint` is mandatory.
6. No fortune-telling diction (§9.1).
7. Wǔ Yùn Liù Qì is pure-function and version-pinned (§9.1).
8. `birth_year_month` is optional and must pass through the redactor at month granularity (§9.1).
9. Cosmological fields (`year_ganzhi / sitian / zaiquan / zhuqi / keqi / current_jieqi / jieqi_moment / zhuyun / keyun`) are *non-personal* — at the same instant and `(lat, lon)`, all users read the same values; only the `personal_ganzhi` sub-field varies per user.

### 9.7 Cross-project contract addenda

The TCM 5-stage chain adds five health-domain modules to the platform: `domains/health/{geo_context, wuyun_liuqi, liujing, bagang, prescription}.py`. The workbench adds a sixth generator class **A6 TCM-Chain** (10% allocation, scheduled Stage 9+), with two additional gates: **Chain Completeness** (all five stages present and consistent; cosmological-baseline citation present) and **Citation Chain** (every drug-level claim traces to a canonical-text citation along the chain). The TCM adapter is trained on a **separate `adapter_version` line** from the conversational adapter and is **never** distributed as quantized open weights — L3 cloud serving only.

### 9.8 Initial corpora and pulse acquisition

The source-canon corpus directory ships, for v1, with **《濒湖脉学》 *(Bīn Hú Mài Xué*, Li Shizhen, 1564)** in a six-file split: full text, four-character verse (四言诀), seven-character verse for the 27 pulses (27 脉七言), classical-Chinese original, vernacular translation, and an OCR diff log for provenance. A pulse-acquisition subdomain (`domains/health/pulse/`, ~28 modules) realizes the engineering counterpart: bilateral-symmetry analysis, nine-position aggregation (`jiuhou_aggregator`), pediatric seven-pulse adapters (`pediatric_seven_pulse`, `pediatric_sizing`), ling-curr stream conversion (`ling_curr_pulse`), depth and position decomposition, morphology and notch detection, evidence routing into the audit log (`evidence_router`), and a wearable-device adapter (`watch_adapter`). This is implementation-only at the time of writing; **no v1 TCM-adapter metrics are claimed** — those await the gates of §11.

### 9.9 Pipeline Orchestrator and REST Surface

The five stages above are *static specifications* — their wiring into a single end-to-end call is contained in **`domains/health/tcm_pipeline.py`** (file budget 420 lines), exposed as **`POST /tcm/differentiate`** through `api/tcm.py` (95 lines) and mounted on the FastAPI gateway. The orchestrator's contract:

- **Input** — a Pydantic `TcmDifferentiationInput` model: `timestamp`, `lat`, `lon`, `chief_complaint` (Chinese free text), and three optional fields (`birth_year_month` reduced to month granularity *before* it crosses the redactor boundary, an optional `pulse_session` reference into the pulse subdomain, and an optional partial `fang` for human-in-the-loop refinement).
- **Output** — a `TcmDifferentiationResult` carrying the Stage-1 `SpacetimeContext`, the Stage-2 `(anchor_id, ordinal_index)`, the Stage-3 `SixMeridianState`, the Stage-4 `BagangState`, the Stage-5 `PrescriptionPlan`, a deduplicated `citations` list aggregating across all stages, a per-stage `stage_logs` array (`name / duration_ms / status / evidence_count / note`), the final `safety_level` (0–4), a boolean `compliance_passed`, and the audit `trace_id`.
- **Hard rules enforced at orchestrator level (independent of per-stage logic)** —
  1. *Anchor-first.* Stage 2 failure raises a typed `ValueError`; the orchestrator catches it, emits an audit span at safety level **L2** (a "hard brake"), and re-raises so that stages 3–5 are *never invoked*. The REST endpoint translates this `ValueError` into HTTP 422 with a structured hint.
  2. *Citation-required for compound prescriptions.* If the Stage-5 result contains a compound formula (i.e. not in the exempt set: `single`, `emergency`, `external`) and any drug entry is missing `citations`, the orchestrator forces `compliance_passed=False` and escalates to **L3**.
  3. *Dosage detection.* If `prescription.metadata` carries an `l3_downgrade` flag (set by the prescription module on dosage emission), the orchestrator escalates to **L3** and substitutes a corpus-reference-only output.
  4. *Privacy at the trace layer.* All audit spans are emitted with `record_span(redact_extra=True)`, so `birth_year_month` is reduced to month granularity in the persisted log even if the input string carried finer precision.
  5. *Anti-fortune-telling at the runtime layer.* The Stage-1 `WuyunLiuqi` output is never surfaced as personal prediction; the orchestrator's audit `extra` records cosmological fields under non-personal keys (§9.6 hard rule 9).

The audit `extra` payload includes: `anchor_id`, `ordinal_index`, `dominant_meridian`, `stage_count`, citation count, `compliance_passed`, and `pipeline_version`. The `pipeline_version` is bumped whenever any stage's version constant (`WUYUN_LIUQI_VERSION`, `LIUJING_MAP_VERSION`, the prescription module's version) is updated, providing a single coarse handle for cross-version replay.

This is the first end-to-end realization of any cognitive subject-layer claim from §8. Subsequent domain projections (rehabilitation robotics, voice interaction, education, mental-health computing) will follow the same orchestrator-and-endpoint pattern: a single domain-specific `*_pipeline.py` module exposing one `differentiate`-style entry point, and a single thin `api/*.py` router enforcing the audit, redact, and safety contracts.

---

## 10. Deployment: Hybrid Local + Cloud

The deployment target is a hybrid user-facing model with **three client surfaces** sharing one cloud backend:

- **Local (user machine).** The quantized LLM runs locally via Ollama (native Mac / Windows / Linux support), exposed at `localhost:11434`. Three client surfaces sit on top:
  - **CLI** — `doorm-client` Python package, distributed via PyPI (`pip install doorm-client`); minimal dependency footprint.
  - **Container** — official Docker image bundling Ollama + the client; for headless / server-side deployment.
  - **Desktop** — packaged GUI application (Electron / Tauri candidate) for end-users without command-line familiarity.
  All three call the same cloud REST surface and use the same device-local key material (below).
- **Cloud (`api.doorm.ai`, ECS-hosted).** Knowledge (`memory`), safety (`/safety/assess`), state-anchor inference (`/state/anchor_infer`), Mind Encoder (`/health/encode`), and an authoritative `/solver/infer` fallback. Plus `/auth/*` (device activation / JWT refresh / revocation) and `/registry/*` (adapter manifest, signed URLs).

Every client request carries a short-lived JWT (cloud-issued, 1 h) *and* an ED25519 signature over the body computed with a device-local key. The keypair is generated on first activation and never leaves the user's machine.

Crucially: the client LLM is the L1 / L2 open adapter (quantized open-weight format). The cloud `/solver/infer` endpoint retains access to the L3 higher-precision adapter line — including the TCM adapter trained per §9 — in FP16 safetensors, **not** distributed as quantized open weights. This is the commercial / clinical boundary (§11).

---

## 11. Open / Closed Source Disclosure Policy

| Tier | Content | License / Venue |
|---|---|---|
| **L1 — Platform skeleton** | `WorldModel-OS/` source tree (7 services, gateway, audit, state space, redactor); `Finetune Workbench/` source tree (5 generators, 3 gates, orchestrator, bucketed evaluator) | **AGPL-3.0-or-later**, GitHub |
| **L2 — Cognitive increments + methodology** | §4, §5, §6, §7 of this paper; `adapter_version` contract; methodology for transition prior, gate calibration, bucketed rollback | **Defensive preprint** (this document), CC BY 4.0 |
| **L3 — Production weights + clinical partnerships + secrets** | Trained LoRA weights on regulated medical corpora; clinical MOU details; production device-auth secret layout; cloud infrastructure topology | **Retained** under DOORM AI PTE. LTD. commercial / clinical governance. Cloud-side serving only. |

We have chosen AGPL-3.0 for the code (copyleft + network-use clause) and CC BY 4.0 for this paper (broad reuse with attribution). Commercial users requiring a non-copyleft license must contact the corresponding author.

Our three hard release gates, which govern whether any L1 / L2 artifact ships as an official DOORM release:

1. **Reproducibility gate.** A third party on a single consumer GPU (RTX 3090 / 4090, 24 GB VRAM) can, within 24 hours, run the full pipeline from `run_dataset_v1.py` through `train` through `eval` and obtain a v1 adapter with `adapter_version` matching our reference.
2. **Benchmark gate.** A domain-specific evaluation (TCM and rehabilitation robotics) on at least one bucketed metric, with pre-registered rollback thresholds.
3. **Real-adoption gate.** At least one medical institution MOU or paid deployment. (We do not disclose partners' identities in this paper.)

No artifact passes "release" without all three.

---

## 12. Limitations, Honest Status Table, and Falsification Criteria

We are pre-empirical in this paper. To avoid the "everything works in principle" failure mode, we make three honest disclosures: (a) a limitations list against the v0.2 plan; (b) an explicit ✅ delivered / 🟡 in-progress / 📋 planned status table aligned with internal release decisions; (c) the falsification criteria under which the entire architectural thesis would be rejected.

### 12.1 Specific limitations

- **No v1 adapter exists yet.** Stage 3 (5,000-sample dataset generation + first QLoRA adapter) is blocked on ECS GPU provisioning and teacher API budgets.
- **Retrieval-side embedding model is download-gated.** `HexaRAG.query()` ships vector-embedding retrieval keyed on `BGE-zh-v1.5` (`RETRIEVER_VERSION = "v1-bge-zh-2026-04-26"`); when `sentence-transformers` is unavailable on the host, retrieval falls back to a deterministic blake2b stream that preserves the API contract but carries no semantic signal. Cross-encoder re-ranking and FAISS scaling are deferred to M3.
- **Embedding in `perception` is a placeholder.** `d=256` deterministic blake2b stream; swap to the same embedding chosen for retrieval.
- **Transition prior is not yet learned, only initialized.** The config freezes to `transition_v1`. We have not yet run the learning refresh described in §4.4.
- **Compliance gate is rule-based.** A learned classifier may reduce false positives on negated occurrences.
- **Device authentication is designed, not implemented.** `/auth/*` and `/registry/*` are scheduled for V0.1 / M1 (2 weeks post this paper).
- **Cross-lingual coverage is Chinese-only at L2.** English surface for outreach exists; English-language domain grounding is not yet done.
- **Quantization precision has not been ablated.** Q4_K_M vs Q5_K_M vs FP16 under the bucketed metric suite is a release-blocking experiment.
- **The 3% rollback threshold is unjustified empirically.** It is a conservative choice whose optimal value awaits dataset-v1 and adapter-v1 data.
- **The base-mode ↔ six-meridian mapping is a v0 draft.** `LIUJING_MAP_VERSION = "v0-draft-2026-04-25"` is **not** TCM-advisor-reviewed; it must not be treated as a clinical claim. It is engineering scaffolding under §9.3.
- **The Wǔ Yùn Liù Qì tables are hand-encoded.** The sitian–zaiquan and zhu / ke-qi tables are pinned to `WUYUN_LIUQI_VERSION = "v0-draft-2026-04-26"` and will be re-audited against multiple canonical-commentary sources before L3 freeze.
- **The pulse-acquisition subdomain (~28 modules) is implementation-only.** No v1 metrics — clinical or otherwise — are claimed. End-to-end validation against synchronized expert-pulse-pad recordings is a release-blocking experiment.
- **The A6 TCM-Chain generator and the two added gates (Chain Completeness, Citation Chain) are scheduled, not running.** Stage 9+; gated on Stage-3 dataset-v1 and TCM-advisor review of §9.3 / §9.5 mappings.
- **The non-self-persuasion property (§7.1) is currently a *contractual* property, not a *proven* one.** Empirical adversarial testing (red-team prompts attempting to bypass the boundary track) is on the v0.2 evaluation roadmap.
- **The strategic memory tier (§3.2) is a design, not an implementation.** The three existing tiers (structural / contextual / episodic) ship in v1; the strategic mid-tier is scheduled for v0.2.

### 12.2 Honest status table (delivered vs. planned)

| # | Claim from §1–§9 | Status | Anchor |
|---|---|---|---|
| 1 | Hybrid state space (continuous + discrete unified kernel) | ✅ Delivered v0.1 | `core/schemas.py` `HybridState` |
| 2 | Four-anti positioning (anti-token / anti-scale / anti-stacking / anti-compute-monopoly) | ✅ Delivered (strategy) | §1 P1 + governance docs |
| 3 | Structural-prior duality (canonical / applied) | ✅ Delivered v0.1 | core state-space module + frozen `transition_v1.yaml` |
| 4 | TCM five-stage chain (architectural) | ✅ Delivered v0.1 | `domains/health/tcm_pipeline.py` |
| 5 | TCM adapter training + clinical-feedback closed loop | 🟡 In progress | Sprint C+ in parallel with B+ (synthetic-data bootstrap; clinical validation deferred) |
| 6 | Explanation / prediction layer separation (evidence + metric) | ✅ Delivered v0.1 | `core/schemas.py` `AuditRecord` (evidence / assumptions / uncertainty / safety_flags / adapter_version) |
| 7 | Schema versioning + carrier-of-id output | ✅ Delivered v0.1 | `ADAPTER_VERSION_PATTERN` + four advisor-signed knowledge tables (six-meridian mapping / 27-pulse↔anchor mapping / Wǔ Yùn Liù Qì / seasonal-pulse thresholds) |
| 8 | Unified core + multi-task heads (7 microservices + health / research domains) | ✅ Delivered v0.1 | `services/` + `domains/health` + `domains/research` |
| 9 | Exoskeleton + small-step rollback training | 🟡 In progress | `pulse/bilateral_symmetry.py` cap 0.4 + 4-level `SafetyRollbackLevel`; client-side adapter distribution Sprint B (cloud-prep) |
| 10 | Auditable / regulator-ready / reproducible output | ✅ Delivered v0.1 | `core/audit.py` trace-id chain + JWT middleware + AGPL dual licensing + mirror matrix |
| 11 | Confidence / entropy / interval as first-class | ✅ Delivered v0.1 | `HybridState.confidence` + `phase_distribution` (softmax 8-d) + `AuditRecord.uncertainty.{type:entropy/variance/credible_interval}` |
| 12 | Health / TCM / rehabilitation / pulse-acquisition domain | ✅ Delivered v0.1 | PRD-02 / PRD-03 / PRD-05 + full `domains/health/` |
| 13 | Other domains (government / finance / meteorology / autonomous driving / logistics) | 📋 Out of scope (v0.2) | Not present; no PRD / no code (see §1.1 non-coverage list) |

### 12.3 Falsification criteria

The architectural thesis of this paper is **falsifiable**. We commit to retracting our positions if any of the following hold under a fair, third-party-reproducible evaluation:

1. **Anti-scale falsification.** A pure-scale Transformer baseline, on the same TCM-syndrome-differentiation dataset and the same per-decision auditable evaluation harness, dominates our HSS-based system on RMSE / Brier / CRPS triple — *and* the gap cannot be explained away by appeal to any of our seven cognitive increments.
2. **Discrete-anchor falsification.** Our 8-mode / 64-paired / 384-anchor discrete structure fails to deliver a stable, statistically significant gain over a continuous-only baseline on at least one of: evidence-citation hit rate, reasoning-chain replay consistency, or clinician-acceptance rate, at large sample size.
3. **Audit-replay falsification.** The `AuditRecord` chain (`trace_id` / `evidence` / `assumptions` / `uncertainty` / `safety_flags` / `adapter_version`) cannot be reproduced to field-level identity by an independent third party in a clean-room environment, on the same adapter and the same input, given our published methodology.

Any of the above three, demonstrated, ends the architectural thesis. We commit to publishing the negative result and revising the framework.

---

## 13. Discussion: Positioning, Operational Universality, and Cross-Domain Migration

We are making an intentionally unfashionable bet. The industry rewards and reinforces the view that capability improvements come from parameters, tokens, and RLHF sweeps; that interpretability is a research side project; that cultural specificity is a marketing layer over a universal base model; and that safety is tuned rather than typed.

We claim three things run contrary to this consensus and will hold:

1. **For regulated or audit-bearing deployment**, the decisive capability is *typed refusal and per-decision citation*, not benchmark score. A 3B model with a refusal contract outperforms a 70B model without, measured against any deployable-production criterion.
2. **For culturally-situated applications**, structural priors from the culture-of-use are cheaper and more legible than end-to-end learning from web data. The 384-anchor vocabulary costs us nothing to embed and gives auditors a shared language with practitioners.
3. **Reproducibility on consumer GPUs is a strategic commitment, not a limitation.** A pipeline that runs end-to-end on a single 3090 within 24 hours enables third-party verification in a way that 512-GPU runs cannot. This compounds over time into ecosystem trust that large-scale proprietary systems cannot match.

None of these claims are yet empirically supported in this paper. They are the thesis against which we will report results in v0.2+ (the falsification criteria of §12.3 are the operational form of these claims).

### 13.1 Universality, defined operationally

We deliberately avoid loose claims of "universality". Instead we adopt a five-axis operational definition: a system is *operationally universal* over a target class iff it satisfies — jointly — (i) **high abstraction level**, (ii) **broad applicability with evidence**, (iii) **low migration cost**, (iv) **explicit boundaries and non-coverage**, and (v) **falsifiability**. By this metric, our v0.1 / v0.2 stack is *not* universal across all health-and-medicine-adjacent domains; it is universal only within TCM + rehabilitation + pulse-acquisition under the migration recipe of §13.2 below.

### 13.2 Cross-domain migration recipe (cost, cleanly stated)

We commit to a precise recipe for extending the platform to any new domain, with explicit "what is invariant" / "what must be replaced" / "approximate cost" budget:

| Layer | Content | When migrating to a new domain |
|---|---|---|
| **Invariant (kernel)** | HSS state space; the four-anti positioning; `AuditRecord` contract; `adapter_version` versioning; the seven-microservice skeleton (`gateway / perception / world_model / planner / executor / event_bus / safety`) | **Zero modification** |
| **Replaceable (domain adapter)** | Domain LoRA adapter weights; domain knowledge tables (in TCM: six-meridian / eight-principle / Wǔ Yùn Liù Qì / 27-pulse↔anchor mapping); domain corpus and evidence sources; domain PRD and scenario API | **Must be re-built per domain** |
| **Cost estimate** | Reproducing the core demo: a single consumer GPU (24 GB, 3090 / 4090), end-to-end ≤ 24 h. Production-grade adapter: < 10 A100s, week-level (per STRATEGY §7) | — |

At v0.1 we have *only* completed "invariant → replaceable" full-stack delivery for the health line (PRD-02 / PRD-05). Domains outside health are at planning stage and are *not* claimed within v0.2 scope. The §1.1 non-coverage list and the §12.2 status table together provide the operational meaning of our universality claim.

---

## 14. Conclusion

DOORM WorldModel-OS commits to a specific, unfashionable architecture: two peer projects bound by a single regex; a discrete state space with structurally-primed anchors borrowed from a culturally-grounded combinatorial corpus; a training pipeline gated on citations and refusals; a safety contract in every span; an open-kernel / closed-weight disclosure structure; an explicit non-coverage list; a published cross-domain migration recipe; and a set of falsification criteria under which the entire thesis can be rejected. We have presented the design and the open / closed boundary in full. We have *not* yet presented evidence. Our next publication (v0.2 empirical) is scheduled for when the three hard release gates are met. Until then, everything in this document is a wager — but a clearly bounded, clearly falsifiable one.

We welcome scrutiny, reproduction attempts at the specified envelope, and collaboration — particularly from practitioners of TCM and rehabilitation who care about deploying AI they can audit.

---

## Author Contributions (CRediT)

We follow the Contributor Roles Taxonomy (CRediT) for transparent attribution.

- **Tong Li**: Conceptualization; Methodology; Software (system architecture); Writing – original draft; Writing – review & editing; Visualization.
- **Tong Tong**: Software (implementation, deployment); Data curation; Investigation (engineering experiments); Validation; Project administration; Resources; Supervision; Funding acquisition.
- **Wu Lihua**: Conceptualization; Investigation (literature review); Writing – review & editing.

All authors have read and approved the manuscript and agree to the order of authorship.

---

## Acknowledgments

The authors thank the maintainers of the open-source libraries on which this work builds — in particular PEFT [Mangrulkar et al., 2022], bitsandbytes, transformers, and PyTorch.

## Data and Code Availability

- Platform code: forthcoming at `github.com/doorm-ai/WorldModel-OS` (AGPL-3.0), release gated by §11.
- Workbench code: forthcoming at `github.com/doorm-ai/Finetune-Workbench` (AGPL-3.0), release gated by §11.
- Source corpora: two classical Chinese texts for which DOORM AI PTE. LTD. holds IP; the *structural* information (chunker output line-ranges, 384-anchor index) will ship with the release; the *textual* content will be available under a separate reading license.
- Adapter weights (L1 / L2 tier): forthcoming on Hugging Face at `doorm-ai/llama3.2-3b-doorm-v1` in Q4_K_M and Q5_K_M.
- Clinical (L3) adapter weights, including the TCM-chain adapter (§9) on its own `adapter_version` line: not distributed; served from cloud endpoints only.
- Source corpora at v1: 《濒湖脉学》 *(Bīn Hú Mài Xué*, Li Shizhen, 1564) — six-file split (full text, 四言诀, 27-脉七言, classical original, vernacular translation, OCR diff log) under `WorldModel-OS/原典/`.

## Conflict of Interest Statement

The authors disclose the following financial and material relationships:

- **Tong Li** is a shareholder of **DOORM AI PTE. LTD.** (UEN 202441729W), which holds the commercial IP on L3 artifacts described in §9.
- **Tong Tong** is the founding Director and a shareholder of **DOORM AI PTE. LTD.**, which holds the commercial IP on L3 artifacts described in §9.
- **Wu Lihua** is an Independent Researcher with no employment, equity, or commercial relationship with DOORM AI PTE. LTD.; her contribution to this paper was made on an unremunerated, academic-interest basis.

Intellectual property in this paper (including copyright and any derivative rights) is held by DOORM AI PTE. LTD. pursuant to an *Author Agreement* among the parties; authorship credit is allocated per the CRediT statement above.

## Funding Statement

This work was supported by internal R&D funds of DOORM AI PTE. LTD. No external public or commercial funding was received during the period covered by this paper.

## References (selected, to be completed in v0.2)

- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.
- Meta AI (2024). *Llama 3.2 Model Card*.
- Mangrulkar, S. et al. (2022). *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning*. Hugging Face.
- Ha, D. & Schmidhuber, J. (2018). *World Models*. arXiv:1803.10122.
- Hafner, D. et al. (2023). *Mastering Diverse Domains through World Models (DreamerV3)*. arXiv:2301.04104.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT\* 2019.
- Gebru, T. et al. (2021). *Datasheets for Datasets*. Communications of the ACM.
- European Union (2024). *Artificial Intelligence Act*.
- U.S. FDA (2024). *AI/ML-Enabled Medical Devices Draft Guidance*.
- Field, R. J. & Noyes, R. M. (1974). *Oscillations in chemical systems IV: Limit-cycle behavior in a model of a real chemical reaction*. J. Chem. Phys.
- 李时珍 [Li Shizhen] (1564). 《濒湖脉学》 *[Bīn Hú Mài Xué]*. Source corpus, see §9.8 and Data Availability.
- 黄帝内经 *[Huángdì Nèijīng]*. Source for the cosmological-baseline doctrine (§9.3) and for the orthotic / pathological framing of six-meridian deviation.

*(Further classical-corpus references, cognitive-architecture references, and clinical-AI references will be added in v0.2.)*

---

## Appendix A — The Cross-Project Handshake (normative)

```python
# WorldModel-OS/src/worldmodel/core/schemas.py
# Finetune Workbench/src/finetune/data_gen/schemas.py
# (the two files must contain this literal, byte-for-byte identical)

ADAPTER_VERSION_PATTERN = r"^v\d+-[0-9a-f]{8}-[0-9a-f]{8}$"
```

Semantics:
- `v{N}` — monotone release version, starting at v1.
- `{data_hash[:8]}` — first 8 hex chars of SHA-256(training JSONL file, canonical order).
- `{code_hash[:8]}` — first 8 hex chars of `git rev-parse HEAD` in the workbench at training time.

Any inference trace carrying `adapter_version` V implies, by this contract, exact knowledge of the training dataset and the code revision that produced V.

## Appendix B — Reproducibility Checklist

- [ ] Single consumer GPU (RTX 3090 or 4090, 24 GB VRAM, CUDA 12.1)
- [ ] Docker image pinned to `finetune-llama32:v0.1` (`docker/Dockerfile` in workbench)
- [ ] Python 3.11, `uv sync` reproduces lockfile
- [ ] Teacher backend configured (Anthropic Claude 3.5 Sonnet primary; OpenAI fallback); API keys in `.env`
- [ ] `run_dataset_v1.py --target 5000 --teacher anthropic --out data_gen/dataset_v1.jsonl --resume` completes within 18 h
- [ ] Resulting `dataset_v1.meta.json` matches reference `data_hash` to 8 chars
- [ ] `train.py` emits `outputs/llama3.2-3b-qlora-v1-{same_data_hash[:8]}-{code_hash[:8]}/`
- [ ] `eval --bucketed` matches reference `metrics_bucketed.json` within reported tolerance
- [ ] No workbench module imports `worldmodel.*` (grep-based lint green)
- [ ] Audit log `<out>.audit.jsonl` replays to identical gate decisions given identical seeds
