# DOORM WorldModel-OS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20053545.svg)](https://doi.org/10.5281/zenodo.20053545)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: AGPL-3.0](https://img.shields.io/badge/Code%20License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)

> **A Governance-First Architecture for Auditable Agentic Reasoning**
> Hybrid State-Space · Structurally-Primed Discrete Anchors · Anti-Scale Training Discipline

This repository hosts the **defensive preprint**, governance documents, threat model, and disclosure boundary for **DOORM WorldModel-OS** — a hybrid local/cloud operating system for auditable agentic world models, paired with the model-agnostic **Finetune Workbench**.

---

## TL;DR

- **What this is.** A position + system paper proposing seven cognitive increments for auditable agentic reasoning, grounded in a discrete state-space (8 base / 64 paired / 384 ordinal anchors), a four-level rollback contract embedded in every inference trace, and an anti-scale training discipline. Applications: Traditional Chinese Medicine (TCM) syndrome differentiation and rehabilitation robotics.
- **What this isn't.** An empirical benchmark paper. No metrics are claimed in v0.2. The reproducibility envelope, three release gates, non-coverage list, and falsification criteria are specified in §11.
- **Open vs closed.** L1 (platform skeleton) + L2 (training methodology) are AGPL-3.0-or-later / CC BY 4.0. L3 (production weights, clinical adapters, partner data, commercial terms, device auth keys) is held under separate commercial / clinical governance.

---

## Repository contents

| File | Description |
|---|---|
| `PAPER-v0.2-en.md` / `.pdf` | English paper |
| `PAPER-v0.2-zh.md` / `.pdf` | Chinese paper (中文版) |
| `LICENSE` | CC BY 4.0 |
| `CITATION.cff` | Citation metadata |
| `SECURITY.md` | Security contact |

---

## Companion code repositories

The platform and workbench code (separately governed):

- **[WorldModel-OS](https://github.com/doorm-ai/WorldModel-OS)** — L1 platform skeleton (AGPL-3.0-or-later)
- **[Finetune-Workbench](https://github.com/doorm-ai/Finetune-Workbench)** — L2 training workbench (AGPL-3.0-or-later); base-model agnostic

---

## How to cite

Use the `CITATION.cff` entry (GitHub auto-generates citation buttons), or:

```bibtex
@misc{doorm2026worldmodel,
  title  = {DOORM WorldModel-OS: A Governance-First Architecture for Auditable Agentic Reasoning},
  author = {Tong, Li and Tong, Tong and Wu, Lihua},
  year   = {2026},
  doi    = {10.5281/zenodo.20053545},
  url    = {https://github.com/doorm-ai/WorldModel-OS-governance}
}
```

---

## Authors

| Author | Role | Affiliation |
|---|---|---|
| **Tong Li**¹\* | Conceptualization; Methodology; Software (system architecture); Writing | DOORM AI PTE. LTD. |
| **Tong Tong**¹ | Software (implementation, deployment); Data curation; Validation; Project administration; Funding | DOORM AI PTE. LTD. |
| **Wu Lihua**² | Conceptualization; Investigation (literature review); Writing – review | Independent Researcher |

¹ DOORM AI PTE. LTD. (UEN 202441729W), Singapore
² Independent Researcher
\* Corresponding author: **service@doorm.ai**

*Author order reflects relative contribution, not academic seniority.*

---

## License

- **Paper + governance documents**: [CC BY 4.0](./LICENSE)
- **Linked code repositories**: AGPL-3.0-or-later (see those repos individually)

---

## Disclosure boundary

| Tier | Status | Scope |
|---|---|---|
| **L1** Platform skeleton | ✅ Open (AGPL-3.0) | Service architecture, gateway, audit, state space, redactor |
| **L2** Training methodology | ✅ Open (AGPL-3.0 + CC BY 4.0) | 5-class data pipeline, 3 quality gates, bucketed evaluator |
| **L3** Production artifacts | ❌ Closed | Production LoRA weights, clinical adapters, partner medical data, commercial license terms, device authentication keys |

L3 is held under separate commercial / clinical governance by **DOORM AI PTE. LTD.**

---

## Contact

- **Corresponding author**: service@doorm.ai
- **Issues**: [GitHub Issues](https://github.com/doorm-ai/WorldModel-OS-governance/issues)
- **Security disclosure**: see [SECURITY.md](./SECURITY.md)

---

## Three release gates (per §11 of the paper)

No artifact is considered *officially released* by DOORM AI until all three gates are passed:

1. **Reproducibility gate** — A third party can run `run_dataset_v1.py → train → eval` on a single consumer GPU (RTX 3090 / 4090, 24 GB VRAM) within 24 hours, producing a v1 adapter matching the reference `adapter_version`.
2. **Benchmark gate** — At least one bucketed metric in the domain evaluation (TCM + rehabilitation robotics) with a pre-registered rollback threshold.
3. **Real-adoption gate** — At least one MOU or paid deployment with a medical institution. (Partner identities not disclosed in this draft.)

Until all three are passed, every claim in this paper is, by our own framing, **falsifiable conjecture** — not validated science.
