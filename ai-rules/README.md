# ai-rules

Working knowledge for anyone (human or AI assistant) changing this repository.
Read the file that matches your task before writing code.

| File | Read it when you need to know |
| --- | --- |
| [01-system-architecture.md](01-system-architecture.md) | What the whole system is, what each layer owns, what is built so far |
| [02-preprocessing-methodology.md](02-preprocessing-methodology.md) | What the research paper prescribes for Layer 1, step by step, and where the paper is silent or inconsistent |
| [03-layer1-implementation.md](03-layer1-implementation.md) | What Layer 1 does, feature by feature, and why each decision was made |
| [04-tech-stack.md](04-tech-stack.md) | Languages, libraries, versions, licences, external services |
| [05-project-structure.md](05-project-structure.md) | Where every file is and what it is responsible for |
| [06-data-flow-and-integration.md](06-data-flow-and-integration.md) | Inputs, outputs, database schema, API, and how later layers plug in |
| [07-configuration-and-testing.md](07-configuration-and-testing.md) | Every setting, how to run things, what the tests cover, measured results |
| [08-limitations-and-pending.md](08-limitations-and-pending.md) | Known weaknesses, assumptions, and the open to-do list |
| [09-invoice-workflow.md](09-invoice-workflow.md) | Nanonets OCR, invoice extraction, validation, judge, review, and live observations |

## Ground rules

1. **Build layer by layer.** Inspect, test, and document each stage. Current
   state: preprocessing, OCR integration, an invoice workflow, and a React
   review interface exist; accuracy evaluation and production hardening remain.
2. **The paper and the architecture diagram are the specification.** When
   something is needed that neither covers, implement the smallest reasonable
   thing, mark it in code as an "engineering default", and record it in
   `08-limitations-and-pending.md`. Never present an unsupported choice as if
   it came from the paper.
3. **Keep the pipeline framework-free.** Everything under
   `backend/preprocessing/pipeline/` must not import Django. Django code calls
   the pipeline, never the reverse.
4. **Every tunable goes in `PreprocessingConfig`.** No magic numbers in
   stage code unless they are internal constants with a comment explaining
   them.
5. **No real documents in the repository.** Uploaded bills contain personal
   and medical data. Tests use synthetic pages from
   `backend/preprocessing/tests/helpers.py`.
6. **Report measurements, not expectations.** Numbers in these files were
   measured on this codebase; update them when behaviour changes.
7. **Update these files in the same change** that alters behaviour, structure,
   configuration or limitations.

## Source documents

- **Paper:** Z. A. M. Nather and H. M. Ahmed, "Designing a Modular Framework
  for Processing and Enhancing Scanned Documents Using Advanced Denoising
  Algorithms", *Journal of Al-Qadisiyah for Computer Science and Mathematics*,
  vol. 18(1), 2026, pp. Comp 160-172. DOI 10.29304/jqcsm.2026.18.12441.
- **Architecture diagram:** supplied by the project owner; transcribed in
  `01-system-architecture.md`.
