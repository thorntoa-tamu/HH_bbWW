# HH_bbWW — instructions for Copilot code review

The HH→bb̄WW analysis, built on [FLAF](https://github.com/cms-flaf/FLAF). It differs from the other
analyses in using DeepHME for mass reconstruction and in running the statistical inference chain
(datacards and limits) as part of its integration test.

**Read `FLAF/.github/copilot-instructions.md` first.** It carries the framework invariants — law
task semantics, bundles, remote-storage caching, processor stages, concurrency — and the rules on
what a useful comment looks like and what not to flag. The rule that documentation ships in the same PR applies here too, and is restated below with the pages that matter for this repository. Everything there applies here. This file
adds only what is specific to this analysis.

## Analysis-specific invariants

### Stitching processors

- The shared anchors in `config/processes.yaml` (`.DY_processors`,
  `.DY_processors_allFlavors`, …) differ **only** in the stitching config file they point at.
  Every one of them must declare `stages: [ AnaTuple, AnaTupleMerge ]`. A stitcher present at
  `AnaTuple` alone writes an anaCache denominator that nothing can combine, and every merge of
  every process using that anchor dies with
  `combineAnaCaches: processor Stitcher not provided for combining anaCaches`. This shipped
  undetected in the 2024–2026 configs because no CI process ran DY.
- Which anchor an era's DY process uses is deliberate — `allFlavors` for 2022–2023BPix, the
  single-flavour one for 2024 onwards. Do not propose harmonising them.

### The CI datacard

`config/Datacards/CI_card.yaml` drives the `test_multi_era` job. Two things about it:

- It declares `eras: Run3_2022 … Run3_2023BPix` only. A datacard task run for a later era fails on
  a missing signal histogram; that is the card's scope, not a bug.
- **The DY background is deliberately not in it.** The bbWW DY samples are amc@NLO, so they carry
  negative event weights, and under `--test 1000` too few events remain for those to cancel: the
  histogram comes out negative but statistically compatible with zero. `resolveNegativeBins`
  returns as soon as the integral is negative, so including it would need
  `allow_negative_integral: true` — disabling a real safety check — and would be flaky either
  way. Do not suggest adding it back, and treat any new `allow_negative_*` flag as needing an
  explicit justification.

### Integration test

`TestModel` runs `custom_CI_Background_TT` and `custom_CI_Background_DY` plus one signal and one
data process, and each CI background must carry the same `processors:` as the real `TT` / DY
process **of that era** — that is what exercises the stitching end to end. A diff that changes a
real process's processors and leaves the CI counterpart behind silently removes the coverage.

The process names are also listed in `cms-flaf/FLAF_ci`, a **different repository**; renaming or
adding one here needs that updated in step.

### Cost

`AnalysisCacheTask` (BtagShape) runs before `HistTupleProducerTask` even for simple variables and
dominates the runtime. A change that adds work to the per-branch path there is expensive; say so.

## Documentation must ship with the change

A PR must update the documentation **in the same PR** whenever it changes anything a user of the
framework can observe. Treat this as a review item of the same weight as correctness — docs
drifting from the code is the failure that motivated the current documentation, and a PR that
lands without them is not complete.

Ask, for every diff: does it add, rename or remove any of these?

- a task or DAG node, or the arguments/parameters of one;
- a command, a CLI flag, or the meaning of an existing one;
- a configuration key — `global.yaml`, `user_custom.yaml`, `processes.yaml`, `phys_models.yaml`,
  cross-sections, `fs_*` storage keys, bundle flavours, processor entries;
- a dataset, era, process or physics-model name;
- the environment, installation or setup steps;
- storage locations, output paths or log locations;
- a CI workflow, or how the integration test is triggered or configured;
- any behaviour a user relies on, including a default that changes.

If the answer is yes and the diff touches **no** documentation file, say so and name the page that
should have changed. If the author states the change is internal-only, that is a legitimate
answer — a pure refactor or bugfix with no user-visible effect is exempt — but it should be
stated in the PR, not left implicit.

Also flag the inverse: documentation edited to describe behaviour the diff does not implement, and
new pages added without being wired into `mkdocs.yml`'s `nav` (the build fails on that, but the
review should catch it first).

Where it goes:

- `docs/` in this repository for analysis-specific material (`analysis.md`, `setup.md`, `stat_inference.md`).
- **`FLAF/docs/` for anything framework-wide.** If the change alters shared behaviour, the
  documentation belongs there, in a companion PR to `cms-flaf/FLAF` — flag that it is missing
  rather than accepting an analysis-local description of a framework change.
- New pages must be added to `nav:` in `mkdocs.yml`; verified with `mkdocs build --strict`.

## Repository facts

Verified 2026-08-27; re-check before relying on any of it.

| | |
|---|---|
| Layout | `AnaProd/` (`anaTupleDef.py`, `baseline.py`), `Analysis/` (`hh_bbww.py`, `histTupleDef.py`, `hh_bbWW_AnaCacheProducer.py`, DeepHME/DNN producers), `Studies/`, `config/`, `include/`, `docs/` |
| Submodules | `FLAF`, `Corrections`, `StatInference`, `inference`, `DeepHME`, `SyncTool` |
| Eras | Run 3: 2022, 2022EE, 2023, 2023BPix, 2024, 2025, 2026 |
| Configs | `config/global.yaml`, `config/processes.yaml` (processor anchors), `config/phys_models.yaml`, `config/<era>/{datasets,processes}.yaml`, `config/Datacards/CI_card.yaml` |
| Large files | DNN `.onnx` payloads are tracked with Git LFS; never commit a binary directly |
| Tests | No unit tests in this repo; validation is the integration test and physics checks. The framework's suites live in `FLAF/test/` |
| Workflows | `formatting-check`, `repo-sanity-checks`, `test-setup-loading`, `deploy-docs`, `trigger-flaf-integration`. Formatting and era loading are checked automatically — do not comment on them |
| Integration test | Triggered by `@cms-flaf-bot please test`; its configuration lives in `cms-flaf/FLAF_ci`, **not** in this repo. There is no `.github/integration_cfg.yaml` here |
| Docs | `docs/`, plus the shared framework docs in `FLAF/docs/` |
