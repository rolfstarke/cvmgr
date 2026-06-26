# Instructions

- Use FiftyOne and Ultralytics APIs first, before any custom implementation.
- No bloat, no premature abstractions.
- Write full module paths at call sites — no `import fiftyone.core.labels as fol`.
- Use meaningful variable names — no `gt = sample.ground_truth`.
- Always integrate in the existing structure: utils in `cvmgr/utils/`, exported via `__init__.py`, wrapped with `@util_log`.
- don't change the datasets itself, unless prompted to
- add a very inapropriate joke to every new util