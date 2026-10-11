# glassbox

**Transparent, auditable linear models in modern C++17, with no dependencies.**

glassbox implements linear regression and logistic regression from first principles. Around them it builds the engineering that trustworthy AI needs: exact per-prediction explanations, model-agnostic feature importance, robustness probes, and an audit trail for every training run. From that trail you can check data integrity and reproduce the model **bit for bit**.

The model is simple on purpose. The project is about what surrounds it. You should be able to answer *what data was this trained on, why did it predict that, how fragile is it, and can I get exactly the same model again?* with evidence instead of guesswork.

```text
CSV data ──► SHA-256 fingerprint ──► seeded split ──► standardise (train stats only)
         ──► gradient descent / normal equations ──► metrics ──► explanations ──► robustness
         ──► model.json + manifest.json + model_card.md + cost_history.csv
```

---

## Features

**Learning algorithms** (`include/glassbox/model.hpp`)
- One `LinearModel` class covers both linear regression (squared-error cost) and logistic regression (log-loss). With these cost functions the gradient has the same form for both tasks: `dJ/dw = 1/m Σ (f(x) − y)·x + (λ/m)·w`.
- Batch gradient descent with L2 regularisation (the bias is not penalised). It detects convergence, flags a rising cost (learning rate too high), and stops with a clear error if training diverges.
- Closed-form ridge solution through the normal equations, solved with a hand-written Cholesky factorisation. It is used to cross-check gradient descent in the tests.
- Numerically stable sigmoid and log-loss (computed with softplus), so extreme inputs never produce `NaN` or `inf`.

**Data handling** (`include/glassbox/dataset.hpp`)
- Strict CSV parser. Every error names the file, line and column. Missing or non-numeric values are rejected, never silently imputed.
- Seeded train/test split, and a z-score `Standardizer` fitted **only on training data**, which avoids test-set leakage.

**Evaluation** (`metrics.hpp`)
- Regression: MSE, RMSE, MAE, R².
- Classification: accuracy, precision, recall, F1, log-loss, confusion matrix, and ROC AUC (Mann–Whitney U, with correct handling of ties).

**Explainability** (`explain.hpp`, `evaluation.hpp`)
- **Exact local explanations.** On standardised inputs, `prediction = bias + Σ wⱼ·zⱼ`, so each prediction splits *exactly* into a baseline plus one additive contribution per feature. These are true contributions, not post-hoc approximations; for classifiers they are in log-odds units.
- **Global explanations.** Standardised coefficients ("the effect of +1 standard deviation") plus **permutation importance**, which works for any model and is measured on the held-out set.

**Robustness** (`evaluation.hpp`)
- Measures how the score degrades under Gaussian input noise. Noise is scaled per feature by its training standard deviation, so the levels are comparable across features with different units.

**Traceability and reproducibility** (`run.hpp`, `report.hpp`)
- Every dataset is fingerprinted with SHA-256 (a self-contained FIPS 180-4 implementation, tested against the official vectors).
- **Content-addressed run IDs.** The ID is a hash of the run specification plus the data bytes, so identical inputs always get the same ID.
- The manifest records the full spec, the data hash, the build (version, git commit, compiler, build type), metrics, explanations, robustness results, and the hashes of all artifacts.
- `glassbox verify` re-checks data and model integrity, then **retrains from the manifest and confirms the model is identical byte for byte**.
- **Portable randomness.** `std::uniform_real_distribution`, `std::normal_distribution` and `std::shuffle` return different values on libstdc++, libc++ and MSVC. glassbox uses only the raw `std::mt19937_64` stream, whose output the standard fully specifies, and implements its own uniform, Box–Muller normal and Fisher–Yates shuffle on top of it.
- Doubles are saved in their shortest exact round-trip form, so saving and reloading a model never changes a single bit.

**Engineering**
- C++17, CMake, zero third-party dependencies (including the JSON library and the test harness).
- Builds cleanly under `-Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Werror`.
- Optional AddressSanitizer + UndefinedBehaviorSanitizer build (`-DGLASSBOX_SANITIZE=ON`).
- Unit tests (gradients checked by finite differences, gradient descent against closed form, recovery of known generating weights, metric test cases, determinism) plus end-to-end tests that drive the real CLI, including a check that tampered data is detected.

---

## Quick start

Requirements: a C++17 compiler (GCC 9+, Clang 10+, Apple Clang, or MSVC 2019+) and CMake ≥ 3.16.

```sh
git clone <this repository> glassbox-ml
cd glassbox-ml
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The CLI is built at `build/glassbox`.

### 1. Generate data with a known ground truth

```sh
./build/glassbox generate --kind houses --rows 1000 --seed 7 --out houses.csv
./build/glassbox generate --kind faults --rows 1000 --seed 7 --out faults.csv
```

| Dataset | Task | Target | Generating process |
|---|---|---|---|
| `houses` | regression | `price_keur` | `50 + 2.8·area + 12·bedrooms − 1.1·age − 4.5·distance + N(0, 20²)` |
| `faults` | classification | `fault` | `logit = −1 + 0.25(temp−70) + 1.2(vibration−3) − 0.6(pressure−6) + 0.08(hours−20)` |

Both datasets include a `noise_feature` that has **no effect** on the target. A trustworthy pipeline should give it near-zero importance, and the model card lets you check that.

### 2. Train

```sh
# Logistic regression with gradient descent
./build/glassbox train --task classification --data faults.csv --target fault \
    --learning-rate 0.5 --l2 1 --seed 42

# Linear regression with the exact normal-equation solver
./build/glassbox train --task regression --data houses.csv --target price_keur \
    --solver closed-form --l2 0.1
```

Each run writes its artifacts to `runs/<run-id>/` (or to the directory you pass with `--out DIR`):

| File | Contents |
|---|---|
| `model.json` | The deployable pipeline: feature list, scaling statistics, weights, bias, threshold. Deterministic content. |
| `manifest.json` | Run spec, data SHA-256, build info, train/test metrics, explanations, robustness results, artifact hashes. |
| `model_card.md` | A human-readable model card generated from the same information. |
| `cost_history.csv` | The learning curve (cost per epoch). |

Training options:

| Option | Default | Meaning |
|---|---|---|
| `--solver gd\|closed-form` | `gd` | Gradient descent, or exact normal equations (regression only) |
| `--learning-rate` | `0.1` | Gradient descent step size |
| `--epochs` | `5000` | Maximum number of epochs |
| `--tolerance` | `1e-10` | Stop when the relative change in cost falls below this |
| `--l2` | `0` | L2 regularisation strength λ |
| `--test-ratio` | `0.2` | Fraction of rows held out for evaluation |
| `--seed` | `42` | Seed for the split and the evaluation |
| `--repeats` | `5` | Repeats for the permutation-importance and noise analyses |

### 3. Predict and explain

```sh
./build/glassbox predict --model runs/<run-id>/model.json --data faults.csv \
    --explain --top 3 --limit 10
```

The output is CSV on stdout. With `--explain`, each row gets its baseline and its largest additive contributions, for example `vibration_mm_s:+1.234; temperature_c:+0.871; pressure_bar:-0.112`. The input CSV only needs the model's feature columns; any other columns are ignored.

### 4. Verify a run

```sh
./build/glassbox verify --run runs/<run-id>
```

```text
verifying run <run-id> in runs/<run-id>
  [ok]   data fingerprint: faults.csv
  [ok]   model.json integrity
  [ok]   bit-for-bit reproduction: retrained model is identical
verification passed
```

Verification fails (exit code 1) if the data has changed since training, if `model.json` was edited, or if retraining produces a different model. `--data FILE` points verification at a relocated copy of the dataset.

---

## Relation to the EU AI Act

Regulation (EU) 2024/1689 (the [AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)) requires high-risk AI systems to handle data governance, documentation, logging, transparency, accuracy and robustness properly. glassbox is a small, concrete illustration of how those obligations turn into engineering practice:

| AI Act requirement | How glassbox addresses it |
|---|---|
| Art. 10, data and data governance | Strict validation of the input data. The dataset is fingerprinted with SHA-256. Scaling statistics come from training data only and are documented per feature. |
| Art. 11 and Annex IV, technical documentation | An auto-generated `model_card.md`: model type, training procedure, data, performance, limitations. |
| Art. 12, record-keeping | `manifest.json` records every input, parameter, build detail and output hash of a run. |
| Art. 13, transparency | Exact per-prediction contributions, standardised coefficients, permutation importance. |
| Art. 15, accuracy, robustness | Held-out metrics, plus a measured noise-sensitivity curve in every run. |
| Reproducibility (supports all of the above) | Content-addressed run IDs, portable RNG, lossless serialisation, and `glassbox verify` retraining. |

> **Disclaimer:** this is an educational and portfolio project. It is not a compliance tool, and using it does not make a system compliant with the AI Act.

---

## Project layout

```text
include/glassbox/   public headers (one module per file, documented)
  matrix.hpp          dense matrix, dot product, Cholesky solver
  random.hpp          seeded RNG with portable distributions
  json.hpp            minimal order-preserving JSON value, parser and writer
  sha256.hpp          SHA-256 (FIPS 180-4)
  dataset.hpp         CSV parsing, datasets, train/test split, standardisation
  model.hpp           LinearModel: linear and logistic regression
  pipeline.hpp        scaler + model + feature contract, (de)serialisation
  metrics.hpp         regression and classification metrics, ROC AUC
  explain.hpp         exact additive explanations, standardised weights
  evaluation.hpp      permutation importance, noise robustness
  synthetic.hpp       datasets with known ground truth
  run.hpp             run specification and end-to-end execution
  report.hpp          manifest, model card, build provenance
src/                implementations
apps/glassbox_cli.cpp   command-line interface
tests/              unit tests, minimal test harness, end-to-end CLI test (e2e.cmake)
cmake/              generated version header template
```

### Using it as a library

```cpp
#include "glassbox/dataset.hpp"
#include "glassbox/explain.hpp"
#include "glassbox/pipeline.hpp"

gbx::Dataset data = gbx::to_dataset(gbx::read_csv("faults.csv"), "fault");

gbx::Pipeline p;
p.feature_names = data.feature_names;
p.target_name = data.target_name;
p.scaler.fit(data.X);
p.model = gbx::LinearModel(gbx::Task::Classification, data.num_features());
p.model.fit(p.scaler.transform(data.X), data.y, {/*learning_rate*/ 0.5, /*max_epochs*/ 5000});

gbx::Explanation why = gbx::explain_prediction(p, data.X.row(0));
// why.output == P(fault | x); why.baseline + Σ contributions == why.decision
```

---

## Design decisions

- **Why linear models?** They are the models whose behaviour can be explained *exactly*, which makes them the right baseline before reaching for anything more complex. When an opaque model is used, a strong, well-documented linear baseline is also how you show that the extra complexity is justified.
- **Why no dependencies?** Every line that affects a prediction can be read and audited in this repository. The project builds in an air-gapped environment, and there is no supply-chain surface. A real system would use Eigen or BLAS for speed; the interfaces here are small enough to swap that in.
- **Why hash everything?** A model is only reproducible if its exact inputs are known. Hashes turn "I think this is the data we used" into a check that either passes or fails.
- **Known limitation:** bit-for-bit reproduction is guaranteed on the same platform and toolchain. Across platforms, `std::log`, `std::exp` and the compiler's floating-point code generation can differ in the last bit, so `verify` reports a mismatch rather than silently accepting it.

## Background

I built this project to put into practice, in C++, the supervised-learning foundations from my Coursera certificate in regression and classification: cost functions, gradient descent, feature scaling, regularisation, and the logistic model. I extended them with the evaluation, explainability and traceability practices that trustworthy AI systems need.

## Roadmap

- Multiclass (softmax) regression and polynomial feature expansion
- Mini-batch / stochastic gradient descent with learning-rate schedules
- Fairness metrics across groups (demographic parity, equalised odds)
- Data-drift detection between the training statistics and new data
- k-fold cross-validation and hyperparameter search recorded in the manifest

## License

[MIT](LICENSE) © 2026 Cosmin Cremarenco
