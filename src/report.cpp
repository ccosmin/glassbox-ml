#include "glassbox/report.hpp"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "glassbox/explain.hpp"
#include "glassbox/format.hpp"
#include "glassbox/metrics.hpp"
#include "glassbox/sha256.hpp"
#include "glassbox/version.hpp"

#ifndef GLASSBOX_BUILD_TYPE
#define GLASSBOX_BUILD_TYPE "unknown"
#endif

namespace gbx {

namespace {

std::string compiler_name() {
#if defined(__clang__)
  return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
  return "msvc " + std::to_string(_MSC_VER);
#else
  return "unknown";
#endif
}

void write_text(const std::filesystem::path& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write '" + path.string() + "'");
  out << content;
  if (!out) throw std::runtime_error("failed writing '" + path.string() + "'");
}

std::string cost_history_csv(const TrainHistory& history) {
  std::string out = "epoch,cost\n";
  for (std::size_t i = 0; i < history.cost.size(); ++i) {
    out += std::to_string(i) + "," + format_double(history.cost[i]) + "\n";
  }
  return out;
}

json::Value importance_json(const std::vector<FeatureImportance>& importance) {
  json::Value out = json::Value::array();
  for (const auto& f : importance) {
    json::Value item = json::Value::object();
    item.set("feature", f.feature);
    item.set("mean_score_drop", f.mean_drop);
    item.set("std_score_drop", f.std_drop);
    out.push_back(item);
  }
  return out;
}

json::Value robustness_json(const std::vector<NoisePoint>& points) {
  json::Value out = json::Value::array();
  for (const auto& p : points) {
    json::Value item = json::Value::object();
    item.set("noise_level", p.noise_level);
    item.set("mean_score", p.mean_score);
    item.set("std_score", p.std_score);
    out.push_back(item);
  }
  return out;
}

json::Value weights_json(const Pipeline& pipeline) {
  json::Value out = json::Value::array();
  for (const auto& w : standardized_weights(pipeline)) {
    json::Value item = json::Value::object();
    item.set("feature", w.feature);
    item.set("weight", w.weight);
    out.push_back(item);
  }
  return out;
}

std::string metric_row(const std::string& name, const json::Value& train, const json::Value& test) {
  return "| " + name + " | " + format_fixed(train.at(name).as_number()) + " | " +
         format_fixed(test.at(name).as_number()) + " |\n";
}

}  // namespace

json::Value build_info() {
  json::Value info = json::Value::object();
  info.set("glassbox_version", GLASSBOX_VERSION);
  info.set("git_commit", GLASSBOX_GIT_COMMIT);
  info.set("compiler", compiler_name());
  info.set("cplusplus", static_cast<double>(__cplusplus));
  info.set("build_type", GLASSBOX_BUILD_TYPE);
  return info;
}

std::string utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  char buffer[32];
  std::strftime(buffer, sizeof buffer, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  return buffer;
}

json::Value write_run_artifacts(const std::string& directory, const RunSpec& spec,
                                const RunResult& result) {
  namespace fs = std::filesystem;
  const fs::path dir(directory);
  fs::create_directories(dir);

  const std::string model_text = result.pipeline.serialize();
  const std::string history_text = cost_history_csv(result.history);
  write_text(dir / "model.json", model_text);
  write_text(dir / "cost_history.csv", history_text);

  json::Value data = json::Value::object();
  data.set("path", spec.data_path);
  data.set("sha256", result.data_sha256);
  data.set("rows_train", result.rows_train);
  data.set("rows_test", result.rows_test);
  data.set("features", json::from_strings(result.pipeline.feature_names));
  data.set("target", result.pipeline.target_name);

  json::Value training = json::Value::object();
  training.set("epochs_run", result.history.epochs_run());
  training.set("converged", result.history.converged);
  training.set("cost_increased", result.history.cost_increased);
  training.set("final_cost", result.history.cost.empty() ? 0.0 : result.history.cost.back());

  json::Value metrics = json::Value::object();
  metrics.set("primary", primary_metric_name(spec.task));
  metrics.set("train", result.train_metrics);
  metrics.set("test", result.test_metrics);

  json::Value explainability = json::Value::object();
  explainability.set("standardized_weights", weights_json(result.pipeline));
  explainability.set("bias", result.pipeline.model.bias());
  explainability.set("permutation_importance", importance_json(result.importance));

  json::Value robustness = json::Value::object();
  robustness.set("method", "additive gaussian noise, level x training std-dev per feature");
  robustness.set("results", robustness_json(result.robustness));

  json::Value artifacts = json::Value::object();
  artifacts.set("model.json", sha256_hex(model_text));
  artifacts.set("cost_history.csv", sha256_hex(history_text));

  json::Value manifest = json::Value::object();
  manifest.set("format", "glassbox.manifest");
  manifest.set("format_version", 1);
  manifest.set("run_id", compute_run_id(spec, result.data_sha256));
  manifest.set("created_utc", utc_timestamp());
  manifest.set("build", build_info());
  manifest.set("spec", spec.to_json());
  manifest.set("data", data);
  manifest.set("training", training);
  manifest.set("metrics", metrics);
  manifest.set("explainability", explainability);
  manifest.set("robustness", robustness);
  manifest.set("artifacts", artifacts);

  write_text(dir / "manifest.json", manifest.dump(2) + "\n");
  write_text(dir / "model_card.md", render_model_card(spec, result, manifest));
  return manifest;
}

std::string render_model_card(const RunSpec& spec, const RunResult& result,
                              const json::Value& manifest) {
  const Pipeline& p = result.pipeline;
  const bool regression = spec.task == Task::Regression;
  const json::Value& build = manifest.at("build");
  const std::string primary = primary_metric_name(spec.task);
  std::ostringstream md;

  md << "# Model card: " << (regression ? "linear regression" : "logistic regression")
     << " for `" << p.target_name << "`\n\n";
  md << "> Generated automatically by glassbox " << build.at("glassbox_version").as_string()
     << " (commit `" << build.at("git_commit").as_string() << "`) on "
     << manifest.at("created_utc").as_string() << ".  \n"
     << "> Run ID `" << manifest.at("run_id").as_string()
     << "`. Every number below is also in `manifest.json`.\n\n";

  md << "## Model details\n\n";
  md << "| | |\n|---|---|\n";
  md << "| Task | " << to_string(spec.task) << " |\n";
  md << "| Model | " << (regression ? "f(x) = w . z + b" : "P(y=1 | x) = sigmoid(w . z + b)")
     << ", z = standardised x |\n";
  md << "| Solver | " << (spec.solver == "gd" ? "batch gradient descent" : "normal equations (Cholesky)")
     << " |\n";
  if (spec.solver == "gd") {
    md << "| Learning rate / max epochs | " << format_double(spec.train.learning_rate) << " / "
       << spec.train.max_epochs << " |\n";
    md << "| Epochs run | " << result.history.epochs_run()
       << (result.history.converged ? " (converged)" : " (**did not converge**)") << " |\n";
  }
  md << "| L2 regularisation (lambda) | " << format_double(spec.train.l2) << " |\n";
  md << "| Inputs | " << p.feature_names.size() << " numeric features |\n";
  if (!regression) md << "| Decision threshold | " << format_double(p.threshold) << " |\n";
  md << "\n";
  if (result.history.cost_increased) {
    md << "> **Warning:** the training cost increased during at least one epoch, which usually "
          "means the learning rate is too high.\n\n";
  }

  md << "## Intended use and limitations\n\n"
     << "This model was produced by an educational, open-source pipeline. It is a linear "
        "model: it assumes each feature has a constant, additive effect on the "
     << (regression ? "target" : "log-odds of the positive class")
     << ", and it cannot capture interactions or non-linear effects unless they are added as "
        "features. Performance is only known for data resembling the training distribution "
        "documented below. It must not be used for decisions about people without a proper "
        "risk assessment, bias evaluation and human oversight.\n\n";

  md << "## Training data\n\n";
  md << "| | |\n|---|---|\n";
  md << "| Source file | `" << spec.data_path << "` |\n";
  md << "| SHA-256 | `" << result.data_sha256 << "` |\n";
  md << "| Rows (train / test) | " << result.rows_train << " / " << result.rows_test << " |\n";
  md << "| Split | random, test ratio " << format_double(spec.test_ratio) << ", seed "
     << spec.seed << " |\n";
  md << "| Target | `" << p.target_name << "` |\n\n";
  md << "| Feature | Training mean | Training std-dev |\n|---|---:|---:|\n";
  for (std::size_t j = 0; j < p.feature_names.size(); ++j) {
    md << "| `" << p.feature_names[j] << "` | " << format_fixed(p.scaler.mean()[j]) << " | "
       << format_fixed(p.scaler.scale()[j]) << " |\n";
  }
  md << "\n";

  md << "## Performance\n\n| Metric | Train | Test (held out) |\n|---|---:|---:|\n";
  const json::Value& train = result.train_metrics;
  const json::Value& test = result.test_metrics;
  if (regression) {
    for (const char* name : {"r2", "rmse", "mae"}) md << metric_row(name, train, test);
  } else {
    for (const char* name : {"roc_auc", "accuracy", "precision", "recall", "f1", "log_loss"}) {
      md << metric_row(name, train, test);
    }
    const json::Value& cm = test.at("confusion_matrix");
    md << "\nTest confusion matrix (rows = actual, columns = predicted):\n\n"
       << "| | predicted 0 | predicted 1 |\n|---|---:|---:|\n"
       << "| **actual 0** | " << format_double(cm.at("tn").as_number()) << " | "
       << format_double(cm.at("fp").as_number()) << " |\n"
       << "| **actual 1** | " << format_double(cm.at("fn").as_number()) << " | "
       << format_double(cm.at("tp").as_number()) << " |\n";
  }
  md << "\n";

  md << "## Explainability\n\n";
  md << "### Coefficients\n\n"
     << "Weights apply to standardised features, so each one is the change in "
     << (regression ? "the prediction" : "log-odds") << " for a one-standard-deviation increase "
     << "in that feature, holding the others fixed. Baseline (bias) = "
     << format_fixed(p.model.bias()) << ".\n\n";
  md << "| Feature | Weight | Direction |\n|---|---:|---|\n";
  for (const auto& w : standardized_weights(p)) {
    md << "| `" << w.feature << "` | " << format_fixed(w.weight) << " | "
       << (w.weight > 0 ? "increases" : w.weight < 0 ? "decreases" : "none") << " |\n";
  }
  md << "\n### Permutation importance (test set, " << spec.evaluation_repeats << " repeats)\n\n"
     << "Drop in " << primary << " when the feature is randomly shuffled. Values near zero "
     << "mean the model does not rely on the feature.\n\n";
  md << "| Feature | Mean drop | Std |\n|---|---:|---:|\n";
  for (const auto& f : result.importance) {
    md << "| `" << f.feature << "` | " << format_fixed(f.mean_drop) << " | "
       << format_fixed(f.std_drop) << " |\n";
  }
  md << "\nPer-prediction explanations: `glassbox predict --model model.json --data <csv> "
        "--explain`. They decompose each prediction exactly into the baseline plus one "
        "additive contribution per feature.\n\n";

  md << "## Robustness\n\n"
     << "Test-set " << primary << " under additive Gaussian noise, scaled per feature by its "
     << "training standard deviation (" << spec.evaluation_repeats << " repeats).\n\n";
  md << "| Noise level (x std-dev) | Mean " << primary << " | Std |\n|---:|---:|---:|\n";
  for (const auto& point : result.robustness) {
    md << "| " << format_double(point.noise_level) << " | " << format_fixed(point.mean_score)
       << " | " << format_fixed(point.std_score) << " |\n";
  }
  md << "\n";

  md << "## Reproducibility and traceability\n\n"
     << "- Build: " << build.at("compiler").as_string() << ", "
     << build.at("build_type").as_string() << " build.\n"
     << "- `model.json` SHA-256: `" << manifest.at("artifacts").at("model.json").as_string()
     << "`\n"
     << "- Re-run with:\n\n  ```sh\n  " << spec.command_line() << "\n  ```\n\n"
     << "- Check data integrity and bit-for-bit reproducibility of this run:\n\n"
     << "  ```sh\n  glassbox verify --run <this directory>\n  ```\n";
  return md.str();
}

}  // namespace gbx
