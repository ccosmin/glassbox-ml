// glassbox -- command-line interface.
//
//   glassbox generate --kind houses|faults --out FILE [--rows N] [--seed S]
//   glassbox train    --task regression|classification --data FILE --target COL [options]
//   glassbox predict  --model FILE --data FILE [--explain] [--top K] [--limit N]
//   glassbox verify   --run DIR [--data FILE]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "glassbox/dataset.hpp"
#include "glassbox/explain.hpp"
#include "glassbox/format.hpp"
#include "glassbox/json.hpp"
#include "glassbox/metrics.hpp"
#include "glassbox/pipeline.hpp"
#include "glassbox/report.hpp"
#include "glassbox/run.hpp"
#include "glassbox/sha256.hpp"
#include "glassbox/synthetic.hpp"
#include "glassbox/version.hpp"

namespace {

constexpr const char* kUsage = R"(glassbox - transparent, auditable linear models

Usage:
  glassbox generate --kind houses|faults --out FILE [--rows 1000] [--seed 7]
      Write a synthetic dataset with a known ground truth.

  glassbox train --task regression|classification --data FILE --target COLUMN
                 [--solver gd|closed-form] [--learning-rate 0.1] [--epochs 5000]
                 [--l2 0] [--tolerance 1e-10] [--test-ratio 0.2] [--seed 42]
                 [--repeats 5] [--out DIR | --out-dir runs]
      Train, evaluate and document a model. Artifacts go to DIR, or to
      <out-dir>/<run-id> where run-id is a hash of the inputs.

  glassbox predict --model FILE --data FILE [--explain] [--top 3] [--limit N]
      Score a CSV (it must contain the model's feature columns) and print CSV
      to stdout. --explain adds the top additive contributions per row.

  glassbox verify --run DIR [--data FILE]
      Check data and model integrity against the run's manifest, then retrain
      from the recorded spec and confirm the model is reproduced bit for bit.

  glassbox version
)";

class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Minimal `--key value` / `--flag` parser with unknown-option detection.
class Args {
 public:
  Args(int argc, char** argv, int first, std::set<std::string> flags, std::set<std::string> options) {
    for (int i = first; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("--", 0) != 0) throw UsageError("unexpected argument '" + arg + "'");
      const std::string key = arg.substr(2);
      if (flags.count(key) != 0) {
        values_[key] = "true";
      } else if (options.count(key) != 0) {
        if (i + 1 >= argc) throw UsageError("option --" + key + " needs a value");
        values_[key] = argv[++i];
      } else {
        throw UsageError("unknown option --" + key);
      }
    }
  }

  bool has(const std::string& key) const { return values_.count(key) != 0; }

  std::string get(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }

  std::string required(const std::string& key) const {
    if (!has(key)) throw UsageError("missing required option --" + key);
    return values_.at(key);
  }

  double number(const std::string& key, double fallback) const {
    if (!has(key)) return fallback;
    const std::string& text = values_.at(key);
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (text.empty() || end != text.c_str() + text.size() || !std::isfinite(value)) {
      throw UsageError("--" + key + " expects a number, got '" + text + "'");
    }
    return value;
  }

  std::uint64_t count(const std::string& key, std::uint64_t fallback) const {
    if (!has(key)) return fallback;
    const std::string& text = values_.at(key);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
      throw UsageError("--" + key + " expects a non-negative integer, got '" + text + "'");
    }
    try {
      return std::stoull(text);
    } catch (const std::out_of_range&) {
      throw UsageError("--" + key + " is out of range");
    }
  }

 private:
  std::map<std::string, std::string> values_;
};

int cmd_generate(const Args& args) {
  const std::string kind = args.required("kind");
  const std::string out = args.required("out");
  const auto rows = static_cast<std::size_t>(args.count("rows", 1000));
  gbx::Rng rng(args.count("seed", 7));

  gbx::Dataset data;
  if (kind == "houses") {
    data = gbx::make_house_prices(rows, rng);
  } else if (kind == "faults") {
    data = gbx::make_machine_faults(rows, rng);
  } else {
    throw UsageError("--kind must be 'houses' or 'faults'");
  }
  gbx::write_csv(out, data);
  std::cerr << "wrote " << rows << " rows to " << out << " (target column: " << data.target_name
            << ")\n";
  return 0;
}

int cmd_train(const Args& args) {
  gbx::RunSpec spec;
  spec.task = gbx::task_from_string(args.required("task"));
  spec.data_path = args.required("data");
  spec.target = args.required("target");
  spec.solver = args.get("solver", "gd");
  spec.train.learning_rate = args.number("learning-rate", spec.train.learning_rate);
  spec.train.max_epochs = static_cast<std::size_t>(args.count("epochs", spec.train.max_epochs));
  spec.train.l2 = args.number("l2", spec.train.l2);
  spec.train.tolerance = args.number("tolerance", spec.train.tolerance);
  spec.test_ratio = args.number("test-ratio", spec.test_ratio);
  spec.seed = args.count("seed", spec.seed);
  spec.evaluation_repeats = static_cast<std::size_t>(args.count("repeats", spec.evaluation_repeats));
  if (args.has("out") && args.has("out-dir")) throw UsageError("use either --out or --out-dir");

  const gbx::RunResult result = gbx::execute_run(spec);
  const std::string run_id = gbx::compute_run_id(spec, result.data_sha256);
  const std::string dir = args.has("out")
                              ? args.get("out", "")
                              : (std::filesystem::path(args.get("out-dir", "runs")) / run_id).string();
  gbx::write_run_artifacts(dir, spec, result);

  const std::string primary = gbx::primary_metric_name(spec.task);
  std::cout << "run " << run_id << ": " << gbx::to_string(spec.task) << " on " << result.rows_train
            << " training rows, " << result.rows_test << " held out\n";
  if (spec.solver == "gd") {
    std::cout << "  gradient descent: " << result.history.epochs_run() << " epochs"
              << (result.history.converged ? " (converged)" : " (max epochs reached, not converged)")
              << "\n";
  } else {
    std::cout << "  solver: closed-form normal equations\n";
  }
  if (result.history.cost_increased) {
    std::cout << "  warning: cost increased during training; consider a smaller --learning-rate\n";
  }
  std::cout << "  test " << primary << ": "
            << gbx::format_fixed(result.test_metrics.at(primary).as_number())
            << "  (train: " << gbx::format_fixed(result.train_metrics.at(primary).as_number()) << ")\n";
  std::cout << "  permutation importance:";
  for (const auto& f : result.importance) {
    std::cout << " " << f.feature << "=" << gbx::format_fixed(f.mean_drop, 3);
  }
  std::cout << "\n  artifacts: " << dir
            << "/{model.json, manifest.json, model_card.md, cost_history.csv}\n";
  return 0;
}

int cmd_predict(const Args& args) {
  const gbx::Pipeline pipeline = gbx::Pipeline::load(args.required("model"));
  const gbx::CsvTable table = gbx::read_csv(args.required("data"));
  const gbx::Matrix X = gbx::select_columns(table, pipeline.feature_names);
  const bool explain = args.has("explain");
  const auto top = static_cast<std::size_t>(args.count("top", 3));
  const auto limit = static_cast<std::size_t>(args.count("limit", X.rows()));
  const bool classification = pipeline.task() == gbx::Task::Classification;

  std::cout << "row," << (classification ? "probability,label" : "prediction")
            << (explain ? ",baseline,top_contributions" : "") << "\n";
  for (std::size_t i = 0; i < std::min(limit, X.rows()); ++i) {
    const gbx::Explanation e = gbx::explain_prediction(pipeline, X.row(i));
    std::cout << i << "," << gbx::format_fixed(e.output, 6);
    if (classification) std::cout << "," << (e.output >= pipeline.threshold ? 1 : 0);
    if (explain) {
      std::cout << "," << gbx::format_fixed(e.baseline, 6) << ",\"" << gbx::summarize(e, top)
                << "\"";
    }
    std::cout << "\n";
  }
  return 0;
}

int cmd_verify(const Args& args) {
  namespace fs = std::filesystem;
  const fs::path dir(args.required("run"));
  const gbx::json::Value manifest = gbx::json::load_file((dir / "manifest.json").string());
  gbx::RunSpec spec = gbx::RunSpec::from_json(manifest.at("spec"));
  if (args.has("data")) spec.data_path = args.get("data", "");

  bool ok = true;
  const auto report = [&ok](const std::string& check, bool passed, const std::string& detail) {
    std::cout << (passed ? "  [ok]   " : "  [FAIL] ") << check << (detail.empty() ? "" : ": ")
              << detail << "\n";
    ok = ok && passed;
  };

  std::cout << "verifying run " << manifest.at("run_id").as_string() << " in " << dir.string()
            << "\n";

  const std::string expected_data = manifest.at("data").at("sha256").as_string();
  const std::string actual_data = gbx::sha256_file(spec.data_path);
  report("data fingerprint", actual_data == expected_data,
         spec.data_path + (actual_data == expected_data ? "" : " has changed since training"));

  const std::string expected_model = manifest.at("artifacts").at("model.json").as_string();
  const std::string stored_model = gbx::sha256_file((dir / "model.json").string());
  report("model.json integrity", stored_model == expected_model,
         stored_model == expected_model ? "" : "file was modified after training");

  if (actual_data != expected_data) {
    report("reproduction", false, "skipped because the data does not match");
  } else {
    const gbx::RunResult rerun = gbx::execute_run(spec);
    const std::string reproduced = gbx::sha256_hex(rerun.pipeline.serialize());
    report("bit-for-bit reproduction", reproduced == expected_model,
           reproduced == expected_model ? "retrained model is identical"
                                        : "retrained model differs (different platform or build?)");
  }
  std::cout << (ok ? "verification passed\n" : "verification FAILED\n");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << kUsage;
    return 2;
  }
  const std::string command = argv[1];
  try {
    if (command == "help" || command == "--help" || command == "-h") {
      std::cout << kUsage;
      return 0;
    }
    if (command == "version" || command == "--version") {
      std::cout << "glassbox " << GLASSBOX_VERSION << " (" << GLASSBOX_GIT_COMMIT << ")\n";
      return 0;
    }
    if (command == "generate") {
      return cmd_generate(Args(argc, argv, 2, {}, {"kind", "out", "rows", "seed"}));
    }
    if (command == "train") {
      return cmd_train(Args(argc, argv, 2, {},
                            {"task", "data", "target", "solver", "learning-rate", "epochs", "l2",
                             "tolerance", "test-ratio", "seed", "repeats", "out", "out-dir"}));
    }
    if (command == "predict") {
      return cmd_predict(Args(argc, argv, 2, {"explain"}, {"model", "data", "top", "limit"}));
    }
    if (command == "verify") {
      return cmd_verify(Args(argc, argv, 2, {}, {"run", "data"}));
    }
    throw UsageError("unknown command '" + command + "'");
  } catch (const UsageError& e) {
    std::cerr << "glassbox: " << e.what() << "\n\n" << kUsage;
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "glassbox: error: " << e.what() << "\n";
    return 1;
  }
}
