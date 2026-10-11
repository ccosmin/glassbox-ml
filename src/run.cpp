#include "glassbox/run.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "glassbox/dataset.hpp"
#include "glassbox/format.hpp"
#include "glassbox/metrics.hpp"
#include "glassbox/random.hpp"
#include "glassbox/sha256.hpp"

namespace gbx {

namespace {

// Independent stream for evaluation so that changing the number of repeats
// never alters the train/test split.
constexpr std::uint64_t kEvaluationStream = 0x9E3779B97F4A7C15ULL;

std::string read_file_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

}  // namespace

json::Value RunSpec::to_json() const {
  json::Value training = json::Value::object();
  training.set("solver", solver);
  training.set("learning_rate", train.learning_rate);
  training.set("max_epochs", train.max_epochs);
  training.set("l2", train.l2);
  training.set("tolerance", train.tolerance);

  json::Value v = json::Value::object();
  v.set("task", to_string(task));
  v.set("data_path", data_path);
  v.set("target", target);
  v.set("test_ratio", test_ratio);
  v.set("seed", std::to_string(seed));  // string: JSON numbers lose precision above 2^53
  v.set("training", training);
  v.set("evaluation_repeats", evaluation_repeats);
  v.set("noise_levels", json::from_vector(noise_levels));
  return v;
}

RunSpec RunSpec::from_json(const json::Value& value) {
  RunSpec s;
  s.task = task_from_string(value.at("task").as_string());
  s.data_path = value.at("data_path").as_string();
  s.target = value.at("target").as_string();
  s.test_ratio = value.at("test_ratio").as_number();
  s.seed = std::stoull(value.at("seed").as_string());
  const json::Value& training = value.at("training");
  s.solver = training.at("solver").as_string();
  s.train.learning_rate = training.at("learning_rate").as_number();
  s.train.max_epochs = static_cast<std::size_t>(training.at("max_epochs").as_number());
  s.train.l2 = training.at("l2").as_number();
  s.train.tolerance = training.at("tolerance").as_number();
  s.evaluation_repeats = static_cast<std::size_t>(value.at("evaluation_repeats").as_number());
  s.noise_levels = json::to_vector(value.at("noise_levels"));
  return s;
}

std::string RunSpec::command_line() const {
  std::string cmd = "glassbox train --task " + to_string(task) + " --data " + data_path +
                    " --target " + target + " --solver " + solver;
  if (solver == "gd") {
    cmd += " --learning-rate " + format_double(train.learning_rate) +
           " --epochs " + std::to_string(train.max_epochs) +
           " --tolerance " + format_double(train.tolerance);
  }
  cmd += " --l2 " + format_double(train.l2) + " --test-ratio " + format_double(test_ratio) +
         " --seed " + std::to_string(seed);
  return cmd;
}

RunResult execute_run(const RunSpec& spec) {
  RunResult r;

  // Hash and parse the same in-memory bytes, so the recorded fingerprint is
  // guaranteed to describe exactly the data that was trained on.
  const std::string bytes = read_file_bytes(spec.data_path);
  r.data_sha256 = sha256_hex(bytes);
  std::istringstream stream(bytes);
  const Dataset data = to_dataset(parse_csv(stream, spec.data_path), spec.target);

  Rng split_rng(spec.seed);
  const Split split = train_test_split(data, spec.test_ratio, split_rng);
  r.rows_train = split.train.size();
  r.rows_test = split.test.size();

  Pipeline& p = r.pipeline;
  p.feature_names = data.feature_names;
  p.target_name = data.target_name;
  p.scaler.fit(split.train.X);  // training statistics only: no test-set leakage
  const Matrix train_scaled = p.scaler.transform(split.train.X);
  p.model = LinearModel(spec.task, data.num_features());

  if (spec.solver == "gd") {
    r.history = p.model.fit(train_scaled, split.train.y, spec.train);
  } else if (spec.solver == "closed-form") {
    p.model.fit_closed_form(train_scaled, split.train.y, spec.train.l2);
    r.history.cost = {p.model.cost(train_scaled, split.train.y, spec.train.l2)};
    r.history.converged = true;
  } else {
    throw std::invalid_argument("unknown solver '" + spec.solver +
                                "' (expected 'gd' or 'closed-form')");
  }

  r.train_metrics = evaluate(spec.task, split.train.y, p.predict(split.train.X), p.threshold);
  r.test_metrics = evaluate(spec.task, split.test.y, p.predict(split.test.X), p.threshold);

  Rng eval_rng(spec.seed ^ kEvaluationStream);
  r.importance = permutation_importance(p, split.test, spec.evaluation_repeats, eval_rng);
  r.robustness =
      noise_sensitivity(p, split.test, spec.noise_levels, spec.evaluation_repeats, eval_rng);
  return r;
}

std::string compute_run_id(const RunSpec& spec, const std::string& data_sha256) {
  json::Value identity = spec.to_json();
  identity.set("data_path", "sha256:" + data_sha256);
  return sha256_hex(identity.dump(0)).substr(0, 12);
}

}  // namespace gbx
