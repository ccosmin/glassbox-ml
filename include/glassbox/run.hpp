#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "glassbox/evaluation.hpp"
#include "glassbox/json.hpp"
#include "glassbox/model.hpp"
#include "glassbox/pipeline.hpp"

namespace gbx {

/// Complete, serialisable description of a training run. Together with the
/// dataset bytes it determines the trained model exactly: re-executing a
/// spec on the same data reproduces model.json bit for bit.
struct RunSpec {
  Task task = Task::Regression;
  std::string data_path;
  std::string target;
  double test_ratio = 0.2;
  std::uint64_t seed = 42;
  std::string solver = "gd";  ///< "gd" (gradient descent) or "closed-form" (regression only)
  TrainConfig train;
  std::size_t evaluation_repeats = 5;
  std::vector<double> noise_levels = {0.0, 0.1, 0.25, 0.5, 1.0};

  json::Value to_json() const;
  static RunSpec from_json(const json::Value& value);
  /// Equivalent `glassbox train ...` command line.
  std::string command_line() const;
};

struct RunResult {
  Pipeline pipeline;
  TrainHistory history;
  std::string data_sha256;
  std::size_t rows_train = 0;
  std::size_t rows_test = 0;
  json::Value train_metrics;
  json::Value test_metrics;
  std::vector<FeatureImportance> importance;
  std::vector<NoisePoint> robustness;
};

/// Loads data, splits, scales, trains, evaluates and probes the model.
RunResult execute_run(const RunSpec& spec);

/// Content-addressed run identifier: a hash of the spec and the data bytes
/// (not the file path), so identical inputs always yield the same ID.
std::string compute_run_id(const RunSpec& spec, const std::string& data_sha256);

}  // namespace gbx
