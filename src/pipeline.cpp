#include "glassbox/pipeline.hpp"

#include <stdexcept>

namespace gbx {

namespace {
constexpr const char* kFormat = "glassbox.pipeline";
constexpr int kFormatVersion = 1;
}  // namespace

double Pipeline::predict_one(const double* raw_x) const {
  Vector scaled(scaler.size());
  scaler.transform_row(raw_x, scaled.data());
  return model.predict_one(scaled.data());
}

Vector Pipeline::predict(const Matrix& raw_X) const { return model.predict(scaler.transform(raw_X)); }

Vector Pipeline::labels(const Vector& probabilities) const {
  if (task() != Task::Classification) throw std::logic_error("labels() requires a classifier");
  Vector out(probabilities.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = probabilities[i] >= threshold ? 1.0 : 0.0;
  return out;
}

json::Value Pipeline::to_json() const {
  json::Value scaling = json::Value::object();
  scaling.set("mean", json::from_vector(scaler.mean()));
  scaling.set("scale", json::from_vector(scaler.scale()));

  json::Value params = json::Value::object();
  params.set("weights", json::from_vector(model.weights()));
  params.set("bias", model.bias());

  json::Value root = json::Value::object();
  root.set("format", kFormat);
  root.set("format_version", kFormatVersion);
  root.set("task", to_string(task()));
  root.set("target", target_name);
  root.set("features", json::from_strings(feature_names));
  root.set("standardizer", scaling);
  root.set("model", params);
  if (task() == Task::Classification) root.set("threshold", threshold);
  return root;
}

Pipeline Pipeline::from_json(const json::Value& value) {
  if (!value.contains("format") || value.at("format").as_string() != kFormat) {
    throw std::runtime_error("not a glassbox model file");
  }
  const auto version = value.at("format_version").as_number();
  if (version != kFormatVersion) {
    throw std::runtime_error("unsupported model format version " + std::to_string(version));
  }

  Pipeline p;
  const Task task = task_from_string(value.at("task").as_string());
  p.target_name = value.at("target").as_string();
  p.feature_names = json::to_strings(value.at("features"));
  const json::Value& scaling = value.at("standardizer");
  p.scaler = Standardizer(json::to_vector(scaling.at("mean")), json::to_vector(scaling.at("scale")));
  const json::Value& params = value.at("model");
  Vector weights = json::to_vector(params.at("weights"));
  if (weights.size() != p.feature_names.size() || p.scaler.size() != p.feature_names.size()) {
    throw std::runtime_error("model file is inconsistent: feature, scaler and weight counts differ");
  }
  p.model = LinearModel(task, weights.size());
  p.model.set_parameters(std::move(weights), params.at("bias").as_number());
  if (task == Task::Classification) p.threshold = value.at("threshold").as_number();
  return p;
}

Pipeline Pipeline::load(const std::string& path) {
  try {
    return from_json(json::load_file(path));
  } catch (const std::exception& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

}  // namespace gbx
