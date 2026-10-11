#include "glassbox/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gbx {

namespace {

void check_sizes(const Vector& a, const Vector& b) {
  if (a.size() != b.size()) throw std::invalid_argument("metrics: size mismatch");
  if (a.empty()) throw std::invalid_argument("metrics: empty input");
}

double safe_ratio(double num, double den) { return den > 0.0 ? num / den : 0.0; }

}  // namespace

RegressionMetrics regression_metrics(const Vector& y_true, const Vector& y_pred) {
  check_sizes(y_true, y_pred);
  const auto n = static_cast<double>(y_true.size());
  const double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / n;
  double sse = 0.0, sae = 0.0, sst = 0.0;
  for (std::size_t i = 0; i < y_true.size(); ++i) {
    const double r = y_pred[i] - y_true[i];
    sse += r * r;
    sae += std::abs(r);
    sst += (y_true[i] - mean) * (y_true[i] - mean);
  }
  RegressionMetrics m;
  m.mse = sse / n;
  m.rmse = std::sqrt(m.mse);
  m.mae = sae / n;
  m.r2 = sst > 0.0 ? 1.0 - sse / sst : std::nan("");
  return m;
}

double roc_auc(const Vector& y_true, const Vector& score) {
  check_sizes(y_true, score);
  const std::size_t n = y_true.size();
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return score[a] < score[b]; });

  double positive_rank_sum = 0.0;
  std::size_t positives = 0;
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i;
    while (j + 1 < n && score[order[j + 1]] == score[order[i]]) ++j;
    const double average_rank = (static_cast<double>(i + j) / 2.0) + 1.0;  // 1-based
    for (std::size_t k = i; k <= j; ++k) {
      if (y_true[order[k]] >= 0.5) {
        positive_rank_sum += average_rank;
        ++positives;
      }
    }
    i = j + 1;
  }
  const std::size_t negatives = n - positives;
  if (positives == 0 || negatives == 0) return std::nan("");
  const auto p = static_cast<double>(positives);
  const auto q = static_cast<double>(negatives);
  return (positive_rank_sum - p * (p + 1.0) / 2.0) / (p * q);
}

ClassificationMetrics classification_metrics(const Vector& y_true, const Vector& probability,
                                             double threshold) {
  check_sizes(y_true, probability);
  ClassificationMetrics m;
  ConfusionMatrix& c = m.confusion;
  double loss = 0.0;
  constexpr double kEps = 1e-15;
  for (std::size_t i = 0; i < y_true.size(); ++i) {
    const bool actual = y_true[i] >= 0.5;
    const bool predicted = probability[i] >= threshold;
    if (actual && predicted) ++c.tp;
    else if (!actual && predicted) ++c.fp;
    else if (!actual && !predicted) ++c.tn;
    else ++c.fn;
    const double p = std::clamp(probability[i], kEps, 1.0 - kEps);
    loss -= actual ? std::log(p) : std::log(1.0 - p);
  }
  const auto n = static_cast<double>(y_true.size());
  const auto tp = static_cast<double>(c.tp);
  m.accuracy = static_cast<double>(c.tp + c.tn) / n;
  m.precision = safe_ratio(tp, tp + static_cast<double>(c.fp));
  m.recall = safe_ratio(tp, tp + static_cast<double>(c.fn));
  m.f1 = safe_ratio(2.0 * m.precision * m.recall, m.precision + m.recall);
  m.roc_auc = roc_auc(y_true, probability);
  m.log_loss = loss / n;
  return m;
}

std::string primary_metric_name(Task task) {
  return task == Task::Regression ? "r2" : "roc_auc";
}

double primary_score(Task task, const Vector& y_true, const Vector& prediction) {
  return task == Task::Regression ? regression_metrics(y_true, prediction).r2
                                  : roc_auc(y_true, prediction);
}

json::Value to_json(const RegressionMetrics& m) {
  json::Value v = json::Value::object();
  v.set("mse", m.mse);
  v.set("rmse", m.rmse);
  v.set("mae", m.mae);
  v.set("r2", m.r2);
  return v;
}

json::Value to_json(const ClassificationMetrics& m) {
  json::Value confusion = json::Value::object();
  confusion.set("tp", m.confusion.tp);
  confusion.set("fp", m.confusion.fp);
  confusion.set("tn", m.confusion.tn);
  confusion.set("fn", m.confusion.fn);

  json::Value v = json::Value::object();
  v.set("accuracy", m.accuracy);
  v.set("precision", m.precision);
  v.set("recall", m.recall);
  v.set("f1", m.f1);
  v.set("roc_auc", m.roc_auc);
  v.set("log_loss", m.log_loss);
  v.set("confusion_matrix", confusion);
  return v;
}

json::Value evaluate(Task task, const Vector& y_true, const Vector& prediction, double threshold) {
  return task == Task::Regression ? to_json(regression_metrics(y_true, prediction))
                                  : to_json(classification_metrics(y_true, prediction, threshold));
}

}  // namespace gbx
