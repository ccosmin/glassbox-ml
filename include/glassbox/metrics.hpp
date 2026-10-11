#pragma once

#include <cstddef>
#include <string>

#include "glassbox/json.hpp"
#include "glassbox/matrix.hpp"
#include "glassbox/model.hpp"

namespace gbx {

struct RegressionMetrics {
  double mse = 0.0;
  double rmse = 0.0;
  double mae = 0.0;
  double r2 = 0.0;  ///< NaN when the targets are constant
};

RegressionMetrics regression_metrics(const Vector& y_true, const Vector& y_pred);

struct ConfusionMatrix {
  std::size_t tp = 0, fp = 0, tn = 0, fn = 0;
};

struct ClassificationMetrics {
  double accuracy = 0.0;
  double precision = 0.0;  ///< 0 when nothing is predicted positive
  double recall = 0.0;     ///< 0 when there are no positives
  double f1 = 0.0;
  double roc_auc = 0.0;    ///< NaN when only one class is present
  double log_loss = 0.0;
  ConfusionMatrix confusion;
};

ClassificationMetrics classification_metrics(const Vector& y_true, const Vector& probability,
                                             double threshold = 0.5);

/// Area under the ROC curve via the Mann-Whitney U statistic (ties get
/// average ranks). Threshold-independent.
double roc_auc(const Vector& y_true, const Vector& score);

/// Headline score used for importance and robustness analyses: R^2 for
/// regression, ROC AUC for classification. Higher is better for both.
std::string primary_metric_name(Task task);
double primary_score(Task task, const Vector& y_true, const Vector& prediction);

json::Value to_json(const RegressionMetrics& m);
json::Value to_json(const ClassificationMetrics& m);
json::Value evaluate(Task task, const Vector& y_true, const Vector& prediction, double threshold);

}  // namespace gbx
