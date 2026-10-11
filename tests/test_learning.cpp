// Tests for data handling, models, metrics, explanations and reproducibility.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "glassbox/dataset.hpp"
#include "glassbox/evaluation.hpp"
#include "glassbox/explain.hpp"
#include "glassbox/metrics.hpp"
#include "glassbox/model.hpp"
#include "glassbox/pipeline.hpp"
#include "glassbox/run.hpp"
#include "glassbox/synthetic.hpp"
#include "test_framework.hpp"

using namespace gbx;

namespace {

CsvTable csv(const std::string& text) {
  std::istringstream in(text);
  return parse_csv(in, "inline.csv");
}

Matrix random_matrix(std::size_t rows, std::size_t cols, Rng& rng) {
  Matrix X(rows, cols);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) X(i, j) = rng.normal();
  return X;
}

// Finite-difference check of the analytic gradient.
void check_gradient(Task task) {
  Rng rng(3);
  const Matrix X = random_matrix(40, 3, rng);
  Vector y(40);
  for (double& v : y) v = task == Task::Regression ? rng.normal() : (rng.uniform() < 0.5 ? 0.0 : 1.0);
  LinearModel model(task, 3);
  model.set_parameters({0.3, -0.7, 0.2}, 0.1);
  const double l2 = 0.5;

  Vector grad_w;
  double grad_b = 0.0;
  model.gradient(X, y, l2, grad_w, grad_b);

  const double h = 1e-6;
  for (std::size_t j = 0; j < 3; ++j) {
    Vector plus = model.weights(), minus = model.weights();
    plus[j] += h;
    minus[j] -= h;
    LinearModel mp = model, mm = model;
    mp.set_parameters(plus, model.bias());
    mm.set_parameters(minus, model.bias());
    const double numeric = (mp.cost(X, y, l2) - mm.cost(X, y, l2)) / (2 * h);
    CHECK_NEAR(grad_w[j], numeric, 1e-6);
  }
  LinearModel bp = model, bm = model;
  bp.set_parameters(model.weights(), model.bias() + h);
  bm.set_parameters(model.weights(), model.bias() - h);
  CHECK_NEAR(grad_b, (bp.cost(X, y, l2) - bm.cost(X, y, l2)) / (2 * h), 1e-6);
}

std::string temp_csv(const Dataset& data, const std::string& name) {
  const auto path = std::filesystem::temp_directory_path() / ("glassbox_test_" + name + ".csv");
  write_csv(path.string(), data);
  return path.string();
}

}  // namespace

// ---------------------------------------------------------------- data

TEST_CASE("csv: parses numeric data and builds a dataset") {
  const CsvTable t = csv("a, b ,y\n1,2,3\n\n4,5.5,-6e1\n");
  CHECK_EQ(t.rows.size(), std::size_t{2});
  const Dataset d = to_dataset(t, "y");
  CHECK(d.feature_names == std::vector<std::string>({"a", "b"}));
  CHECK_EQ(d.X(1, 1), 5.5);
  CHECK_EQ(d.y[1], -60.0);
  const Matrix m = select_columns(t, {"b", "a"});
  CHECK_EQ(m(0, 0), 2.0);
  CHECK_EQ(m(0, 1), 1.0);
}

TEST_CASE("csv: bad input produces precise errors") {
  CHECK_THROWS(csv(""));
  CHECK_THROWS(csv("a,b\n"));
  CHECK_THROWS(csv("a,a\n1,2\n"));
  CHECK_THROWS(csv("a,b\n1\n"));
  CHECK_THROWS(csv("a,b\n1,\n"));
  CHECK_THROWS(csv("a,b\n1,abc\n"));
  CHECK_THROWS(csv("a,b\n1,nan\n"));
  CHECK_THROWS(to_dataset(csv("a,b\n1,2\n"), "missing"));
  try {
    csv("a,b\n1,2\n3,x\n");
    CHECK(false);
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("inline.csv:3") != std::string::npos);
  }
}

TEST_CASE("split: deterministic, disjoint and complete") {
  Rng gen(1);
  const Dataset d = make_house_prices(100, gen);
  Rng r1(5), r2(5);
  const Split a = train_test_split(d, 0.25, r1);
  const Split b = train_test_split(d, 0.25, r2);
  CHECK_EQ(a.test.size(), std::size_t{25});
  CHECK_EQ(a.train.size(), std::size_t{75});
  CHECK(a.test.y == b.test.y);

  std::multiset<double> all(d.y.begin(), d.y.end());
  std::multiset<double> rejoined(a.train.y.begin(), a.train.y.end());
  rejoined.insert(a.test.y.begin(), a.test.y.end());
  CHECK(all == rejoined);
  CHECK_THROWS(train_test_split(d, 1.0, r1));
}

TEST_CASE("standardizer: zero mean, unit variance, constant columns are safe") {
  Matrix X(4, 2);
  const double values[4][2] = {{1, 5}, {2, 5}, {3, 5}, {4, 5}};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 2; ++j) X(i, j) = values[i][j];
  Standardizer s;
  s.fit(X);
  const Matrix Z = s.transform(X);
  double mean = 0, sq = 0;
  for (int i = 0; i < 4; ++i) {
    mean += Z(i, 0);
    sq += Z(i, 0) * Z(i, 0);
    CHECK_EQ(Z(i, 1), 0.0);
  }
  CHECK_NEAR(mean, 0.0, 1e-12);
  CHECK_NEAR(sq / 4, 1.0, 1e-12);
}

// ---------------------------------------------------------------- models

TEST_CASE("model: sigmoid is numerically stable") {
  CHECK_EQ(sigmoid(0.0), 0.5);
  CHECK(sigmoid(-1000.0) >= 0.0 && sigmoid(-1000.0) < 1e-300);
  CHECK_EQ(sigmoid(1000.0), 1.0);
  CHECK_NEAR(sigmoid(2.0) + sigmoid(-2.0), 1.0, 1e-15);
}

TEST_CASE("model: analytic gradient matches finite differences (regression)") {
  check_gradient(Task::Regression);
}

TEST_CASE("model: analytic gradient matches finite differences (classification)") {
  check_gradient(Task::Classification);
}

TEST_CASE("model: gradient descent converges to the closed-form ridge solution") {
  Rng rng(11);
  const Matrix X = random_matrix(200, 4, rng);
  Vector y(200);
  const double true_w[] = {3.0, -2.0, 0.5, 0.0};
  for (std::size_t i = 0; i < 200; ++i) y[i] = dot(true_w, X.row(i), 4) + 7.0 + rng.normal(0, 0.1);

  for (const double l2 : {0.0, 5.0}) {
    LinearModel exact(Task::Regression, 4);
    exact.fit_closed_form(X, y, l2);
    LinearModel gd(Task::Regression, 4);
    const TrainHistory h = gd.fit(X, y, {0.1, 20000, l2, 1e-14});
    CHECK(h.converged);
    CHECK(!h.cost_increased);
    for (std::size_t j = 0; j < 4; ++j) CHECK_NEAR(gd.weights()[j], exact.weights()[j], 1e-5);
    CHECK_NEAR(gd.bias(), exact.bias(), 1e-5);
  }
  LinearModel exact(Task::Regression, 4);
  exact.fit_closed_form(X, y, 0.0);
  CHECK_NEAR(exact.weights()[0], 3.0, 0.05);
  CHECK_NEAR(exact.bias(), 7.0, 0.05);
}

TEST_CASE("model: logistic regression recovers the generating weights") {
  Rng rng(21);
  const Matrix X = random_matrix(5000, 2, rng);
  Vector y(5000);
  for (std::size_t i = 0; i < 5000; ++i) {
    y[i] = rng.uniform() < sigmoid(2.0 * X(i, 0) - 1.0 * X(i, 1) + 0.5) ? 1.0 : 0.0;
  }
  LinearModel m(Task::Classification, 2);
  const TrainHistory h = m.fit(X, y, {0.5, 20000, 0.0, 1e-12});
  CHECK(h.converged);
  CHECK_NEAR(m.weights()[0], 2.0, 0.2);
  CHECK_NEAR(m.weights()[1], -1.0, 0.15);
  CHECK_NEAR(m.bias(), 0.5, 0.15);
}

TEST_CASE("model: invalid inputs and divergence are reported") {
  LinearModel clf(Task::Classification, 1);
  Matrix X(2, 1, 1.0);
  CHECK_THROWS(clf.fit(X, {0.0, 2.0}, {}));
  CHECK_THROWS(clf.fit_closed_form(X, {0.0, 1.0}, 0.0));
  CHECK_THROWS(clf.fit(X, {0.0}, {}));

  Rng rng(4);
  const Matrix Z = random_matrix(50, 2, rng);
  Vector y(50);
  for (std::size_t i = 0; i < 50; ++i) y[i] = 100.0 * Z(i, 0);
  LinearModel reg(Task::Regression, 2);
  CHECK_THROWS(reg.fit(Z, y, {50.0, 1000, 0.0, 0.0}));  // learning rate far too high
}

// ---------------------------------------------------------------- metrics

TEST_CASE("metrics: regression") {
  const RegressionMetrics perfect = regression_metrics({1, 2, 3}, {1, 2, 3});
  CHECK_EQ(perfect.r2, 1.0);
  CHECK_EQ(perfect.mse, 0.0);
  const RegressionMetrics m = regression_metrics({1, 2, 3, 4}, {2, 2, 3, 2});
  CHECK_NEAR(m.mse, 5.0 / 4.0, 1e-12);
  CHECK_NEAR(m.mae, 3.0 / 4.0, 1e-12);
  CHECK_NEAR(m.r2, 1.0 - 5.0 / 5.0, 1e-12);
  CHECK(std::isnan(regression_metrics({2, 2}, {1, 3}).r2));
}

TEST_CASE("metrics: classification and ROC AUC") {
  const Vector y = {0, 0, 1, 1};
  CHECK_EQ(roc_auc(y, {0.1, 0.4, 0.35, 0.8}), 0.75);  // classic textbook example
  CHECK_EQ(roc_auc(y, {0.5, 0.5, 0.5, 0.5}), 0.5);    // ties
  CHECK_EQ(roc_auc(y, {0.9, 0.8, 0.2, 0.1}), 0.0);
  CHECK(std::isnan(roc_auc({1, 1}, {0.2, 0.3})));

  const ClassificationMetrics m = classification_metrics({1, 1, 1, 0, 0}, {0.9, 0.6, 0.2, 0.7, 0.1});
  CHECK_EQ(m.confusion.tp, std::size_t{2});
  CHECK_EQ(m.confusion.fn, std::size_t{1});
  CHECK_EQ(m.confusion.fp, std::size_t{1});
  CHECK_EQ(m.confusion.tn, std::size_t{1});
  CHECK_NEAR(m.accuracy, 0.6, 1e-12);
  CHECK_NEAR(m.precision, 2.0 / 3.0, 1e-12);
  CHECK_NEAR(m.recall, 2.0 / 3.0, 1e-12);
}

// ---------------------------------------------------------------- explainability

TEST_CASE("explain: contributions add up exactly to the prediction") {
  Rng rng(8);
  const Dataset d = make_machine_faults(500, rng);
  Pipeline p;
  p.feature_names = d.feature_names;
  p.scaler.fit(d.X);
  p.model = LinearModel(Task::Classification, d.num_features());
  p.model.fit(p.scaler.transform(d.X), d.y, {});

  for (std::size_t i = 0; i < 20; ++i) {
    const Explanation e = explain_prediction(p, d.X.row(i));
    double total = e.baseline;
    for (const auto& c : e.contributions) total += c.contribution;
    CHECK_NEAR(total, e.decision, 1e-12);
    CHECK_NEAR(e.output, p.predict_one(d.X.row(i)), 1e-12);
    for (std::size_t k = 1; k < e.contributions.size(); ++k) {
      CHECK(std::abs(e.contributions[k - 1].contribution) >= std::abs(e.contributions[k].contribution));
    }
  }
}

TEST_CASE("evaluation: irrelevant feature is least important; noise hurts") {
  Rng rng(2);
  const Dataset d = make_house_prices(800, rng);
  Pipeline p;
  p.feature_names = d.feature_names;
  p.scaler.fit(d.X);
  p.model = LinearModel(Task::Regression, d.num_features());
  p.model.fit_closed_form(p.scaler.transform(d.X), d.y, 0.0);

  Rng eval(3);
  const auto importance = permutation_importance(p, d, 5, eval);
  CHECK_EQ(importance.front().feature, std::string("area_m2"));
  CHECK_EQ(importance.back().feature, std::string("noise_feature"));
  CHECK_NEAR(importance.back().mean_drop, 0.0, 0.01);

  const auto noise = noise_sensitivity(p, d, {0.0, 0.5, 1.0}, 5, eval);
  CHECK(noise[0].mean_score > noise[1].mean_score);
  CHECK(noise[1].mean_score > noise[2].mean_score);
}

// ---------------------------------------------------------------- reproducibility

TEST_CASE("pipeline: JSON round trip preserves predictions exactly") {
  Rng rng(6);
  const Dataset d = make_machine_faults(300, rng);
  Pipeline p;
  p.feature_names = d.feature_names;
  p.target_name = d.target_name;
  p.scaler.fit(d.X);
  p.model = LinearModel(Task::Classification, d.num_features());
  p.model.fit(p.scaler.transform(d.X), d.y, {});
  p.threshold = 0.4;

  const Pipeline q = Pipeline::from_json(json::Value::parse(p.serialize()));
  CHECK(p.predict(d.X) == q.predict(d.X));
  CHECK_EQ(q.threshold, 0.4);
  CHECK_EQ(q.serialize(), p.serialize());
}

TEST_CASE("run: identical inputs reproduce the model bit for bit") {
  Rng rng(10);
  RunSpec spec;
  spec.task = Task::Classification;
  spec.data_path = temp_csv(make_machine_faults(400, rng), "faults");
  spec.target = "fault";
  spec.evaluation_repeats = 2;

  const RunResult a = execute_run(spec);
  const RunResult b = execute_run(spec);
  CHECK_EQ(a.pipeline.serialize(), b.pipeline.serialize());
  CHECK_EQ(a.test_metrics.dump(), b.test_metrics.dump());
  CHECK_EQ(compute_run_id(spec, a.data_sha256), compute_run_id(spec, b.data_sha256));

  RunSpec other = spec;
  other.seed = 43;
  CHECK(compute_run_id(other, a.data_sha256) != compute_run_id(spec, a.data_sha256));

  const RunSpec restored = RunSpec::from_json(json::Value::parse(spec.to_json().dump()));
  CHECK_EQ(restored.to_json().dump(), spec.to_json().dump());
  std::filesystem::remove(spec.data_path);
}
