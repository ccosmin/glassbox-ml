#pragma once

#include <cstddef>
#include <istream>
#include <string>
#include <vector>

#include "glassbox/matrix.hpp"
#include "glassbox/random.hpp"

namespace gbx {

/// A numeric CSV file: a header row followed by rows of numbers.
struct CsvTable {
  std::vector<std::string> header;
  std::vector<Vector> rows;
};

/// Parses numeric CSV. Errors name the source, line and column, because a
/// silently mis-parsed value is a data-quality defect that propagates into
/// the model. Missing values are rejected rather than imputed.
CsvTable parse_csv(std::istream& in, const std::string& source_name);
CsvTable read_csv(const std::string& path);

/// Features plus a target column.
struct Dataset {
  std::vector<std::string> feature_names;
  std::string target_name;
  Matrix X;
  Vector y;

  std::size_t size() const noexcept { return y.size(); }
  std::size_t num_features() const noexcept { return feature_names.size(); }
};

Dataset to_dataset(const CsvTable& table, const std::string& target);

/// Extracts the named columns (in that order) from a table, e.g. to score new
/// data with a saved model whose feature list is known.
Matrix select_columns(const CsvTable& table, const std::vector<std::string>& names);

void write_csv(const std::string& path, const Dataset& data);

Dataset subset(const Dataset& data, const std::vector<std::size_t>& indices);

struct Split {
  Dataset train;
  Dataset test;
};

/// Random hold-out split. Deterministic for a given Rng state.
Split train_test_split(const Dataset& data, double test_ratio, Rng& rng);

/// Z-score feature scaling. Fit on training data only, then apply the same
/// statistics to test and production data to avoid information leakage.
class Standardizer {
 public:
  Standardizer() = default;
  Standardizer(Vector mean, Vector scale);

  void fit(const Matrix& X);
  Matrix transform(const Matrix& X) const;
  void transform_row(const double* in, double* out) const;

  const Vector& mean() const noexcept { return mean_; }
  const Vector& scale() const noexcept { return scale_; }
  std::size_t size() const noexcept { return mean_.size(); }

 private:
  Vector mean_;
  Vector scale_;
};

}  // namespace gbx
