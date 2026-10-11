#include "glassbox/dataset.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "glassbox/format.hpp"

namespace gbx {

namespace {

std::string trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::vector<std::string> split_fields(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = line.find(',', start);
    fields.push_back(trim(line.substr(start, comma - start)));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return fields;
}

// std::strtod honours the C locale; glassbox never calls setlocale, so '.' is
// always the decimal separator.
bool parse_number(const std::string& text, double& value) {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  value = std::strtod(text.c_str(), &end);
  return end == text.c_str() + text.size() && errno != ERANGE && std::isfinite(value);
}

std::size_t column_index(const std::vector<std::string>& header, const std::string& name) {
  const auto it = std::find(header.begin(), header.end(), name);
  if (it == header.end()) {
    std::string available;
    for (const auto& h : header) available += (available.empty() ? "" : ", ") + h;
    throw std::invalid_argument("column '" + name + "' not found (available: " + available + ")");
  }
  return static_cast<std::size_t>(it - header.begin());
}

}  // namespace

CsvTable parse_csv(std::istream& in, const std::string& source_name) {
  CsvTable table;
  std::string line;
  std::size_t line_number = 0;

  while (std::getline(in, line)) {
    ++line_number;
    if (!trim(line).empty()) {
      table.header = split_fields(trim(line));
      break;
    }
  }
  if (table.header.empty()) throw std::runtime_error(source_name + ": file is empty");
  for (std::size_t i = 0; i < table.header.size(); ++i) {
    if (table.header[i].empty()) {
      throw std::runtime_error(source_name + ": header column " + std::to_string(i + 1) +
                               " has no name");
    }
    if (std::count(table.header.begin(), table.header.end(), table.header[i]) > 1) {
      throw std::runtime_error(source_name + ": duplicate column '" + table.header[i] + "'");
    }
  }

  while (std::getline(in, line)) {
    ++line_number;
    if (trim(line).empty()) continue;
    const std::vector<std::string> fields = split_fields(trim(line));
    const std::string where = source_name + ":" + std::to_string(line_number);
    if (fields.size() != table.header.size()) {
      throw std::runtime_error(where + ": expected " + std::to_string(table.header.size()) +
                               " fields, found " + std::to_string(fields.size()));
    }
    Vector row(fields.size());
    for (std::size_t i = 0; i < fields.size(); ++i) {
      if (!parse_number(fields[i], row[i])) {
        throw std::runtime_error(where + ": column '" + table.header[i] + "': '" + fields[i] +
                                 "' is not a finite number");
      }
    }
    table.rows.push_back(std::move(row));
  }
  if (table.rows.empty()) throw std::runtime_error(source_name + ": no data rows");
  return table;
}

CsvTable read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  return parse_csv(in, path);
}

Dataset to_dataset(const CsvTable& table, const std::string& target) {
  const std::size_t target_index = column_index(table.header, target);
  if (table.header.size() < 2) throw std::invalid_argument("dataset has no feature columns");

  Dataset data;
  data.target_name = target;
  for (std::size_t i = 0; i < table.header.size(); ++i) {
    if (i != target_index) data.feature_names.push_back(table.header[i]);
  }
  data.X = Matrix(table.rows.size(), data.feature_names.size());
  data.y.resize(table.rows.size());
  for (std::size_t r = 0; r < table.rows.size(); ++r) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < table.header.size(); ++i) {
      if (i == target_index) {
        data.y[r] = table.rows[r][i];
      } else {
        data.X(r, c++) = table.rows[r][i];
      }
    }
  }
  return data;
}

Matrix select_columns(const CsvTable& table, const std::vector<std::string>& names) {
  std::vector<std::size_t> indices;
  for (const auto& name : names) indices.push_back(column_index(table.header, name));
  Matrix out(table.rows.size(), names.size());
  for (std::size_t r = 0; r < table.rows.size(); ++r) {
    for (std::size_t c = 0; c < indices.size(); ++c) out(r, c) = table.rows[r][indices[c]];
  }
  return out;
}

void write_csv(const std::string& path, const Dataset& data) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write '" + path + "'");
  for (const auto& name : data.feature_names) out << name << ',';
  out << data.target_name << '\n';
  for (std::size_t r = 0; r < data.size(); ++r) {
    for (std::size_t c = 0; c < data.num_features(); ++c) out << format_double(data.X(r, c)) << ',';
    out << format_double(data.y[r]) << '\n';
  }
}

Dataset subset(const Dataset& data, const std::vector<std::size_t>& indices) {
  Dataset out;
  out.feature_names = data.feature_names;
  out.target_name = data.target_name;
  out.X = Matrix(indices.size(), data.num_features());
  out.y.resize(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const std::size_t src = indices[i];
    if (src >= data.size()) throw std::out_of_range("subset: index out of range");
    std::copy(data.X.row(src), data.X.row(src) + data.num_features(), out.X.row(i));
    out.y[i] = data.y[src];
  }
  return out;
}

Split train_test_split(const Dataset& data, double test_ratio, Rng& rng) {
  if (!(test_ratio > 0.0 && test_ratio < 1.0)) {
    throw std::invalid_argument("test ratio must be in (0, 1), got " + format_double(test_ratio));
  }
  std::vector<std::size_t> indices(data.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  rng.shuffle(indices);

  const auto n_test =
      static_cast<std::size_t>(std::llround(test_ratio * static_cast<double>(data.size())));
  if (n_test == 0 || n_test >= data.size()) {
    throw std::invalid_argument("not enough rows (" + std::to_string(data.size()) +
                                ") for a train/test split");
  }
  const std::vector<std::size_t> test_idx(indices.begin(),
                                          indices.begin() + static_cast<std::ptrdiff_t>(n_test));
  const std::vector<std::size_t> train_idx(indices.begin() + static_cast<std::ptrdiff_t>(n_test),
                                           indices.end());
  return Split{subset(data, train_idx), subset(data, test_idx)};
}

Standardizer::Standardizer(Vector mean, Vector scale) : mean_(std::move(mean)), scale_(std::move(scale)) {
  if (mean_.size() != scale_.size()) {
    throw std::invalid_argument("Standardizer: mean and scale sizes differ");
  }
  for (const double s : scale_) {
    if (!(s > 0.0)) throw std::invalid_argument("Standardizer: scale must be positive");
  }
}

void Standardizer::fit(const Matrix& X) {
  if (X.rows() == 0) throw std::invalid_argument("Standardizer::fit: no rows");
  const auto n = static_cast<double>(X.rows());
  mean_.assign(X.cols(), 0.0);
  scale_.assign(X.cols(), 0.0);
  for (std::size_t r = 0; r < X.rows(); ++r) {
    for (std::size_t c = 0; c < X.cols(); ++c) mean_[c] += X(r, c);
  }
  for (double& m : mean_) m /= n;
  for (std::size_t r = 0; r < X.rows(); ++r) {
    for (std::size_t c = 0; c < X.cols(); ++c) {
      const double d = X(r, c) - mean_[c];
      scale_[c] += d * d;
    }
  }
  for (double& s : scale_) {
    s = std::sqrt(s / n);
    if (s < 1e-12) s = 1.0;  // constant column: centre it but do not divide by ~0
  }
}

void Standardizer::transform_row(const double* in, double* out) const {
  for (std::size_t c = 0; c < mean_.size(); ++c) out[c] = (in[c] - mean_[c]) / scale_[c];
}

Matrix Standardizer::transform(const Matrix& X) const {
  if (X.cols() != mean_.size()) {
    throw std::invalid_argument("Standardizer: expected " + std::to_string(mean_.size()) +
                                " columns, got " + std::to_string(X.cols()));
  }
  Matrix out(X.rows(), X.cols());
  for (std::size_t r = 0; r < X.rows(); ++r) transform_row(X.row(r), out.row(r));
  return out;
}

}  // namespace gbx
