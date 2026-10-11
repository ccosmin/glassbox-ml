#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace gbx::json {

class Value;
using Array = std::vector<Value>;
/// Objects keep insertion order so that serialised files are stable and
/// diff-friendly (important when artifacts are hashed and audited).
using Object = std::vector<std::pair<std::string, Value>>;

enum class Type { Null, Bool, Number, String, Array, Object };

/// A small JSON value type: enough to persist models, run specifications and
/// manifests without pulling in a third-party dependency.
class Value {
 public:
  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool b) : type_(Type::Bool), bool_(b) {}
  template <typename T,
            typename = std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>>>
  Value(T number) : type_(Type::Number), number_(static_cast<double>(number)) {}
  Value(const char* s) : type_(Type::String), string_(s) {}
  Value(std::string s) : type_(Type::String), string_(std::move(s)) {}
  Value(Array a) : type_(Type::Array), array_(std::move(a)) {}
  Value(Object o) : type_(Type::Object), object_(std::move(o)) {}

  static Value array() { return Value(Array{}); }
  static Value object() { return Value(Object{}); }

  Type type() const noexcept { return type_; }
  bool is_null() const noexcept { return type_ == Type::Null; }

  bool as_bool() const;
  double as_number() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  /// Object access. `set` replaces an existing key or appends a new one.
  Value& set(const std::string& key, Value value);
  bool contains(const std::string& key) const;
  const Value& at(const std::string& key) const;

  /// Array access.
  void push_back(Value value);

  /// Serialises the value. indent > 0 pretty-prints; arrays holding only
  /// scalars stay on one line to keep weight vectors readable.
  std::string dump(int indent = 2) const;

  static Value parse(const std::string& text);

 private:
  Type type_ = Type::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  Array array_;
  Object object_;
};

Value from_vector(const std::vector<double>& values);
std::vector<double> to_vector(const Value& value);
Value from_strings(const std::vector<std::string>& values);
std::vector<std::string> to_strings(const Value& value);

Value load_file(const std::string& path);
void save_file(const std::string& path, const Value& value);

}  // namespace gbx::json
