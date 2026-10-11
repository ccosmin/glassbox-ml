#include "glassbox/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "glassbox/format.hpp"

namespace gbx::json {

namespace {

const char* type_name(Type type) {
  switch (type) {
    case Type::Null: return "null";
    case Type::Bool: return "bool";
    case Type::Number: return "number";
    case Type::String: return "string";
    case Type::Array: return "array";
    case Type::Object: return "object";
  }
  return "unknown";
}

void require(Type actual, Type expected) {
  if (actual != expected) {
    throw std::runtime_error(std::string("JSON type error: expected ") + type_name(expected) +
                             ", got " + type_name(actual));
  }
}

void write_string(const std::string& s, std::string& out) {
  out += '"';
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof buffer, "\\u%04x", static_cast<unsigned>(c));
          out += buffer;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

bool is_scalar(const Value& v) { return v.type() != Type::Array && v.type() != Type::Object; }

void write_value(const Value& v, int indent, int depth, std::string& out) {
  const auto newline = [&](int level) {
    if (indent <= 0) return;
    out += '\n';
    out.append(static_cast<std::size_t>(indent * level), ' ');
  };

  switch (v.type()) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += v.as_bool() ? "true" : "false"; break;
    case Type::Number: {
      const double d = v.as_number();
      out += std::isfinite(d) ? format_double(d) : "null";  // JSON has no NaN/Inf
      break;
    }
    case Type::String: write_string(v.as_string(), out); break;
    case Type::Array: {
      const Array& items = v.as_array();
      if (items.empty()) {
        out += "[]";
        break;
      }
      const bool one_line = std::all_of(items.begin(), items.end(), is_scalar);
      out += '[';
      for (std::size_t i = 0; i < items.size(); ++i) {
        if (i > 0) out += (one_line && indent > 0) ? ", " : ",";
        if (!one_line) newline(depth + 1);
        write_value(items[i], indent, depth + 1, out);
      }
      if (!one_line) newline(depth);
      out += ']';
      break;
    }
    case Type::Object: {
      const Object& members = v.as_object();
      if (members.empty()) {
        out += "{}";
        break;
      }
      out += '{';
      for (std::size_t i = 0; i < members.size(); ++i) {
        if (i > 0) out += ',';
        newline(depth + 1);
        write_string(members[i].first, out);
        out += indent > 0 ? ": " : ":";
        write_value(members[i].second, indent, depth + 1, out);
      }
      newline(depth);
      out += '}';
      break;
    }
  }
}

void append_utf8(unsigned code, std::string& out) {
  if (code < 0x80) {
    out += static_cast<char>(code);
  } else if (code < 0x800) {
    out += static_cast<char>(0xC0 | (code >> 6));
    out += static_cast<char>(0x80 | (code & 0x3F));
  } else if (code < 0x10000) {
    out += static_cast<char>(0xE0 | (code >> 12));
    out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (code & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (code >> 18));
    out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (code & 0x3F));
  }
}

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}

  Value parse_document() {
    Value v = parse_value(0);
    skip_whitespace();
    if (pos_ != s_.size()) fail("unexpected trailing characters");
    return v;
  }

 private:
  static constexpr int kMaxDepth = 256;

  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("JSON parse error at offset " + std::to_string(pos_) + ": " + what);
  }

  void skip_whitespace() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
      ++pos_;
    }
  }

  char peek() {
    skip_whitespace();
    if (pos_ >= s_.size()) fail("unexpected end of input");
    return s_[pos_];
  }

  void expect_literal(const char* literal) {
    const std::size_t n = std::strlen(literal);
    if (s_.compare(pos_, n, literal) != 0) fail(std::string("expected '") + literal + "'");
    pos_ += n;
  }

  Value parse_value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    switch (peek()) {
      case '{': return parse_object(depth);
      case '[': return parse_array(depth);
      case '"': return Value(parse_string());
      case 't': expect_literal("true"); return Value(true);
      case 'f': expect_literal("false"); return Value(false);
      case 'n': expect_literal("null"); return Value();
      default: return parse_number();
    }
  }

  Value parse_object(int depth) {
    ++pos_;  // '{'
    Value object = Value::object();
    if (peek() == '}') {
      ++pos_;
      return object;
    }
    while (true) {
      if (peek() != '"') fail("expected string key");
      std::string key = parse_string();
      if (peek() != ':') fail("expected ':'");
      ++pos_;
      object.set(key, parse_value(depth + 1));
      const char c = peek();
      if (c == '}') {
        ++pos_;
        return object;
      }
      if (c != ',') fail("expected ',' or '}'");
      ++pos_;
    }
  }

  Value parse_array(int depth) {
    ++pos_;  // '['
    Value array = Value::array();
    if (peek() == ']') {
      ++pos_;
      return array;
    }
    while (true) {
      array.push_back(parse_value(depth + 1));
      const char c = peek();
      if (c == ']') {
        ++pos_;
        return array;
      }
      if (c != ',') fail("expected ',' or ']'");
      ++pos_;
    }
  }

  unsigned parse_hex4() {
    if (pos_ + 4 > s_.size()) fail("truncated \\u escape");
    unsigned code = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[pos_++];
      code <<= 4;
      if (c >= '0' && c <= '9') code |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') code |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') code |= static_cast<unsigned>(c - 'A' + 10);
      else fail("invalid hex digit in \\u escape");
    }
    return code;
  }

  std::string parse_string() {
    ++pos_;  // opening quote
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) fail("unterminated string");
      const char c = s_[pos_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= s_.size()) fail("unterminated escape");
      const char e = s_[pos_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned code = parse_hex4();
          if (code >= 0xD800 && code <= 0xDBFF && s_.compare(pos_, 2, "\\u") == 0) {
            pos_ += 2;
            const unsigned low = parse_hex4();
            if (low < 0xDC00 || low > 0xDFFF) fail("invalid surrogate pair");
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
          }
          append_utf8(code, out);
          break;
        }
        default: fail("invalid escape sequence");
      }
    }
  }

  Value parse_number() {
    const std::size_t start = pos_;
    while (pos_ < s_.size() && std::strchr("+-0123456789.eE", s_[pos_]) != nullptr) ++pos_;
    if (pos_ == start) fail("unexpected character");
    const std::string token = s_.substr(start, pos_ - start);
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end != token.c_str() + token.size()) fail("invalid number '" + token + "'");
    return Value(value);
  }

  const std::string& s_;
  std::size_t pos_ = 0;
};

}  // namespace

bool Value::as_bool() const {
  require(type_, Type::Bool);
  return bool_;
}

double Value::as_number() const {
  // Non-finite numbers are written as null, so read null back as NaN.
  if (type_ == Type::Null) return std::nan("");
  require(type_, Type::Number);
  return number_;
}

const std::string& Value::as_string() const {
  require(type_, Type::String);
  return string_;
}

const Array& Value::as_array() const {
  require(type_, Type::Array);
  return array_;
}

const Object& Value::as_object() const {
  require(type_, Type::Object);
  return object_;
}

Value& Value::set(const std::string& key, Value value) {
  require(type_, Type::Object);
  for (auto& member : object_) {
    if (member.first == key) {
      member.second = std::move(value);
      return *this;
    }
  }
  object_.emplace_back(key, std::move(value));
  return *this;
}

bool Value::contains(const std::string& key) const {
  if (type_ != Type::Object) return false;
  return std::any_of(object_.begin(), object_.end(),
                     [&](const auto& member) { return member.first == key; });
}

const Value& Value::at(const std::string& key) const {
  require(type_, Type::Object);
  for (const auto& member : object_) {
    if (member.first == key) return member.second;
  }
  throw std::runtime_error("JSON: missing key '" + key + "'");
}

void Value::push_back(Value value) {
  require(type_, Type::Array);
  array_.push_back(std::move(value));
}

std::string Value::dump(int indent) const {
  std::string out;
  write_value(*this, indent, 0, out);
  return out;
}

Value Value::parse(const std::string& text) { return Parser(text).parse_document(); }

Value from_vector(const std::vector<double>& values) {
  Array items(values.begin(), values.end());
  return Value(std::move(items));
}

std::vector<double> to_vector(const Value& value) {
  std::vector<double> out;
  for (const Value& item : value.as_array()) out.push_back(item.as_number());
  return out;
}

Value from_strings(const std::vector<std::string>& values) {
  Array items(values.begin(), values.end());
  return Value(std::move(items));
}

std::vector<std::string> to_strings(const Value& value) {
  std::vector<std::string> out;
  for (const Value& item : value.as_array()) out.push_back(item.as_string());
  return out;
}

Value load_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  std::ostringstream buffer;
  buffer << in.rdbuf();
  try {
    return Value::parse(buffer.str());
  } catch (const std::exception& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

void save_file(const std::string& path, const Value& value) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write '" + path + "'");
  out << value.dump(2) << '\n';
}

}  // namespace gbx::json
