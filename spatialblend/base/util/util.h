// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_BASE_UTIL_UTIL_H_
#define SPATIALBLEND_BASE_UTIL_UTIL_H_

#include <future>
#include <string>

#include "spatialblend/base/common/common.h"
#if defined(__GNUC__)
#include <cxxabi.h>  // For demangling type names (GCC and Clang)
#endif

namespace spatialblend {
namespace util {

// Helper function to create a future that is already ready.
// https://en.cppreference.com/w/cpp/experimental/make_ready_future
template <typename T>
std::future<T> MakeReadyFuture(T value) {
  std::promise<T> promise;
  promise.set_value(value);
  return promise.get_future();
}

template <typename T>
bool IsFutureReady(std::future<T>& future) {
  return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

template <typename T>
std::vector<T> ElementwiseAdd(const std::vector<T>& lhs,
                              const std::vector<T>& rhs) {
  std::vector<T> result = lhs;
  for (size_t i = 0; i < rhs.size(); ++i) {
    result[i] += rhs[i];
  }
  return result;
}

// Helper function to demangle type names (for GCC and Clang)
inline std::string Demangle(const char* name) {
#if defined(__GNUC__)
  int status = -4;  // some arbitrary value to eliminate the compiler warning
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : name;
#else
  return name;
#endif
}

// Function to get the type name as a string
template <typename T>
std::string GetTypeName() {
  return Demangle(typeid(T).name());
}

// Overloaded function to handle multiple types using fold expressions (C++17)
template <typename... Ts>
std::string TypesToString() {
  std::ostringstream oss;
  // Using a fold expression to concatenate type names separated by commas
  ((oss << GetTypeName<Ts>() << ", "), ...);
  std::string result = oss.str();
  if (!result.empty()) {
    result.pop_back();  // Remove the last space
    result.pop_back();  // Remove the last comma
  }
  return result;
}

}  // namespace util
}  // namespace spatialblend

#define RETURN_IF_ERROR(status) \
  if (!status.ok()) {           \
    return status;              \
  }

#define RETURN_VALUE_IF_ERROR(status, return_value) \
  if (!status.ok()) {                               \
    return return_value;                            \
  }

#define RETURN_FUTURE_IF_ERROR(status) \
  RETURN_VALUE_IF_ERROR(status, util::MakeReadyFuture(status))

#define RUNTIME_ERROR_IF_ERROR(status)                       \
  if (!status.ok()) {                                        \
    throw std::runtime_error(std::string(status.message())); \
  }

#define RETURN_IF_ERROR_OR_DEFAULT(status, default_value) \
  if (!status.ok()) {                                     \
    return default_value;                                 \
  } else {                                                \
    return absl::OkStatus();                              \
  }

#endif /* SPATIALBLEND_BASE_UTIL_UTIL_H_ */
