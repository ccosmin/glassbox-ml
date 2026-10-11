#pragma once

#include <string>

#include "glassbox/json.hpp"
#include "glassbox/run.hpp"

namespace gbx {

/// Version, source commit, compiler and build type of this binary.
json::Value build_info();

/// Current time as ISO 8601 UTC, e.g. "2026-10-11T09:30:00Z".
std::string utc_timestamp();

/// Writes the audit trail of a run into `directory` (created if needed):
///
///   model.json         the deployable pipeline (deterministic content)
///   cost_history.csv   the learning curve
///   manifest.json      spec, data fingerprint, build info, metrics,
///                      explanations, robustness results, artifact hashes
///   model_card.md      human-readable documentation of all of the above
///
/// Returns the manifest.
json::Value write_run_artifacts(const std::string& directory, const RunSpec& spec,
                                const RunResult& result);

std::string render_model_card(const RunSpec& spec, const RunResult& result,
                              const json::Value& manifest);

}  // namespace gbx
