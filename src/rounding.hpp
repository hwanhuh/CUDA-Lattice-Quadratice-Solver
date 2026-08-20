/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace lattice_qp {

enum class ProjectionPolicy {
    Nearest,
    Floor,
    Ceil,
    AwayFromZero,
};

ProjectionPolicy parseProjectionPolicy(const std::string& projection);
std::int64_t projectCoordinate(double coordinate, ProjectionPolicy projection);

class ProjectionState final {
public:
    ProjectionState(ProjectionPolicy projection, std::size_t integerCount,
        const std::vector<std::int32_t>& incidenceIndptr,
        const std::vector<std::int32_t>& incidenceIndices,
        const std::vector<std::int64_t>& incidenceCoefficients,
        const std::vector<std::int64_t>& baseCones,
        const std::vector<std::int64_t>& minimumCones);

    std::int64_t propose(std::size_t position, double coordinate) const;
    std::int64_t commit(std::size_t position, double coordinate);

    const std::vector<std::int64_t>& values() const noexcept;
    std::size_t correctionCount() const noexcept;
    std::vector<std::int64_t> violations() const;

private:
    ProjectionPolicy projection_;
    std::vector<std::int64_t> values_;
    std::vector<std::int64_t> minimum_;
    std::vector<std::size_t> remaining_;
    std::vector<std::vector<std::pair<std::size_t, std::int64_t>>>
        entriesByPosition_;
    std::size_t correctionCount_ = 0;
};

} // namespace lattice_qp
