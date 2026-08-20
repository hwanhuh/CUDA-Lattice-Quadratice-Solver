/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rounding.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lattice_qp {

ProjectionPolicy parseProjectionPolicy(const std::string& projection)
{
    if (projection == "nearest")
        return ProjectionPolicy::Nearest;
    if (projection == "floor")
        return ProjectionPolicy::Floor;
    if (projection == "ceil")
        return ProjectionPolicy::Ceil;
    if (projection == "away_from_zero")
        return ProjectionPolicy::AwayFromZero;
    throw std::invalid_argument("unsupported rounding projection policy");
}

std::int64_t projectCoordinate(double coordinate, ProjectionPolicy projection)
{
    if (!std::isfinite(coordinate))
        throw std::runtime_error(
            "rounding encountered a non-finite lattice coordinate");
    double projected = 0.0;
    switch (projection) {
    case ProjectionPolicy::Nearest:
        projected = std::floor(coordinate + 0.5);
        break;
    case ProjectionPolicy::Floor:
        projected = std::floor(coordinate);
        break;
    case ProjectionPolicy::Ceil:
        projected = std::ceil(coordinate);
        break;
    case ProjectionPolicy::AwayFromZero:
        projected = coordinate > 0.0
            ? std::ceil(coordinate) : std::floor(coordinate);
        break;
    }
    const long double exact = static_cast<long double>(projected);
    if (exact < static_cast<long double>(
            std::numeric_limits<std::int64_t>::min())
        || exact > static_cast<long double>(
            std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error(
            "projected lattice coordinate exceeds int64 range");
    }
    return static_cast<std::int64_t>(projected);
}

ProjectionState::ProjectionState(ProjectionPolicy projection,
    std::size_t integerCount,
    const std::vector<std::int32_t>& incidenceIndptr,
    const std::vector<std::int32_t>& incidenceIndices,
    const std::vector<std::int64_t>& incidenceCoefficients,
    const std::vector<std::int64_t>& baseCones,
    const std::vector<std::int64_t>& minimumCones)
    : projection_(projection), values_(baseCones), minimum_(minimumCones),
      remaining_(baseCones.size(), 0), entriesByPosition_(integerCount)
{
    if (baseCones.size() != minimumCones.size()
        || incidenceIndptr.size() != baseCones.size() + 1
        || incidenceIndptr.empty() || incidenceIndptr.front() != 0
        || incidenceIndices.size() != incidenceCoefficients.size()
        || incidenceIndptr.back()
            != static_cast<std::int32_t>(incidenceIndices.size())) {
        throw std::invalid_argument("invalid cone incidence CSR arrays");
    }
    for (std::size_t cone = 0; cone < baseCones.size(); ++cone) {
        const std::int32_t begin = incidenceIndptr[cone];
        const std::int32_t end = incidenceIndptr[cone + 1];
        if (begin < 0 || end < begin)
            throw std::invalid_argument(
                "cone incidence offsets are not monotone");
        remaining_[cone] = static_cast<std::size_t>(end - begin);
        std::vector<std::int32_t> rowPositions;
        rowPositions.reserve(static_cast<std::size_t>(end - begin));
        for (std::int32_t entry = begin; entry < end; ++entry) {
            const std::int32_t position = incidenceIndices[entry];
            const std::int64_t coefficient = incidenceCoefficients[entry];
            if (position < 0
                || position >= static_cast<std::int32_t>(integerCount)) {
                throw std::invalid_argument(
                    "cone incidence integer position is out of range");
            }
            if (coefficient == 0) {
                throw std::invalid_argument(
                    "cone incidence coefficient must be nonzero");
            }
            if (std::find(rowPositions.begin(), rowPositions.end(), position)
                != rowPositions.end()) {
                throw std::invalid_argument(
                    "cone incidence rows must contain unique positions");
            }
            rowPositions.push_back(position);
            entriesByPosition_[position].emplace_back(cone, coefficient);
        }
    }
}

std::int64_t ProjectionState::propose(
    std::size_t position, double coordinate) const
{
    std::int64_t projected = projectCoordinate(coordinate, projection_);
    bool hasLower = false;
    bool hasUpper = false;
    long double lower = 0.0L;
    long double upper = 0.0L;
    for (const auto& entry : entriesByPosition_[position]) {
        const std::size_t cone = entry.first;
        const std::int64_t coefficient = entry.second;
        if (remaining_[cone] != 1)
            continue;
        const long double deficit = static_cast<long double>(minimum_[cone])
            - static_cast<long double>(values_[cone]);
        if (coefficient > 0) {
            const long double bound = std::ceil(
                deficit / static_cast<long double>(coefficient));
            lower = hasLower ? std::max(lower, bound) : bound;
            hasLower = true;
        } else {
            const long double bound = std::floor(
                deficit / static_cast<long double>(coefficient));
            upper = hasUpper ? std::min(upper, bound) : bound;
            hasUpper = true;
        }
    }
    if (hasLower && hasUpper && lower > upper)
        return projected;
    long double corrected = static_cast<long double>(projected);
    if (hasLower)
        corrected = std::max(corrected, lower);
    if (hasUpper)
        corrected = std::min(corrected, upper);
    if (corrected < static_cast<long double>(
            std::numeric_limits<std::int64_t>::min())
        || corrected > static_cast<long double>(
            std::numeric_limits<std::int64_t>::max())) {
        // No representable correction exists. Retain the deterministic basic
        // projection and expose the unsatisfied row in the audit.
        return projected;
    }
    return static_cast<std::int64_t>(corrected);
}

std::int64_t ProjectionState::commit(
    std::size_t position, double coordinate)
{
    const std::int64_t basic = projectCoordinate(coordinate, projection_);
    const std::int64_t projected = propose(position, coordinate);
    if (basic != projected)
        ++correctionCount_;
    for (const auto& entry : entriesByPosition_[position]) {
        const std::size_t cone = entry.first;
        const std::int64_t coefficient = entry.second;
        const long double next = static_cast<long double>(values_[cone])
            + static_cast<long double>(coefficient)
                * static_cast<long double>(projected);
        if (next < static_cast<long double>(
                std::numeric_limits<std::int64_t>::min())
            || next > static_cast<long double>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::runtime_error("cone accumulation exceeds int64 range");
        }
        values_[cone] = static_cast<std::int64_t>(next);
        --remaining_[cone];
    }
    return projected;
}

const std::vector<std::int64_t>& ProjectionState::values() const noexcept
{
    return values_;
}

std::size_t ProjectionState::correctionCount() const noexcept
{
    return correctionCount_;
}

std::vector<std::int64_t> ProjectionState::violations() const
{
    std::vector<std::int64_t> result(values_.size(), 0);
    for (std::size_t cone = 0; cone < values_.size(); ++cone) {
        if (values_[cone] < minimum_[cone]) {
            const long double difference
                = static_cast<long double>(minimum_[cone])
                - static_cast<long double>(values_[cone]);
            result[cone] = difference > static_cast<long double>(
                std::numeric_limits<std::int64_t>::max())
                ? std::numeric_limits<std::int64_t>::max()
                : static_cast<std::int64_t>(difference);
        }
    }
    return result;
}

} // namespace lattice_qp
