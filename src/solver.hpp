/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lattice_qp {

struct SolverInput {
    std::vector<std::int32_t> rowOffsets;
    std::vector<std::int32_t> columnIndices;
    std::vector<double> values;
    std::vector<double> linear;
    std::vector<std::int64_t> integerIndices;
    std::vector<double> periods;
    std::vector<double> initial;
    std::vector<std::int64_t> blockPairs;
    std::vector<std::int32_t> incidenceIndptr;
    std::vector<std::int32_t> incidenceIndices;
    std::vector<std::int64_t> incidenceCoefficients;
    std::vector<std::int64_t> baseCones;
    std::vector<std::int64_t> minimumCones;
};

struct SolverOptions {
    std::string roundingSelection;
    std::string roundingProjection;
    std::string finalSolve;
    double tolerance = 0.0;
    std::int32_t maximumIterations = 0;
    double intermediateTolerance = 0.0;
    std::int32_t intermediateMaximumIterations = 0;
    double multipleRoundingThreshold = 0.0;
    std::int32_t pcgCheckInterval = 0;
};

struct SolverOutput {
    std::vector<double> x;
    std::vector<double> relaxationX;
    double relativeResidual = 0.0;
    bool converged = false;
    std::size_t roundingBatches = 0;
    std::size_t linearSolves = 0;
    std::size_t linearIterations = 0;
    std::size_t pcgHostSynchronizations = 0;
    std::size_t continuationSolves = 0;
    std::size_t correctionCount = 0;
    std::vector<std::int64_t> coneValues;
    std::vector<std::int64_t> coneViolations;
    std::size_t coneViolationCount = 0;
    std::int64_t coneMaxViolation = 0;
};

SolverOutput solveLatticeQp(const SolverInput& input,
    const SolverOptions& options);

} // namespace lattice_qp
