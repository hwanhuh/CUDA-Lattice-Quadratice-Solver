/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lattice_qp {

class CudaSystem;

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

struct PreparedCudaSolverStats {
    std::size_t solveCalls = 0;
    std::size_t cudaSystemCreations = 0;
    std::size_t cudaSystemReuses = 0;
};

// Owns the immutable CSR analysis, block-Jacobi preconditioner, CUDA matrix,
// and PCG work buffers used by a family of solves with the same structure.
// Per-solve projected state is fully initialized by solve(), so fixed masks,
// RHS deltas, and iterates never leak between public calls.
class PreparedCudaSolver final {
public:
    explicit PreparedCudaSolver(SolverInput structuralInput);
    ~PreparedCudaSolver();

    PreparedCudaSolver(const PreparedCudaSolver&) = delete;
    PreparedCudaSolver& operator=(const PreparedCudaSolver&) = delete;
    PreparedCudaSolver(PreparedCudaSolver&&) = delete;
    PreparedCudaSolver& operator=(PreparedCudaSolver&&) = delete;

    SolverOutput solve(std::vector<double> linear,
        std::vector<double> initial,
        std::vector<std::int32_t> incidenceIndptr,
        std::vector<std::int32_t> incidenceIndices,
        std::vector<std::int64_t> incidenceCoefficients,
        std::vector<std::int64_t> baseCones,
        std::vector<std::int64_t> minimumCones,
        const SolverOptions& options);

    PreparedCudaSolverStats stats() const;
    void resetStats();

private:
    SolverInput input_;
    std::unique_ptr<CudaSystem> system_;
    mutable std::mutex solveMutex_;
    std::size_t lifetimeSolveCalls_ = 0;
    PreparedCudaSolverStats stats_ { 0, 1, 0 };
};

} // namespace lattice_qp
