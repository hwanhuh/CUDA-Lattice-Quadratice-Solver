/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace lattice_qp {

struct PcgResult {
    std::vector<double> x;
    std::int32_t iterations = 0;
    double relativeResidual = 0.0;
    bool converged = false;
    std::int32_t hostSynchronizations = 0;
};

class CudaSystem final {
public:
    CudaSystem(const std::vector<std::int32_t>& rowOffsets,
        const std::vector<std::int32_t>& columnIndices,
        const std::vector<double>& values,
        const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs);
    ~CudaSystem();

    CudaSystem(const CudaSystem&) = delete;
    CudaSystem& operator=(const CudaSystem&) = delete;
    CudaSystem(CudaSystem&&) = delete;
    CudaSystem& operator=(CudaSystem&&) = delete;

    void subtractFixedColumn(std::int32_t column, double value,
        const std::vector<std::uint8_t>& fixed, std::vector<double>* rhs,
        std::vector<std::uint8_t>* rhsTouched,
        std::vector<std::int32_t>* rhsUpdateIndices) const;

    PcgResult solve(const std::vector<double>& rhs,
        const std::vector<double>& initial,
        const std::vector<std::uint8_t>& fixed,
        double tolerance, std::int32_t maximumIterations,
        std::int32_t checkInterval);

    void initializeProjectedState(const std::vector<double>& rhs,
        const std::vector<double>& initial,
        const std::vector<std::uint8_t>& fixed);

    PcgResult solveProjected(
        const std::vector<std::int32_t>& newlyFixedIndices,
        const std::vector<std::int32_t>& rhsUpdateIndices,
        const std::vector<double>& rhsUpdateValues,
        double tolerance, std::int32_t maximumIterations,
        std::int32_t checkInterval);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

bool cudaAvailable();

} // namespace lattice_qp
