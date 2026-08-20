/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace lattice_qp::cuda_detail {

enum DeviceScalar : std::size_t {
    kRhsNormSquared = 0,
    kResidualNormSquared,
    kRho,
    kDirectionProduct,
    kNextRho,
    kBeta,
    kConvergenceThreshold,
    kDirectionAddScale,
    kDeviceScalarCount,
};

enum DeviceState : std::size_t {
    kPcgActive = 0,
    kPcgBreakdown,
    kPcgIterations,
    kDeviceStateCount,
};

void launchZeroFixed(std::int32_t dimension,
    const std::uint8_t* fixed, double* values, cudaStream_t stream);

void launchApplyProjectedFixedUpdates(std::int32_t count,
    const std::int32_t* indices, std::uint8_t* fixed, double* solution,
    cudaStream_t stream);

void launchApplyProjectedRhsUpdates(std::int32_t count,
    const std::int32_t* indices, const double* values, double* rhs,
    cudaStream_t stream);

void launchBlockJacobi(std::int32_t dimension,
    const double* residual, const double* inverseDiagonal,
    const std::int32_t* blockPartner,
    const double* blockInverseDiagonal,
    const double* blockInverseOffDiagonal,
    const std::uint8_t* fixed, double* preconditioned,
    cudaStream_t stream);

void launchInitializePcgState(double toleranceSquared,
    double* scalars, std::int32_t* state, cudaStream_t stream);

void launchUpdateSolutionResidual(std::int32_t dimension,
    double* solution, double* residual, const double* direction,
    const double* matrixDirection, double* scalars,
    const std::uint8_t* fixed, std::int32_t* state, cudaStream_t stream);

void launchPrepareBeta(double* scalars, std::int32_t* state,
    cudaStream_t stream);

void launchScalarSpmv(double matrixValue, const double* input, double* output,
    double alpha, double beta, cudaStream_t stream);

} // namespace lattice_qp::cuda_detail
