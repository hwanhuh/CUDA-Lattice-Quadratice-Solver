/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_kernels.cuh"

#include <cfloat>
#include <cmath>

namespace lattice_qp::cuda_detail {
namespace {

__global__ void zeroFixedKernel(std::int32_t dimension,
    const std::uint8_t* fixed, double* values)
{
    const std::int32_t index = static_cast<std::int32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index < dimension && fixed[index] != 0)
        values[index] = 0.0;
}

__global__ void applyProjectedFixedUpdatesKernel(std::int32_t count,
    const std::int32_t* indices, std::uint8_t* fixed, double* solution)
{
    const std::int32_t item = static_cast<std::int32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (item >= count)
        return;
    const std::int32_t index = indices[item];
    fixed[index] = 1;
    solution[index] = 0.0;
}

__global__ void applyProjectedRhsUpdatesKernel(std::int32_t count,
    const std::int32_t* indices, const double* values, double* rhs)
{
    const std::int32_t item = static_cast<std::int32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (item < count)
        rhs[indices[item]] = values[item];
}

__global__ void blockJacobiKernel(std::int32_t dimension,
    const double* residual, const double* inverseDiagonal,
    const std::int32_t* blockPartner,
    const double* blockInverseDiagonal,
    const double* blockInverseOffDiagonal,
    const std::uint8_t* fixed, double* preconditioned)
{
    const std::int32_t index = static_cast<std::int32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= dimension)
        return;
    if (fixed != nullptr && fixed[index] != 0) {
        preconditioned[index] = 0.0;
        return;
    }
    const std::int32_t partner = blockPartner[index];
    if (partner >= 0 && (fixed == nullptr || fixed[partner] == 0)) {
        preconditioned[index] = fma(blockInverseOffDiagonal[index],
            residual[partner], blockInverseDiagonal[index] * residual[index]);
    } else {
        preconditioned[index] = inverseDiagonal[index] * residual[index];
    }
}

__global__ void initializePcgStateKernel(double toleranceSquared,
    double* scalars, std::int32_t* state)
{
    const double rhsNormSquared = scalars[kRhsNormSquared];
    const double residualNormSquared = scalars[kResidualNormSquared];
    state[kPcgBreakdown] = 0;
    state[kPcgIterations] = 0;
    scalars[kBeta] = 0.0;
    scalars[kDirectionAddScale] = 0.0;
    if (!isfinite(rhsNormSquared) || !isfinite(residualNormSquared)) {
        state[kPcgActive] = 0;
        state[kPcgBreakdown] = 1;
        return;
    }
    const double threshold = toleranceSquared * fmax(rhsNormSquared, DBL_MIN);
    scalars[kConvergenceThreshold] = threshold;
    state[kPcgActive] = residualNormSquared > threshold ? 1 : 0;
}

__global__ void updateSolutionResidualKernel(std::int32_t dimension,
    double* solution, double* residual, const double* direction,
    const double* matrixDirection, double* scalars,
    const std::uint8_t* fixed, std::int32_t* state)
{
    if (state[kPcgActive] == 0)
        return;
    const double rho = scalars[kRho];
    const double product = scalars[kDirectionProduct];
    const double alpha = rho / product;
    if (!isfinite(rho) || !isfinite(product) || fabs(product) <= DBL_MIN
        || !isfinite(alpha)) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            state[kPcgActive] = 0;
            state[kPcgBreakdown] = 1;
        }
        return;
    }
    const std::int32_t index = static_cast<std::int32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index == 0)
        ++state[kPcgIterations];
    if (index >= dimension)
        return;
    if (fixed != nullptr && fixed[index] != 0) {
        residual[index] = 0.0;
        return;
    }
    solution[index] = fma(alpha, direction[index], solution[index]);
    residual[index] = fma(-alpha, matrixDirection[index], residual[index]);
}

__global__ void prepareBetaKernel(double* scalars, std::int32_t* state)
{
    if (state[kPcgActive] == 0) {
        scalars[kBeta] = 0.0;
        scalars[kDirectionAddScale] = 0.0;
        return;
    }
    const double residualNormSquared = scalars[kResidualNormSquared];
    if (!isfinite(residualNormSquared)) {
        scalars[kBeta] = 0.0;
        scalars[kDirectionAddScale] = 0.0;
        state[kPcgActive] = 0;
        state[kPcgBreakdown] = 1;
        return;
    }
    if (residualNormSquared <= scalars[kConvergenceThreshold]) {
        scalars[kBeta] = 0.0;
        scalars[kDirectionAddScale] = 0.0;
        state[kPcgActive] = 0;
        return;
    }
    const double rho = scalars[kRho];
    const double nextRho = scalars[kNextRho];
    const double beta = nextRho / rho;
    if (!isfinite(rho) || fabs(rho) <= DBL_MIN || !isfinite(nextRho)
        || !isfinite(beta)) {
        scalars[kBeta] = 0.0;
        scalars[kDirectionAddScale] = 0.0;
        state[kPcgActive] = 0;
        state[kPcgBreakdown] = 1;
        return;
    }
    scalars[kBeta] = beta;
    scalars[kRho] = nextRho;
    scalars[kDirectionAddScale] = 1.0;
}

__global__ void scalarSpmvKernel(double matrixValue, const double* input,
    double* output, double alpha, double beta)
{
    output[0] = fma(alpha * matrixValue, input[0], beta * output[0]);
}

} // namespace

void launchZeroFixed(std::int32_t dimension,
    const std::uint8_t* fixed, double* values, cudaStream_t stream)
{
    constexpr std::int32_t threads = 256;
    const std::int32_t blocks = 1 + (dimension - 1) / threads;
    zeroFixedKernel<<<blocks, threads, 0, stream>>>(dimension, fixed, values);
}

void launchApplyProjectedFixedUpdates(std::int32_t count,
    const std::int32_t* indices, std::uint8_t* fixed, double* solution,
    cudaStream_t stream)
{
    constexpr std::int32_t threads = 256;
    const std::int32_t blocks = 1 + (count - 1) / threads;
    applyProjectedFixedUpdatesKernel<<<blocks, threads, 0, stream>>>(
        count, indices, fixed, solution);
}

void launchApplyProjectedRhsUpdates(std::int32_t count,
    const std::int32_t* indices, const double* values, double* rhs,
    cudaStream_t stream)
{
    constexpr std::int32_t threads = 256;
    const std::int32_t blocks = 1 + (count - 1) / threads;
    applyProjectedRhsUpdatesKernel<<<blocks, threads, 0, stream>>>(
        count, indices, values, rhs);
}

void launchBlockJacobi(std::int32_t dimension,
    const double* residual, const double* inverseDiagonal,
    const std::int32_t* blockPartner,
    const double* blockInverseDiagonal,
    const double* blockInverseOffDiagonal,
    const std::uint8_t* fixed, double* preconditioned,
    cudaStream_t stream)
{
    constexpr std::int32_t threads = 256;
    const std::int32_t blocks = 1 + (dimension - 1) / threads;
    blockJacobiKernel<<<blocks, threads, 0, stream>>>(dimension,
        residual, inverseDiagonal, blockPartner, blockInverseDiagonal,
        blockInverseOffDiagonal, fixed, preconditioned);
}

void launchInitializePcgState(double toleranceSquared,
    double* scalars, std::int32_t* state, cudaStream_t stream)
{
    initializePcgStateKernel<<<1, 1, 0, stream>>>(
        toleranceSquared, scalars, state);
}

void launchUpdateSolutionResidual(std::int32_t dimension,
    double* solution, double* residual, const double* direction,
    const double* matrixDirection, double* scalars,
    const std::uint8_t* fixed, std::int32_t* state, cudaStream_t stream)
{
    constexpr std::int32_t threads = 256;
    const std::int32_t blocks = 1 + (dimension - 1) / threads;
    updateSolutionResidualKernel<<<blocks, threads, 0, stream>>>(dimension,
        solution, residual, direction, matrixDirection, scalars, fixed, state);
}

void launchPrepareBeta(double* scalars, std::int32_t* state,
    cudaStream_t stream)
{
    prepareBetaKernel<<<1, 1, 0, stream>>>(scalars, state);
}

void launchScalarSpmv(double matrixValue, const double* input, double* output,
    double alpha, double beta, cudaStream_t stream)
{
    scalarSpmvKernel<<<1, 1, 0, stream>>>(
        matrixValue, input, output, alpha, beta);
}

} // namespace lattice_qp::cuda_detail
