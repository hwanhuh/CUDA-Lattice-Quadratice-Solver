/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace lattice_qp {
namespace {

[[noreturn]] void throwCuda(cudaError_t status, const char* operation)
{
    throw std::runtime_error(std::string(operation) + ": "
        + cudaGetErrorString(status));
}

void checkCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throwCuda(status, operation);
}

void checkCublas(cublasStatus_t status, const char* operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation)
            + ": cuBLAS status " + std::to_string(int(status)));
    }
}

void checkCusparse(cusparseStatus_t status, const char* operation)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation)
            + ": cuSPARSE status " + std::to_string(int(status)));
    }
}

template <typename T>
void allocateCuda(T** pointer, std::size_t bytes, const char* operation)
{
    checkCuda(cudaMalloc(reinterpret_cast<void**>(pointer), bytes), operation);
}

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
    if (index == 0) {
        ++state[kPcgIterations];
    }
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

struct SolveResult {
    std::vector<double> x;
    std::int32_t iterations = 0;
    double relativeResidual = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::int32_t hostSynchronizations = 0;
};

class CudaSystem {
public:
    CudaSystem(const std::vector<std::int32_t>& rowOffsets,
        const std::vector<std::int32_t>& columnIndices,
        const std::vector<double>& values,
        const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs)
        : dimension_(static_cast<std::int32_t>(rowOffsets.size() - 1)),
          nonzeros_(static_cast<std::int32_t>(values.size())),
          hostRowOffsets_(rowOffsets), hostColumnIndices_(columnIndices),
          hostValues_(values)
    {
        if (dimension_ <= 0 || nonzeros_ <= 0
            || columnIndices.size() != values.size()
            || rowOffsets.front() != 0 || rowOffsets.back() != nonzeros_) {
            throw std::invalid_argument("invalid CSR matrix");
        }
        try {
            prepareHostPreconditioner(explicitPairs);
            prepareHostColumns();
            allocateDevice();
        } catch (...) {
            destroy();
            throw;
        }
    }

    CudaSystem(const CudaSystem&) = delete;
    CudaSystem& operator=(const CudaSystem&) = delete;

    ~CudaSystem() { destroy(); }

    void subtractFixedColumn(std::int32_t column, double value,
        const std::vector<std::uint8_t>& fixed, std::vector<double>* rhs,
        std::vector<std::uint8_t>* rhsTouched,
        std::vector<std::int32_t>* rhsUpdateIndices) const
    {
        const std::int32_t begin = columnOffsets_[column];
        const std::int32_t end = columnOffsets_[column + 1];
        for (std::int32_t entry = begin; entry < end; ++entry) {
            const std::int32_t row = columnRows_[entry];
            if (fixed[row] == 0) {
                (*rhs)[row] -= columnValues_[entry] * value;
                if ((*rhsTouched)[row] == 0) {
                    (*rhsTouched)[row] = 1;
                    rhsUpdateIndices->push_back(row);
                }
            }
        }
    }

    SolveResult solve(const std::vector<double>& rhs,
        const std::vector<double>& initial,
        const std::vector<std::uint8_t>& fixed,
        double tolerance, std::int32_t maximumIterations,
        std::int32_t checkInterval)
    {
        if (rhs.size() != static_cast<std::size_t>(dimension_)
            || initial.size() != rhs.size() || fixed.size() != rhs.size()
            || !(tolerance > 0.0) || !std::isfinite(tolerance)
            || maximumIterations <= 0 || checkInterval <= 0) {
            throw std::invalid_argument("invalid CUDA PCG input");
        }
        const bool projected = std::any_of(fixed.begin(), fixed.end(),
            [](std::uint8_t value) { return value != 0; });
        const std::size_t vectorBytes = rhs.size() * sizeof(double);
        const std::size_t maskBytes = fixed.size() * sizeof(std::uint8_t);
        checkCuda(cudaMemcpyAsync(deviceRhs_, rhs.data(), vectorBytes,
            cudaMemcpyHostToDevice, stream_), "copy rhs");
        checkCuda(cudaMemcpyAsync(deviceX_, initial.data(), vectorBytes,
            cudaMemcpyHostToDevice, stream_), "copy initial guess");
        if (projected) {
            checkCuda(cudaMemcpyAsync(deviceFixed_, fixed.data(), maskBytes,
                cudaMemcpyHostToDevice, stream_), "copy fixed mask");
        }
        projectedStateReady_ = false;
        return solveUploaded(projected, tolerance, maximumIterations,
            checkInterval);
    }

    void initializeProjectedState(const std::vector<double>& rhs,
        const std::vector<double>& initial,
        const std::vector<std::uint8_t>& fixed)
    {
        if (rhs.size() != static_cast<std::size_t>(dimension_)
            || initial.size() != rhs.size() || fixed.size() != rhs.size()) {
            throw std::invalid_argument(
                "invalid persistent projected CUDA state input");
        }
        const std::size_t vectorBytes = rhs.size() * sizeof(double);
        const std::size_t maskBytes = fixed.size() * sizeof(std::uint8_t);
        projectedStateReady_ = false;
        checkCuda(cudaMemcpyAsync(deviceFixed_, fixed.data(), maskBytes,
            cudaMemcpyHostToDevice, stream_), "initialize projected fixed mask");
        checkCuda(cudaMemcpyAsync(deviceRhs_, rhs.data(), vectorBytes,
            cudaMemcpyHostToDevice, stream_), "initialize projected rhs");
        checkCuda(cudaMemcpyAsync(deviceX_, initial.data(), vectorBytes,
            cudaMemcpyHostToDevice, stream_), "initialize projected solution");
        checkCuda(cudaStreamSynchronize(stream_),
            "synchronize projected state initialization");
        projectedStateReady_ = true;
    }

    SolveResult solveProjected(
        const std::vector<std::int32_t>& newlyFixedIndices,
        const std::vector<std::int32_t>& rhsUpdateIndices,
        const std::vector<double>& rhsUpdateValues,
        double tolerance, std::int32_t maximumIterations,
        std::int32_t checkInterval)
    {
        if (!projectedStateReady_)
            throw std::invalid_argument("persistent projected CUDA state is not initialized");
        if (newlyFixedIndices.size() > static_cast<std::size_t>(dimension_)
            || rhsUpdateIndices.size() > static_cast<std::size_t>(dimension_)
            || rhsUpdateValues.size() != rhsUpdateIndices.size()
            || !(tolerance > 0.0) || !std::isfinite(tolerance)
            || maximumIterations <= 0 || checkInterval <= 0) {
            throw std::invalid_argument("invalid persistent projected CUDA solve input");
        }
        for (std::int32_t index : newlyFixedIndices) {
            if (index < 0 || index >= dimension_)
                throw std::invalid_argument("projected fixed index is out of range");
        }
        for (std::int32_t index : rhsUpdateIndices) {
            if (index < 0 || index >= dimension_)
                throw std::invalid_argument("projected rhs index is out of range");
        }

        try {
            constexpr std::int32_t threads = 256;
            if (!newlyFixedIndices.empty()) {
                const std::size_t bytes = newlyFixedIndices.size()
                    * sizeof(std::int32_t);
                checkCuda(cudaMemcpyAsync(deviceUpdateIndices_,
                    newlyFixedIndices.data(), bytes, cudaMemcpyHostToDevice,
                    stream_), "upload projected fixed updates");
                const std::int32_t count = static_cast<std::int32_t>(
                    newlyFixedIndices.size());
                const std::int32_t blocks = 1 + (count - 1) / threads;
                applyProjectedFixedUpdatesKernel<<<blocks, threads, 0, stream_>>>(
                    count, deviceUpdateIndices_, deviceFixed_, deviceX_);
                checkCuda(cudaGetLastError(), "apply projected fixed updates");
            }
            if (!rhsUpdateIndices.empty()) {
                const std::size_t indexBytes = rhsUpdateIndices.size()
                    * sizeof(std::int32_t);
                const std::size_t valueBytes = rhsUpdateValues.size()
                    * sizeof(double);
                checkCuda(cudaMemcpyAsync(deviceUpdateIndices_,
                    rhsUpdateIndices.data(), indexBytes, cudaMemcpyHostToDevice,
                    stream_), "upload projected rhs indices");
                checkCuda(cudaMemcpyAsync(deviceUpdateValues_,
                    rhsUpdateValues.data(), valueBytes, cudaMemcpyHostToDevice,
                    stream_), "upload projected rhs values");
                const std::int32_t count = static_cast<std::int32_t>(
                    rhsUpdateIndices.size());
                const std::int32_t blocks = 1 + (count - 1) / threads;
                applyProjectedRhsUpdatesKernel<<<blocks, threads, 0, stream_>>>(
                    count, deviceUpdateIndices_, deviceUpdateValues_, deviceRhs_);
                checkCuda(cudaGetLastError(), "apply projected rhs updates");
            }
            return solveUploaded(true, tolerance, maximumIterations,
                checkInterval);
        } catch (...) {
            // A failed asynchronous update or PCG pass may have partially
            // advanced x/rhs/mask.  Force a full host refresh before this
            // state can ever be reused instead of accepting another delta.
            projectedStateReady_ = false;
            throw;
        }
    }

private:
    SolveResult solveUploaded(bool projected, double tolerance,
        std::int32_t maximumIterations, std::int32_t checkInterval)
    {
        const std::size_t vectorBytes = static_cast<std::size_t>(dimension_)
            * sizeof(double);
        checkCuda(cudaMemcpyAsync(deviceResidual_, deviceRhs_, vectorBytes,
            cudaMemcpyDeviceToDevice, stream_), "initialize residual");
        spmv(vectorX_, vectorResidual_, -1.0, 1.0);
        if (projected)
            zeroFixed(deviceResidual_);

        checkCublas(cublasDdot(cublas_, dimension_, deviceRhs_, 1,
            deviceRhs_, 1, deviceScalars_ + kRhsNormSquared), "rhs dot");
        checkCublas(cublasDdot(cublas_, dimension_, deviceResidual_, 1,
            deviceResidual_, 1, deviceScalars_ + kResidualNormSquared),
            "residual dot");
        initializePcgStateKernel<<<1, 1, 0, stream_>>>(
            tolerance * tolerance, deviceScalars_, deviceState_);
        checkCuda(cudaGetLastError(), "initialize PCG state kernel");
        checkCuda(cudaMemcpyAsync(hostNorms_,
            deviceScalars_ + kRhsNormSquared, 2 * sizeof(double),
            cudaMemcpyDeviceToHost, stream_), "copy initial norms");
        checkCuda(cudaStreamSynchronize(stream_), "synchronize initial residual");
        const double rhsNormSquared = hostNorms_[0];
        double residualNormSquared = hostNorms_[1];
        if (!std::isfinite(rhsNormSquared)
            || !std::isfinite(residualNormSquared)) {
            throw std::runtime_error("non-finite CUDA PCG initial residual");
        }

        const double denominator = std::max(
            rhsNormSquared, std::numeric_limits<double>::min());
        const double threshold = tolerance * tolerance * denominator;
        hostState_[kPcgActive] = residualNormSquared > threshold ? 1 : 0;
        hostState_[kPcgBreakdown] = 0;
        hostState_[kPcgIterations] = 0;
        std::int32_t launchedIterations = 0;
        std::int32_t hostSynchronizations = 1;
        if (residualNormSquared > threshold) {
            applyPreconditioner(projected);
            checkCuda(cudaMemcpyAsync(deviceDirection_, devicePreconditioned_,
                vectorBytes, cudaMemcpyDeviceToDevice, stream_),
                "initialize direction");
            checkCublas(cublasDdot(cublas_, dimension_, deviceResidual_, 1,
                devicePreconditioned_, 1, deviceScalars_ + kRho), "rho dot");
            std::int32_t chunkSize = 1;
            while (launchedIterations < maximumIterations
                && hostState_[kPcgActive] != 0) {
                const std::int32_t chunkEnd = std::min(maximumIterations,
                    launchedIterations + chunkSize);
                while (launchedIterations < chunkEnd) {
                    spmv(vectorDirection_, vectorMatrixDirection_, 1.0, 0.0);
                    checkCublas(cublasDdot(cublas_, dimension_, deviceDirection_, 1,
                        deviceMatrixDirection_, 1,
                        deviceScalars_ + kDirectionProduct), "direction dot");
                    constexpr std::int32_t threads = 256;
                    const std::int32_t blocks = 1 + (dimension_ - 1) / threads;
                    updateSolutionResidualKernel<<<blocks, threads, 0, stream_>>>(
                        dimension_, deviceX_, deviceResidual_, deviceDirection_,
                        deviceMatrixDirection_, deviceScalars_,
                        projected ? deviceFixed_ : nullptr, deviceState_);
                    checkCuda(cudaGetLastError(), "update solution kernel");
                    checkCublas(cublasDdot(cublas_, dimension_, deviceResidual_, 1,
                        deviceResidual_, 1,
                        deviceScalars_ + kResidualNormSquared), "residual dot");
                    ++launchedIterations;
                    if (launchedIterations < maximumIterations) {
                        applyPreconditioner(projected);
                        checkCublas(cublasDdot(cublas_, dimension_,
                            deviceResidual_, 1, devicePreconditioned_, 1,
                            deviceScalars_ + kNextRho), "next rho dot");
                        prepareBetaKernel<<<1, 1, 0, stream_>>>(
                            deviceScalars_, deviceState_);
                        checkCuda(cudaGetLastError(), "prepare beta kernel");
                        checkCublas(cublasDscal(cublas_, dimension_,
                            deviceScalars_ + kBeta, deviceDirection_, 1),
                            "scale direction");
                        checkCublas(cublasDaxpy(cublas_, dimension_,
                            deviceScalars_ + kDirectionAddScale,
                            devicePreconditioned_, 1, deviceDirection_, 1),
                            "update direction");
                    }
                }
                checkCuda(cudaMemcpyAsync(hostState_, deviceState_,
                    kDeviceStateCount * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost, stream_), "copy PCG state");
                checkCuda(cudaStreamSynchronize(stream_),
                    "synchronize PCG chunk");
                ++hostSynchronizations;
                if (hostState_[kPcgBreakdown] != 0)
                    throw std::runtime_error("CUDA PCG breakdown");
                // Two single-iteration probes avoid speculative work in the
                // short projected systems common during multiple rounding.
                // Longer solves then amortize host checks with 2, 4, ...
                // iteration chunks up to the public maximum.
                if (launchedIterations >= 2)
                    chunkSize = std::min(checkInterval, chunkSize * 2);
            }
        }

        // Recompute b-Hx once so the returned certificate is not based on the
        // recursively updated residual.
        checkCuda(cudaMemcpyAsync(deviceResidual_, deviceRhs_, vectorBytes,
            cudaMemcpyDeviceToDevice, stream_), "reset explicit residual");
        spmv(vectorX_, vectorResidual_, -1.0, 1.0);
        if (projected)
            zeroFixed(deviceResidual_);
        checkCublas(cublasDdot(cublas_, dimension_, deviceResidual_, 1,
            deviceResidual_, 1, deviceScalars_ + kResidualNormSquared),
            "explicit residual dot");
        SolveResult result;
        result.x.resize(static_cast<std::size_t>(dimension_));
        checkCuda(cudaMemcpyAsync(hostNorms_ + 1,
            deviceScalars_ + kResidualNormSquared, sizeof(double),
            cudaMemcpyDeviceToHost, stream_), "copy explicit residual");
        checkCuda(cudaMemcpyAsync(result.x.data(), deviceX_, vectorBytes,
            cudaMemcpyDeviceToHost, stream_), "copy solution");
        checkCuda(cudaStreamSynchronize(stream_), "synchronize solution");
        ++hostSynchronizations;
        residualNormSquared = hostNorms_[1];
        result.iterations = hostState_[kPcgIterations];
        result.relativeResidual = std::sqrt(residualNormSquared / denominator);
        result.converged = residualNormSquared <= threshold;
        result.hostSynchronizations = hostSynchronizations;
        return result;
    }

    void prepareHostPreconditioner(
        const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs)
    {
        inverseDiagonal_.assign(dimension_, 1.0);
        std::vector<double> diagonal(dimension_, 0.0);
        for (std::int32_t row = 0; row < dimension_; ++row) {
            if (hostRowOffsets_[row] > hostRowOffsets_[row + 1])
                throw std::invalid_argument("CSR row offsets must be monotone");
            for (std::int32_t entry = hostRowOffsets_[row];
                 entry < hostRowOffsets_[row + 1]; ++entry) {
                const std::int32_t column = hostColumnIndices_[entry];
                if (column < 0 || column >= dimension_)
                    throw std::invalid_argument("CSR column is out of range");
                const double value = hostValues_[entry];
                if (!std::isfinite(value))
                    throw std::invalid_argument("CSR matrix contains non-finite values");
                if (column == row) {
                    diagonal[row] = value;
                }
            }
        }
        for (std::int32_t index = 0; index < dimension_; ++index) {
            if (diagonal[index] < -1.0e-14)
                throw std::invalid_argument("H has a negative diagonal entry");
            if (std::abs(diagonal[index]) > std::numeric_limits<double>::min())
                inverseDiagonal_[index] = 1.0 / diagonal[index];
        }

        blockPartner_.assign(dimension_, -1);
        blockInverseDiagonal_ = inverseDiagonal_;
        blockInverseOffDiagonal_.assign(dimension_, 0.0);
        std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
        if (!explicitPairs.empty()) {
            pairs = explicitPairs;
        } else {
            std::vector<std::int32_t> strongest(dimension_, -1);
            std::vector<double> strongestScore(dimension_, 0.0);
            for (std::int32_t row = 0; row < dimension_; ++row) {
                const double rowDiagonal = diagonal[row];
                if (!(rowDiagonal > 0.0) || !std::isfinite(rowDiagonal))
                    continue;
                for (std::int32_t entry = hostRowOffsets_[row];
                     entry < hostRowOffsets_[row + 1]; ++entry) {
                    const std::int32_t column = hostColumnIndices_[entry];
                    if (column == row)
                        continue;
                    const double columnDiagonal = diagonal[column];
                    const double coupling = hostValues_[entry];
                    if (!(columnDiagonal > 0.0)
                        || !std::isfinite(columnDiagonal)
                        || !std::isfinite(coupling)) {
                        continue;
                    }
                    const double score = coupling * coupling
                        / (rowDiagonal * columnDiagonal);
                    if (score > strongestScore[row]
                        || (score == strongestScore[row]
                            && score > 0.0
                            && (strongest[row] < 0
                                || column < strongest[row]))) {
                        strongestScore[row] = score;
                        strongest[row] = column;
                    }
                }
            }
            for (std::int32_t row = 0; row < dimension_; ++row) {
                const std::int32_t partner = strongest[row];
                if (partner > row && strongest[partner] == row)
                    pairs.emplace_back(row, partner);
            }
        }
        std::vector<std::uint8_t> used(dimension_, 0);
        for (const auto& pair : pairs) {
            const std::int32_t first = pair.first;
            const std::int32_t second = pair.second;
            if (first < 0 || second < 0 || first >= dimension_
                || second >= dimension_ || first == second
                || used[first] || used[second]) {
                throw std::invalid_argument("block pairs must be valid and disjoint");
            }
            used[first] = used[second] = 1;
            double offDiagonal = 0.0;
            for (std::int32_t entry = hostRowOffsets_[first];
                 entry < hostRowOffsets_[first + 1]; ++entry) {
                if (hostColumnIndices_[entry] == second) {
                    offDiagonal = hostValues_[entry];
                    break;
                }
            }
            const double product = diagonal[first] * diagonal[second];
            const double determinant = product - offDiagonal * offDiagonal;
            if (!(diagonal[first] > 0.0) || !(diagonal[second] > 0.0)
                || !std::isfinite(product) || !std::isfinite(determinant)
                || !(determinant > 1.0e-10 * product)) {
                continue;
            }
            blockPartner_[first] = second;
            blockPartner_[second] = first;
            blockInverseDiagonal_[first] = diagonal[second] / determinant;
            blockInverseDiagonal_[second] = diagonal[first] / determinant;
            const double inverseOffDiagonal = -offDiagonal / determinant;
            blockInverseOffDiagonal_[first] = inverseOffDiagonal;
            blockInverseOffDiagonal_[second] = inverseOffDiagonal;
        }
    }

    void prepareHostColumns()
    {
        columnOffsets_.assign(static_cast<std::size_t>(dimension_) + 1, 0);
        for (std::int32_t column : hostColumnIndices_)
            ++columnOffsets_[static_cast<std::size_t>(column) + 1];
        std::partial_sum(columnOffsets_.begin(), columnOffsets_.end(),
            columnOffsets_.begin());
        columnRows_.resize(nonzeros_);
        columnValues_.resize(nonzeros_);
        std::vector<std::int32_t> next = columnOffsets_;
        for (std::int32_t row = 0; row < dimension_; ++row) {
            for (std::int32_t entry = hostRowOffsets_[row];
                 entry < hostRowOffsets_[row + 1]; ++entry) {
                const std::int32_t column = hostColumnIndices_[entry];
                const std::int32_t destination = next[column]++;
                columnRows_[destination] = row;
                columnValues_[destination] = hostValues_[entry];
            }
        }
    }

    void allocateDevice()
    {
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "create CUDA stream");
        checkCublas(cublasCreate(&cublas_), "create cuBLAS handle");
        checkCublas(cublasSetStream(cublas_, stream_), "set cuBLAS stream");
        checkCublas(cublasSetPointerMode(cublas_, CUBLAS_POINTER_MODE_DEVICE),
            "set cuBLAS pointer mode");
        checkCusparse(cusparseCreate(&cusparse_), "create cuSPARSE handle");
        checkCusparse(cusparseSetStream(cusparse_, stream_),
            "set cuSPARSE stream");

        const std::size_t rowBytes = hostRowOffsets_.size() * sizeof(std::int32_t);
        const std::size_t columnBytes = hostColumnIndices_.size() * sizeof(std::int32_t);
        const std::size_t valueBytes = hostValues_.size() * sizeof(double);
        const std::size_t vectorBytes = static_cast<std::size_t>(dimension_) * sizeof(double);
        const std::size_t maskBytes = static_cast<std::size_t>(dimension_) * sizeof(std::uint8_t);
        allocateCuda(&deviceRowOffsets_, rowBytes, "allocate row offsets");
        allocateCuda(&deviceColumnIndices_, columnBytes, "allocate columns");
        allocateCuda(&deviceValues_, valueBytes, "allocate values");
        allocateCuda(&deviceInverseDiagonal_, vectorBytes, "allocate diagonal");
        allocateCuda(&deviceBlockPartner_,
            static_cast<std::size_t>(dimension_) * sizeof(std::int32_t),
            "allocate block partners");
        allocateCuda(&deviceBlockInverseDiagonal_, vectorBytes,
            "allocate block diagonal");
        allocateCuda(&deviceBlockInverseOffDiagonal_, vectorBytes,
            "allocate block off-diagonal");
        allocateCuda(&deviceRhs_, vectorBytes, "allocate rhs");
        allocateCuda(&deviceX_, vectorBytes, "allocate solution");
        allocateCuda(&deviceResidual_, vectorBytes, "allocate residual");
        allocateCuda(&devicePreconditioned_, vectorBytes,
            "allocate preconditioned residual");
        allocateCuda(&deviceDirection_, vectorBytes, "allocate direction");
        allocateCuda(&deviceMatrixDirection_, vectorBytes,
            "allocate matrix direction");
        allocateCuda(&deviceFixed_, maskBytes, "allocate fixed mask");
        allocateCuda(&deviceUpdateIndices_,
            static_cast<std::size_t>(dimension_) * sizeof(std::int32_t),
            "allocate projected update indices");
        allocateCuda(&deviceUpdateValues_, vectorBytes,
            "allocate projected update values");
        allocateCuda(&deviceScalars_, kDeviceScalarCount * sizeof(double),
            "allocate device scalars");
        allocateCuda(&deviceState_, kDeviceStateCount * sizeof(std::int32_t),
            "allocate PCG state");
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hostNorms_),
            2 * sizeof(double)), "allocate pinned host norms");
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hostState_),
            kDeviceStateCount * sizeof(std::int32_t)),
            "allocate pinned host PCG state");

        checkCuda(cudaMemcpyAsync(deviceRowOffsets_, hostRowOffsets_.data(),
            rowBytes, cudaMemcpyHostToDevice, stream_), "upload row offsets");
        checkCuda(cudaMemcpyAsync(deviceColumnIndices_, hostColumnIndices_.data(),
            columnBytes, cudaMemcpyHostToDevice, stream_), "upload columns");
        checkCuda(cudaMemcpyAsync(deviceValues_, hostValues_.data(), valueBytes,
            cudaMemcpyHostToDevice, stream_), "upload values");
        checkCuda(cudaMemcpyAsync(deviceInverseDiagonal_, inverseDiagonal_.data(),
            vectorBytes, cudaMemcpyHostToDevice, stream_), "upload diagonal");
        checkCuda(cudaMemcpyAsync(deviceBlockPartner_, blockPartner_.data(),
            static_cast<std::size_t>(dimension_) * sizeof(std::int32_t),
            cudaMemcpyHostToDevice, stream_), "upload block partners");
        checkCuda(cudaMemcpyAsync(deviceBlockInverseDiagonal_,
            blockInverseDiagonal_.data(), vectorBytes,
            cudaMemcpyHostToDevice, stream_), "upload block diagonal");
        checkCuda(cudaMemcpyAsync(deviceBlockInverseOffDiagonal_,
            blockInverseOffDiagonal_.data(), vectorBytes,
            cudaMemcpyHostToDevice, stream_), "upload block off-diagonal");
        std::vector<double> scalarInitial(kDeviceScalarCount, 0.0);
        checkCuda(cudaMemcpyAsync(deviceScalars_, scalarInitial.data(),
            scalarInitial.size() * sizeof(double), cudaMemcpyHostToDevice,
            stream_), "initialize device scalars");

        checkCusparse(cusparseCreateCsr(&matrix_, dimension_, dimension_,
            nonzeros_, deviceRowOffsets_, deviceColumnIndices_, deviceValues_,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "create CSR descriptor");
        checkCusparse(cusparseCreateDnVec(&vectorX_, dimension_, deviceX_,
            CUDA_R_64F), "create x descriptor");
        checkCusparse(cusparseCreateDnVec(&vectorDirection_, dimension_,
            deviceDirection_, CUDA_R_64F), "create direction descriptor");
        checkCusparse(cusparseCreateDnVec(&vectorResidual_, dimension_,
            deviceResidual_, CUDA_R_64F), "create residual descriptor");
        checkCusparse(cusparseCreateDnVec(&vectorMatrixDirection_, dimension_,
            deviceMatrixDirection_, CUDA_R_64F),
            "create matrix direction descriptor");
        const double one = 1.0;
        const double zero = 0.0;
        checkCusparse(cusparseSpMV_bufferSize(cusparse_,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matrix_, vectorX_, &zero,
            vectorResidual_, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
            &spmvBufferBytes_), "query SpMV buffer");
        if (spmvBufferBytes_ != 0)
            allocateCuda(reinterpret_cast<std::uint8_t**>(&spmvBuffer_),
                spmvBufferBytes_, "allocate SpMV buffer");
        checkCuda(cudaStreamSynchronize(stream_), "finish CUDA system creation");
    }

    void applyPreconditioner(bool projected)
    {
        constexpr std::int32_t threads = 256;
        const std::int32_t blocks = 1 + (dimension_ - 1) / threads;
        blockJacobiKernel<<<blocks, threads, 0, stream_>>>(dimension_,
            deviceResidual_, deviceInverseDiagonal_, deviceBlockPartner_,
            deviceBlockInverseDiagonal_, deviceBlockInverseOffDiagonal_,
            projected ? deviceFixed_ : nullptr, devicePreconditioned_);
        checkCuda(cudaGetLastError(), "block Jacobi kernel");
    }

    void zeroFixed(double* values)
    {
        constexpr std::int32_t threads = 256;
        const std::int32_t blocks = 1 + (dimension_ - 1) / threads;
        zeroFixedKernel<<<blocks, threads, 0, stream_>>>(
            dimension_, deviceFixed_, values);
        checkCuda(cudaGetLastError(), "zero fixed kernel");
    }

    void spmv(cusparseDnVecDescr_t input, cusparseDnVecDescr_t output,
        double alpha, double beta)
    {
        checkCusparse(cusparseSpMV(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matrix_, input, &beta, output, CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG2, spmvBuffer_), "CSR SpMV");
    }

    void destroy() noexcept
    {
        if (stream_ != nullptr)
            cudaStreamSynchronize(stream_);
        if (vectorMatrixDirection_ != nullptr)
            cusparseDestroyDnVec(vectorMatrixDirection_);
        if (vectorResidual_ != nullptr)
            cusparseDestroyDnVec(vectorResidual_);
        if (vectorDirection_ != nullptr)
            cusparseDestroyDnVec(vectorDirection_);
        if (vectorX_ != nullptr)
            cusparseDestroyDnVec(vectorX_);
        if (matrix_ != nullptr)
            cusparseDestroySpMat(matrix_);
        cudaFree(spmvBuffer_);
        cudaFree(deviceState_);
        cudaFree(deviceScalars_);
        cudaFree(deviceUpdateValues_);
        cudaFree(deviceUpdateIndices_);
        cudaFree(deviceFixed_);
        cudaFree(deviceMatrixDirection_);
        cudaFree(deviceDirection_);
        cudaFree(devicePreconditioned_);
        cudaFree(deviceResidual_);
        cudaFree(deviceX_);
        cudaFree(deviceRhs_);
        cudaFree(deviceBlockInverseOffDiagonal_);
        cudaFree(deviceBlockInverseDiagonal_);
        cudaFree(deviceBlockPartner_);
        cudaFree(deviceInverseDiagonal_);
        cudaFree(deviceValues_);
        cudaFree(deviceColumnIndices_);
        cudaFree(deviceRowOffsets_);
        cudaFreeHost(hostState_);
        cudaFreeHost(hostNorms_);
        if (cusparse_ != nullptr)
            cusparseDestroy(cusparse_);
        if (cublas_ != nullptr)
            cublasDestroy(cublas_);
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    std::int32_t dimension_ = 0;
    std::int32_t nonzeros_ = 0;
    std::vector<std::int32_t> hostRowOffsets_;
    std::vector<std::int32_t> hostColumnIndices_;
    std::vector<double> hostValues_;
    std::vector<double> inverseDiagonal_;
    std::vector<std::int32_t> blockPartner_;
    std::vector<double> blockInverseDiagonal_;
    std::vector<double> blockInverseOffDiagonal_;
    std::vector<std::int32_t> columnOffsets_;
    std::vector<std::int32_t> columnRows_;
    std::vector<double> columnValues_;

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cusparseHandle_t cusparse_ = nullptr;
    std::int32_t* deviceRowOffsets_ = nullptr;
    std::int32_t* deviceColumnIndices_ = nullptr;
    double* deviceValues_ = nullptr;
    double* deviceInverseDiagonal_ = nullptr;
    std::int32_t* deviceBlockPartner_ = nullptr;
    double* deviceBlockInverseDiagonal_ = nullptr;
    double* deviceBlockInverseOffDiagonal_ = nullptr;
    double* deviceRhs_ = nullptr;
    double* deviceX_ = nullptr;
    double* deviceResidual_ = nullptr;
    double* devicePreconditioned_ = nullptr;
    double* deviceDirection_ = nullptr;
    double* deviceMatrixDirection_ = nullptr;
    std::uint8_t* deviceFixed_ = nullptr;
    std::int32_t* deviceUpdateIndices_ = nullptr;
    double* deviceUpdateValues_ = nullptr;
    double* deviceScalars_ = nullptr;
    std::int32_t* deviceState_ = nullptr;
    double* hostNorms_ = nullptr;
    std::int32_t* hostState_ = nullptr;
    cusparseSpMatDescr_t matrix_ = nullptr;
    cusparseDnVecDescr_t vectorX_ = nullptr;
    cusparseDnVecDescr_t vectorDirection_ = nullptr;
    cusparseDnVecDescr_t vectorResidual_ = nullptr;
    cusparseDnVecDescr_t vectorMatrixDirection_ = nullptr;
    void* spmvBuffer_ = nullptr;
    std::size_t spmvBufferBytes_ = 0;
    bool projectedStateReady_ = false;
};

template <typename T>
std::vector<T> copyArray(const py::array_t<T,
    py::array::c_style | py::array::forcecast>& array)
{
    const auto info = array.request();
    const T* data = static_cast<const T*>(info.ptr);
    return std::vector<T>(data, data + info.size);
}

double snappedValue(double value, double period)
{
    return period * std::floor(value / period + 0.5);
}

py::dict solveCuda(
    const py::array_t<std::int32_t,
        py::array::c_style | py::array::forcecast>& rowOffsetsArray,
    const py::array_t<std::int32_t,
        py::array::c_style | py::array::forcecast>& columnIndicesArray,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& valuesArray,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& linearArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& integerIndicesArray,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& periodsArray,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& initialArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& blockPairsArray,
    const std::string& rounding, const std::string& finalSolve,
    double tolerance, std::int32_t maximumIterations,
    double intermediateTolerance,
    std::int32_t intermediateMaximumIterations,
    double multipleRoundingThreshold,
    std::int32_t pcgCheckInterval)
{
    const std::vector<std::int32_t> rowOffsets = copyArray(rowOffsetsArray);
    const std::vector<std::int32_t> columnIndices = copyArray(columnIndicesArray);
    const std::vector<double> values = copyArray(valuesArray);
    std::vector<double> linear = copyArray(linearArray);
    const std::vector<std::int64_t> integerIndices64 = copyArray(integerIndicesArray);
    const std::vector<double> periods = copyArray(periodsArray);
    std::vector<double> initial = copyArray(initialArray);
    const std::vector<std::int64_t> blockPairs64 = copyArray(blockPairsArray);
    if (rowOffsets.size() < 2)
        throw std::invalid_argument("row_offsets is empty");
    const std::size_t dimension = rowOffsets.size() - 1;
    if (linear.size() != dimension || initial.size() != dimension
        || periods.size() != integerIndices64.size()
        || blockPairs64.size() % 2 != 0) {
        throw std::invalid_argument("native array sizes are inconsistent");
    }
    if (rounding != "multiple" && rounding != "greedy")
        throw std::invalid_argument("unsupported rounding policy");
    if (finalSolve != "pcg" && finalSolve != "none")
        throw std::invalid_argument("unsupported final solve policy");
    if (pcgCheckInterval < 1 || pcgCheckInterval > 64)
        throw std::invalid_argument("pcg_check_interval must be in [1, 64]");

    std::vector<std::int32_t> integerIndices;
    integerIndices.reserve(integerIndices64.size());
    for (std::int64_t index : integerIndices64) {
        if (index < 0 || index >= static_cast<std::int64_t>(dimension))
            throw std::invalid_argument("integer index is out of range");
        integerIndices.push_back(static_cast<std::int32_t>(index));
    }
    std::vector<std::pair<std::int32_t, std::int32_t>> blockPairs;
    for (std::size_t offset = 0; offset < blockPairs64.size(); offset += 2) {
        const std::int64_t first = blockPairs64[offset];
        const std::int64_t second = blockPairs64[offset + 1];
        if (first < 0 || second < 0
            || first >= static_cast<std::int64_t>(dimension)
            || second >= static_cast<std::int64_t>(dimension)) {
            throw std::invalid_argument("block pair is out of range");
        }
        blockPairs.emplace_back(static_cast<std::int32_t>(first),
            static_cast<std::int32_t>(second));
    }

    std::vector<double> relaxation;
    std::vector<double> solution;
    std::vector<std::uint8_t> fixed(dimension, 0);
    std::vector<double> fixedValues(dimension, 0.0);
    std::vector<double> rhs = linear;
    std::vector<std::uint8_t> rhsTouched(dimension, 0);
    std::vector<std::int32_t> pendingFixedIndices;
    std::vector<std::int32_t> pendingRhsIndices;
    bool projectedStateInitialized = false;
    std::size_t fixedVariableCount = 0;
    std::size_t roundingBatches = 0;
    std::size_t linearSolves = 0;
    std::size_t linearIterations = 0;
    std::size_t pcgHostSynchronizations = 0;
    std::size_t continuationSolves = 0;
    double finalResidual = std::numeric_limits<double>::infinity();
    bool finalConverged = false;

    {
        py::gil_scoped_release release;
        CudaSystem system(rowOffsets, columnIndices, values, blockPairs);
        SolveResult relaxed = system.solve(
            rhs, initial, fixed, tolerance, maximumIterations,
            pcgCheckInterval);
        ++linearSolves;
        linearIterations += static_cast<std::size_t>(relaxed.iterations);
        pcgHostSynchronizations += static_cast<std::size_t>(
            relaxed.hostSynchronizations);
        if (!relaxed.converged) {
            throw std::runtime_error("CUDA continuous relaxation did not converge ("
                + std::to_string(relaxed.relativeResidual) + ")");
        }
        relaxation = relaxed.x;
        solution = relaxed.x;

        auto commit = [&](const std::vector<std::size_t>& positions) {
            std::vector<std::pair<std::int32_t, double>> newlyFixed;
            newlyFixed.reserve(positions.size());
            for (std::size_t position : positions) {
                const std::int32_t variable = integerIndices[position];
                if (fixed[variable] != 0)
                    continue;
                const double value = snappedValue(
                    solution[variable], periods[position]);
                fixed[variable] = 1;
                fixedValues[variable] = value;
                solution[variable] = 0.0;
                newlyFixed.emplace_back(variable, value);
            }
            for (const auto& item : newlyFixed) {
                pendingFixedIndices.push_back(item.first);
                system.subtractFixedColumn(item.first, item.second, fixed, &rhs,
                    &rhsTouched, &pendingRhsIndices);
            }
            for (const auto& item : newlyFixed) {
                rhs[item.first] = 0.0;
                if (rhsTouched[item.first] == 0) {
                    rhsTouched[item.first] = 1;
                    pendingRhsIndices.push_back(item.first);
                }
            }
            fixedVariableCount += newlyFixed.size();
        };

        auto projectedSolve = [&](double solveTolerance,
                                  std::int32_t solveIterations) {
            std::vector<double> rhsUpdateValues;
            const bool fullRefresh = !projectedStateInitialized;
            if (fullRefresh) {
                system.initializeProjectedState(rhs, solution, fixed);
                projectedStateInitialized = true;
            } else {
                rhsUpdateValues.reserve(pendingRhsIndices.size());
                for (std::int32_t index : pendingRhsIndices)
                    rhsUpdateValues.push_back(rhs[index]);
            }
            const std::vector<std::int32_t> noUpdates;
            const std::vector<std::int32_t>& fixedUpdates = fullRefresh
                ? noUpdates : pendingFixedIndices;
            const std::vector<std::int32_t>& rhsUpdates = fullRefresh
                ? noUpdates : pendingRhsIndices;
            SolveResult solved = system.solveProjected(
                fixedUpdates, rhsUpdates,
                rhsUpdateValues,
                solveTolerance, solveIterations, pcgCheckInterval);
            for (std::int32_t index : pendingRhsIndices)
                rhsTouched[index] = 0;
            pendingFixedIndices.clear();
            pendingRhsIndices.clear();
            ++linearSolves;
            linearIterations += static_cast<std::size_t>(solved.iterations);
            pcgHostSynchronizations += static_cast<std::size_t>(
                solved.hostSynchronizations);
            solution = std::move(solved.x);
            return solved;
        };

        if (rounding == "greedy") {
            if (!integerIndices.empty()) {
                std::vector<std::size_t> positions(integerIndices.size());
                std::iota(positions.begin(), positions.end(), 0);
                commit(positions);
                roundingBatches = 1;
            }
        } else {
            std::size_t fixedIntegerCount = 0;
            const std::size_t noRemaining = integerIndices.size();
            std::vector<std::size_t> remainingPrevious(
                integerIndices.size(), noRemaining);
            std::vector<std::size_t> remainingNext(
                integerIndices.size(), noRemaining);
            std::size_t remainingHead = integerIndices.empty()
                ? noRemaining : 0;
            for (std::size_t position = 0;
                 position < integerIndices.size(); ++position) {
                if (position != 0)
                    remainingPrevious[position] = position - 1;
                if (position + 1 < integerIndices.size())
                    remainingNext[position] = position + 1;
            }
            struct Candidate {
                double residue;
                std::int32_t variable;
                std::size_t position;
            };
            while (fixedIntegerCount < integerIndices.size()) {
                std::vector<Candidate> candidates;
                candidates.reserve(integerIndices.size() - fixedIntegerCount);
                for (std::size_t position = remainingHead;
                     position != noRemaining;
                     position = remainingNext[position]) {
                    const std::int32_t variable = integerIndices[position];
                    const double period = periods[position];
                    const double snapped = snappedValue(solution[variable], period);
                    candidates.push_back({
                        std::abs(solution[variable] - snapped) / period,
                        variable, position });
                }
                std::sort(candidates.begin(), candidates.end(),
                    [](const Candidate& left, const Candidate& right) {
                        if (left.residue != right.residue)
                            return left.residue < right.residue;
                        return left.variable < right.variable;
                    });
                std::vector<std::size_t> selected;
                double residueSum = 0.0;
                for (const Candidate& candidate : candidates) {
                    if (!selected.empty()
                        && residueSum + candidate.residue
                            > multipleRoundingThreshold) {
                        break;
                    }
                    selected.push_back(candidate.position);
                    residueSum += candidate.residue;
                }
                if (selected.empty())
                    throw std::runtime_error("multiple rounding selected no variable");
                commit(selected);
                for (std::size_t position : selected) {
                    const std::size_t previous = remainingPrevious[position];
                    const std::size_t next = remainingNext[position];
                    if (previous == noRemaining)
                        remainingHead = next;
                    else
                        remainingNext[previous] = next;
                    if (next != noRemaining)
                        remainingPrevious[next] = previous;
                    ++fixedIntegerCount;
                }
                ++roundingBatches;
                if (fixedIntegerCount < integerIndices.size()) {
                    SolveResult intermediate = projectedSolve(
                        intermediateTolerance, intermediateMaximumIterations);
                    if (!std::isfinite(intermediate.relativeResidual)
                        || intermediate.relativeResidual
                            > 2.0 * intermediateTolerance) {
                        ++continuationSolves;
                        SolveResult continued = projectedSolve(
                            tolerance, maximumIterations);
                        if (!continued.converged) {
                            throw std::runtime_error(
                                "CUDA projected continuation did not converge ("
                                + std::to_string(continued.relativeResidual) + ")");
                        }
                    }
                }
            }
        }

        const bool hasFree = fixedVariableCount < dimension;
        if (finalSolve == "pcg" && hasFree) {
            SolveResult finalResult = projectedSolve(tolerance, maximumIterations);
            finalResidual = finalResult.relativeResidual;
            finalConverged = finalResult.converged;
            if (!finalResult.converged) {
                throw std::runtime_error("CUDA final PCG did not converge ("
                    + std::to_string(finalResult.relativeResidual) + ")");
            }
        } else if (!hasFree) {
            finalResidual = 0.0;
            finalConverged = true;
        } else {
            std::vector<double> gradient(dimension, 0.0);
            double gradientNormSquared = 0.0;
            double linearNormSquared = 0.0;
            for (std::size_t row = 0; row < dimension; ++row) {
                if (fixed[row] != 0)
                    continue;
                double value = -linear[row];
                for (std::int32_t entry = rowOffsets[row];
                     entry < rowOffsets[row + 1]; ++entry) {
                    const std::int32_t column = columnIndices[entry];
                    value += values[entry] * (fixed[column] != 0
                        ? fixedValues[column] : solution[column]);
                }
                gradientNormSquared += value * value;
                linearNormSquared += linear[row] * linear[row];
            }
            finalResidual = std::sqrt(gradientNormSquared
                / std::max(linearNormSquared,
                    std::numeric_limits<double>::min()));
            finalConverged = finalResidual <= tolerance;
        }
        // Projected iterates deliberately keep fixed coordinates at zero.
        // Materialize lattice values only once, after no more projected solve
        // can use the free-space host vector as its initial state.
        for (std::size_t index = 0; index < dimension; ++index) {
            if (fixed[index] != 0)
                solution[index] = fixedValues[index];
        }
    }

    py::dict result;
    py::array_t<double> outputSolution(solution.size());
    py::array_t<double> outputRelaxation(relaxation.size());
    std::copy(solution.begin(), solution.end(), outputSolution.mutable_data());
    std::copy(relaxation.begin(), relaxation.end(),
        outputRelaxation.mutable_data());
    result["x"] = std::move(outputSolution);
    result["relaxation_x"] = std::move(outputRelaxation);
    result["relative_residual"] = finalResidual;
    result["converged"] = finalConverged;
    result["rounding_batches"] = roundingBatches;
    result["linear_solves"] = linearSolves;
    result["linear_iterations"] = linearIterations;
    result["pcg_host_synchronizations"] = pcgHostSynchronizations;
    result["continuation_solves"] = continuationSolves;
    return result;
}

bool cudaAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace
} // namespace lattice_qp

PYBIND11_MODULE(_core, module)
{
    module.doc() = "Native CUDA backend for lattice-qp";
    module.def("cuda_available", &lattice_qp::cudaAvailable);
    module.def("solve_cuda", &lattice_qp::solveCuda,
        py::arg("row_offsets"), py::arg("column_indices"),
        py::arg("values"), py::arg("g"), py::arg("integer_indices"),
        py::arg("lattice_steps"), py::arg("x0"), py::arg("block_pairs"),
        py::arg("rounding"), py::arg("final_solve"),
        py::arg("tolerance"), py::arg("maximum_iterations"),
        py::arg("intermediate_tolerance"),
        py::arg("intermediate_maximum_iterations"),
        py::arg("multiple_rounding_threshold"),
        py::arg("pcg_check_interval"));
}
