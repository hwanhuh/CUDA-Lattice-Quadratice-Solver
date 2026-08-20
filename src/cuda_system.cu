/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_system.hpp"

#include "cuda_kernels.cuh"
#include "matrix_analysis.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

} // namespace

using namespace cuda_detail;

class CudaSystem::Impl {
public:
    Impl(const std::vector<std::int32_t>& rowOffsets,
        const std::vector<std::int32_t>& columnIndices,
        const std::vector<double>& values,
        const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs)
        : analysis_(rowOffsets, columnIndices, values, explicitPairs),
          dimension_(analysis_.dimension()), nonzeros_(analysis_.nonzeros()),
          scalarMatrixValue_(dimension_ == 1
                  ? std::accumulate(analysis_.values().begin(),
                        analysis_.values().end(), 0.0)
                  : 0.0)
    {
        try {
            allocateDevice();
        } catch (...) {
            destroy();
            throw;
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    ~Impl() { destroy(); }

    void subtractFixedColumn(std::int32_t column, double value,
        const std::vector<std::uint8_t>& fixed, std::vector<double>* rhs,
        std::vector<std::uint8_t>* rhsTouched,
        std::vector<std::int32_t>* rhsUpdateIndices) const
    {
        analysis_.subtractFixedColumn(column, value, fixed, rhs, rhsTouched,
            rhsUpdateIndices);
    }

    PcgResult solve(const std::vector<double>& rhs,
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

    PcgResult solveProjected(
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
            if (!newlyFixedIndices.empty()) {
                const std::size_t bytes = newlyFixedIndices.size()
                    * sizeof(std::int32_t);
                checkCuda(cudaMemcpyAsync(deviceUpdateIndices_,
                    newlyFixedIndices.data(), bytes, cudaMemcpyHostToDevice,
                    stream_), "upload projected fixed updates");
                const std::int32_t count = static_cast<std::int32_t>(
                    newlyFixedIndices.size());
                launchApplyProjectedFixedUpdates(count, deviceUpdateIndices_,
                    deviceFixed_, deviceX_, stream_);
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
                launchApplyProjectedRhsUpdates(count, deviceUpdateIndices_,
                    deviceUpdateValues_, deviceRhs_, stream_);
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
    PcgResult solveUploaded(bool projected, double tolerance,
        std::int32_t maximumIterations, std::int32_t checkInterval)
    {
        const std::size_t vectorBytes = static_cast<std::size_t>(dimension_)
            * sizeof(double);
        checkCuda(cudaMemcpyAsync(deviceResidual_, deviceRhs_, vectorBytes,
            cudaMemcpyDeviceToDevice, stream_), "initialize residual");
        spmv(vectorX_, vectorResidual_, deviceX_, deviceResidual_, -1.0, 1.0);
        if (projected)
            zeroFixed(deviceResidual_);

        checkCublas(cublasDdot(cublas_, dimension_, deviceRhs_, 1,
            deviceRhs_, 1, deviceScalars_ + kRhsNormSquared), "rhs dot");
        checkCublas(cublasDdot(cublas_, dimension_, deviceResidual_, 1,
            deviceResidual_, 1, deviceScalars_ + kResidualNormSquared),
            "residual dot");
        launchInitializePcgState(tolerance * tolerance, deviceScalars_,
            deviceState_, stream_);
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
                    spmv(vectorDirection_, vectorMatrixDirection_,
                        deviceDirection_, deviceMatrixDirection_, 1.0, 0.0);
                    checkCublas(cublasDdot(cublas_, dimension_, deviceDirection_, 1,
                        deviceMatrixDirection_, 1,
                        deviceScalars_ + kDirectionProduct), "direction dot");
                    launchUpdateSolutionResidual(dimension_, deviceX_,
                        deviceResidual_, deviceDirection_,
                        deviceMatrixDirection_, deviceScalars_,
                        projected ? deviceFixed_ : nullptr, deviceState_,
                        stream_);
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
                        launchPrepareBeta(deviceScalars_, deviceState_, stream_);
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
        spmv(vectorX_, vectorResidual_, deviceX_, deviceResidual_, -1.0, 1.0);
        if (projected)
            zeroFixed(deviceResidual_);
        checkCublas(cublasDdot(cublas_, dimension_, deviceResidual_, 1,
            deviceResidual_, 1, deviceScalars_ + kResidualNormSquared),
            "explicit residual dot");
        PcgResult result;
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

        const std::size_t rowBytes = analysis_.rowOffsets().size() * sizeof(std::int32_t);
        const std::size_t columnBytes = analysis_.columnIndices().size() * sizeof(std::int32_t);
        const std::size_t valueBytes = analysis_.values().size() * sizeof(double);
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

        checkCuda(cudaMemcpyAsync(deviceRowOffsets_, analysis_.rowOffsets().data(),
            rowBytes, cudaMemcpyHostToDevice, stream_), "upload row offsets");
        checkCuda(cudaMemcpyAsync(deviceColumnIndices_, analysis_.columnIndices().data(),
            columnBytes, cudaMemcpyHostToDevice, stream_), "upload columns");
        checkCuda(cudaMemcpyAsync(deviceValues_, analysis_.values().data(), valueBytes,
            cudaMemcpyHostToDevice, stream_), "upload values");
        checkCuda(cudaMemcpyAsync(deviceInverseDiagonal_, analysis_.inverseDiagonal().data(),
            vectorBytes, cudaMemcpyHostToDevice, stream_), "upload diagonal");
        checkCuda(cudaMemcpyAsync(deviceBlockPartner_, analysis_.blockPartner().data(),
            static_cast<std::size_t>(dimension_) * sizeof(std::int32_t),
            cudaMemcpyHostToDevice, stream_), "upload block partners");
        checkCuda(cudaMemcpyAsync(deviceBlockInverseDiagonal_,
            analysis_.blockInverseDiagonal().data(), vectorBytes,
            cudaMemcpyHostToDevice, stream_), "upload block diagonal");
        checkCuda(cudaMemcpyAsync(deviceBlockInverseOffDiagonal_,
            analysis_.blockInverseOffDiagonal().data(), vectorBytes,
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
        launchBlockJacobi(dimension_,
            deviceResidual_, deviceInverseDiagonal_, deviceBlockPartner_,
            deviceBlockInverseDiagonal_, deviceBlockInverseOffDiagonal_,
            projected ? deviceFixed_ : nullptr, devicePreconditioned_, stream_);
        checkCuda(cudaGetLastError(), "block Jacobi kernel");
    }

    void zeroFixed(double* values)
    {
        launchZeroFixed(dimension_, deviceFixed_, values, stream_);
        checkCuda(cudaGetLastError(), "zero fixed kernel");
    }

    void spmv(cusparseDnVecDescr_t inputDescriptor,
        cusparseDnVecDescr_t outputDescriptor, const double* input,
        double* output, double alpha, double beta)
    {
        if (dimension_ == 1) {
            launchScalarSpmv(scalarMatrixValue_, input, output,
                alpha, beta, stream_);
            checkCuda(cudaGetLastError(), "scalar CSR SpMV kernel");
            return;
        }
        checkCusparse(cusparseSpMV(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matrix_, inputDescriptor, &beta, outputDescriptor, CUDA_R_64F,
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

    CsrMatrixAnalysis analysis_;
    std::int32_t dimension_ = 0;
    std::int32_t nonzeros_ = 0;
    double scalarMatrixValue_ = 0.0;

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


CudaSystem::CudaSystem(const std::vector<std::int32_t>& rowOffsets,
    const std::vector<std::int32_t>& columnIndices,
    const std::vector<double>& values,
    const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs)
    : impl_(std::make_unique<Impl>(rowOffsets, columnIndices, values,
          explicitPairs))
{
}

CudaSystem::~CudaSystem() = default;

void CudaSystem::subtractFixedColumn(std::int32_t column, double value,
    const std::vector<std::uint8_t>& fixed, std::vector<double>* rhs,
    std::vector<std::uint8_t>* rhsTouched,
    std::vector<std::int32_t>* rhsUpdateIndices) const
{
    impl_->subtractFixedColumn(column, value, fixed, rhs, rhsTouched,
        rhsUpdateIndices);
}

PcgResult CudaSystem::solve(const std::vector<double>& rhs,
    const std::vector<double>& initial,
    const std::vector<std::uint8_t>& fixed,
    double tolerance, std::int32_t maximumIterations,
    std::int32_t checkInterval)
{
    return impl_->solve(rhs, initial, fixed, tolerance, maximumIterations,
        checkInterval);
}

void CudaSystem::initializeProjectedState(const std::vector<double>& rhs,
    const std::vector<double>& initial,
    const std::vector<std::uint8_t>& fixed)
{
    impl_->initializeProjectedState(rhs, initial, fixed);
}

PcgResult CudaSystem::solveProjected(
    const std::vector<std::int32_t>& newlyFixedIndices,
    const std::vector<std::int32_t>& rhsUpdateIndices,
    const std::vector<double>& rhsUpdateValues,
    double tolerance, std::int32_t maximumIterations,
    std::int32_t checkInterval)
{
    return impl_->solveProjected(newlyFixedIndices, rhsUpdateIndices,
        rhsUpdateValues, tolerance, maximumIterations, checkInterval);
}

bool cudaAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace lattice_qp
