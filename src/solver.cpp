/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "solver.hpp"

#include "cuda_system.hpp"
#include "rounding.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace lattice_qp {

namespace {

std::vector<std::pair<std::int32_t, std::int32_t>> prepareBlockPairs(
    const std::vector<std::int64_t>& blockPairs64, std::size_t dimension)
{
    if (blockPairs64.size() % 2 != 0)
        throw std::invalid_argument("native array sizes are inconsistent");
    std::vector<std::pair<std::int32_t, std::int32_t>> blockPairs;
    blockPairs.reserve(blockPairs64.size() / 2);
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
    return blockPairs;
}

SolverOutput solveWithCudaSystem(const SolverInput& input,
    const SolverOptions& options, CudaSystem& system)
{

    const auto& rowOffsets = input.rowOffsets;
    const auto& columnIndices = input.columnIndices;
    const auto& values = input.values;
    const auto& linear = input.linear;
    const auto& integerIndices64 = input.integerIndices;
    const auto& periods = input.periods;
    const auto& initial = input.initial;
    const auto& incidenceIndptr = input.incidenceIndptr;
    const auto& incidenceIndices = input.incidenceIndices;
    const auto& incidenceCoefficients = input.incidenceCoefficients;
    const auto& baseCones = input.baseCones;
    const auto& minimumCones = input.minimumCones;
    const auto& roundingSelection = options.roundingSelection;
    const auto& roundingProjection = options.roundingProjection;
    const auto& finalSolve = options.finalSolve;
    const double tolerance = options.tolerance;
    const std::int32_t maximumIterations = options.maximumIterations;
    const double intermediateTolerance = options.intermediateTolerance;
    const std::int32_t intermediateMaximumIterations
        = options.intermediateMaximumIterations;
    const double multipleRoundingThreshold
        = options.multipleRoundingThreshold;
    const std::int32_t pcgCheckInterval = options.pcgCheckInterval;

    if (rowOffsets.size() < 2)
        throw std::invalid_argument("row_offsets is empty");
    const std::size_t dimension = rowOffsets.size() - 1;
    if (linear.size() != dimension || initial.size() != dimension
        || periods.size() != integerIndices64.size()) {
        throw std::invalid_argument("native array sizes are inconsistent");
    }
    if (roundingSelection != "sequential"
        && roundingSelection != "multiple" && roundingSelection != "all") {
        throw std::invalid_argument("unsupported rounding selection policy");
    }
    const ProjectionPolicy projection
        = parseProjectionPolicy(roundingProjection);
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
    SolverOutput output;
    std::vector<std::uint8_t> fixed(dimension, 0);
    std::vector<double> fixedValues(dimension, 0.0);
    std::vector<double> rhs = linear;
    std::vector<std::uint8_t> rhsTouched(dimension, 0);
    std::vector<std::int32_t> pendingFixedIndices;
    std::vector<std::int32_t> pendingRhsIndices;
    bool projectedStateInitialized = false;
    std::size_t fixedVariableCount = 0;

    ProjectionState projectionState(projection, integerIndices.size(),
        incidenceIndptr, incidenceIndices, incidenceCoefficients,
        baseCones, minimumCones);
    PcgResult relaxed = system.solve(
        rhs, initial, fixed, tolerance, maximumIterations,
        pcgCheckInterval);
    ++output.linearSolves;
    output.linearIterations += static_cast<std::size_t>(relaxed.iterations);
    output.pcgHostSynchronizations += static_cast<std::size_t>(
        relaxed.hostSynchronizations);
    if (!relaxed.converged) {
        throw std::runtime_error("CUDA continuous relaxation did not converge ("
            + std::to_string(relaxed.relativeResidual) + ")");
    }
    output.relaxationX = relaxed.x;
    output.x = relaxed.x;

    auto commit = [&](const std::vector<std::size_t>& positions) {
        std::vector<std::pair<std::int32_t, double>> newlyFixed;
        newlyFixed.reserve(positions.size());
        for (std::size_t position : positions) {
            const std::int32_t variable = integerIndices[position];
            if (fixed[variable] != 0)
                continue;
            const double coordinate = output.x[variable] / periods[position];
            const std::int64_t latticeValue
                = projectionState.commit(position, coordinate);
            const double value
                = static_cast<double>(latticeValue) * periods[position];
            fixed[variable] = 1;
            fixedValues[variable] = value;
            output.x[variable] = 0.0;
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
            system.initializeProjectedState(rhs, output.x, fixed);
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
        PcgResult solved = system.solveProjected(
            fixedUpdates, rhsUpdates, rhsUpdateValues,
            solveTolerance, solveIterations, pcgCheckInterval);
        for (std::int32_t index : pendingRhsIndices)
            rhsTouched[index] = 0;
        pendingFixedIndices.clear();
        pendingRhsIndices.clear();
        ++output.linearSolves;
        output.linearIterations += static_cast<std::size_t>(solved.iterations);
        output.pcgHostSynchronizations += static_cast<std::size_t>(
            solved.hostSynchronizations);
        output.x = std::move(solved.x);
        return solved;
    };

    if (roundingSelection == "all") {
        if (!integerIndices.empty()) {
            std::vector<std::size_t> positions(integerIndices.size());
            std::iota(positions.begin(), positions.end(), 0);
            commit(positions);
            output.roundingBatches = 1;
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
                const double coordinate
                    = output.x[variable] / periods[position];
                const std::int64_t projected
                    = projectionState.propose(position, coordinate);
                candidates.push_back({
                    std::abs(coordinate - static_cast<double>(projected)),
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
                if (roundingSelection == "sequential" && !selected.empty())
                    break;
                if (!selected.empty()
                    && residueSum + candidate.residue
                        > multipleRoundingThreshold) {
                    break;
                }
                selected.push_back(candidate.position);
                residueSum += candidate.residue;
            }
            if (selected.empty()) {
                throw std::runtime_error(
                    "multiple rounding selected no variable");
            }
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
            ++output.roundingBatches;
            if (fixedIntegerCount < integerIndices.size()) {
                PcgResult intermediate = projectedSolve(
                    intermediateTolerance, intermediateMaximumIterations);
                if (!std::isfinite(intermediate.relativeResidual)
                    || intermediate.relativeResidual
                        > 2.0 * intermediateTolerance) {
                    ++output.continuationSolves;
                    PcgResult continued = projectedSolve(
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
        PcgResult finalResult = projectedSolve(tolerance, maximumIterations);
        output.relativeResidual = finalResult.relativeResidual;
        output.converged = finalResult.converged;
        if (!finalResult.converged) {
            throw std::runtime_error("CUDA final PCG did not converge ("
                + std::to_string(finalResult.relativeResidual) + ")");
        }
    } else if (!hasFree) {
        output.relativeResidual = 0.0;
        output.converged = true;
    } else {
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
                    ? fixedValues[column] : output.x[column]);
            }
            gradientNormSquared += value * value;
            linearNormSquared += linear[row] * linear[row];
        }
        output.relativeResidual = std::sqrt(gradientNormSquared
            / std::max(linearNormSquared,
                std::numeric_limits<double>::min()));
        output.converged = output.relativeResidual <= tolerance;
    }

    // Projected iterates deliberately keep fixed coordinates at zero.
    // Materialize lattice values only once, after no more projected solve can
    // use the free-space host vector as its initial state.
    for (std::size_t index = 0; index < dimension; ++index) {
        if (fixed[index] != 0)
            output.x[index] = fixedValues[index];
    }

    output.correctionCount = projectionState.correctionCount();
    output.coneValues = projectionState.values();
    output.coneViolations = projectionState.violations();
    for (std::int64_t violation : output.coneViolations) {
        if (violation != 0)
            ++output.coneViolationCount;
        output.coneMaxViolation = std::max(
            output.coneMaxViolation, violation);
    }
    return output;
}

} // namespace

SolverOutput solveLatticeQp(const SolverInput& input,
    const SolverOptions& options)
{
    if (input.rowOffsets.size() < 2)
        throw std::invalid_argument("row_offsets is empty");
    const auto blockPairs = prepareBlockPairs(
        input.blockPairs, input.rowOffsets.size() - 1);
    CudaSystem system(input.rowOffsets, input.columnIndices, input.values,
        blockPairs);
    return solveWithCudaSystem(input, options, system);
}

PreparedCudaSolver::PreparedCudaSolver(SolverInput structuralInput)
    : input_(std::move(structuralInput))
{
    if (input_.rowOffsets.size() < 2)
        throw std::invalid_argument("row_offsets is empty");
    const std::size_t dimension = input_.rowOffsets.size() - 1;
    if (input_.periods.size() != input_.integerIndices.size())
        throw std::invalid_argument("native array sizes are inconsistent");
    for (std::int64_t index : input_.integerIndices) {
        if (index < 0 || index >= static_cast<std::int64_t>(dimension))
            throw std::invalid_argument("integer index is out of range");
    }
    const auto blockPairs = prepareBlockPairs(input_.blockPairs, dimension);
    system_ = std::make_unique<CudaSystem>(input_.rowOffsets,
        input_.columnIndices, input_.values, blockPairs);
}

PreparedCudaSolver::~PreparedCudaSolver() = default;

SolverOutput PreparedCudaSolver::solve(std::vector<double> linear,
    std::vector<double> initial,
    std::vector<std::int32_t> incidenceIndptr,
    std::vector<std::int32_t> incidenceIndices,
    std::vector<std::int64_t> incidenceCoefficients,
    std::vector<std::int64_t> baseCones,
    std::vector<std::int64_t> minimumCones,
    const SolverOptions& options)
{
    std::unique_lock<std::mutex> lock(solveMutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        throw std::runtime_error(
            "concurrent solves on one LatticeQPSession are not supported");
    }
    ++stats_.solveCalls;
    if (lifetimeSolveCalls_ != 0)
        ++stats_.cudaSystemReuses;
    ++lifetimeSolveCalls_;
    input_.linear = std::move(linear);
    input_.initial = std::move(initial);
    input_.incidenceIndptr = std::move(incidenceIndptr);
    input_.incidenceIndices = std::move(incidenceIndices);
    input_.incidenceCoefficients = std::move(incidenceCoefficients);
    input_.baseCones = std::move(baseCones);
    input_.minimumCones = std::move(minimumCones);
    return solveWithCudaSystem(input_, options, *system_);
}

PreparedCudaSolverStats PreparedCudaSolver::stats() const
{
    std::unique_lock<std::mutex> lock(solveMutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        throw std::runtime_error(
            "session statistics are unavailable during a solve");
    }
    return stats_;
}

void PreparedCudaSolver::resetStats()
{
    std::unique_lock<std::mutex> lock(solveMutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        throw std::runtime_error(
            "session statistics cannot be reset during a solve");
    }
    stats_ = PreparedCudaSolverStats {};
}

} // namespace lattice_qp
