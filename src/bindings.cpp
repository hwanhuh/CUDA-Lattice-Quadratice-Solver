/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_system.hpp"
#include "solver.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace lattice_qp {
namespace {

template <typename T>
std::vector<T> copyArray(const py::array_t<T,
    py::array::c_style | py::array::forcecast>& array)
{
    const auto info = array.request();
    const T* data = static_cast<const T*>(info.ptr);
    return std::vector<T>(data, data + info.size);
}

SolverOptions makeOptions(const std::string& roundingSelection,
    const std::string& roundingProjection, const std::string& finalSolve,
    double tolerance, std::int32_t maximumIterations,
    double intermediateTolerance,
    std::int32_t intermediateMaximumIterations,
    double multipleRoundingThreshold, std::int32_t pcgCheckInterval)
{
    SolverOptions options;
    options.roundingSelection = roundingSelection;
    options.roundingProjection = roundingProjection;
    options.finalSolve = finalSolve;
    options.tolerance = tolerance;
    options.maximumIterations = maximumIterations;
    options.intermediateTolerance = intermediateTolerance;
    options.intermediateMaximumIterations = intermediateMaximumIterations;
    options.multipleRoundingThreshold = multipleRoundingThreshold;
    options.pcgCheckInterval = pcgCheckInterval;
    return options;
}

py::dict outputDictionary(SolverOutput output)
{
    py::dict result;
    py::array_t<double> outputSolution(output.x.size());
    py::array_t<double> outputRelaxation(output.relaxationX.size());
    py::array_t<std::int64_t> outputConeValues(output.coneValues.size());
    py::array_t<std::int64_t> outputConeViolations(
        output.coneViolations.size());
    std::copy(output.x.begin(), output.x.end(),
        outputSolution.mutable_data());
    std::copy(output.relaxationX.begin(), output.relaxationX.end(),
        outputRelaxation.mutable_data());
    std::copy(output.coneValues.begin(), output.coneValues.end(),
        outputConeValues.mutable_data());
    std::copy(output.coneViolations.begin(), output.coneViolations.end(),
        outputConeViolations.mutable_data());
    result["x"] = std::move(outputSolution);
    result["relaxation_x"] = std::move(outputRelaxation);
    result["relative_residual"] = output.relativeResidual;
    result["converged"] = output.converged;
    result["rounding_batches"] = output.roundingBatches;
    result["linear_solves"] = output.linearSolves;
    result["linear_iterations"] = output.linearIterations;
    result["pcg_host_synchronizations"] = output.pcgHostSynchronizations;
    result["continuation_solves"] = output.continuationSolves;
    result["correction_count"] = output.correctionCount;
    result["cone_values"] = std::move(outputConeValues);
    result["cone_violations"] = std::move(outputConeViolations);
    result["cone_violation_count"] = output.coneViolationCount;
    result["cone_max_violation"] = output.coneMaxViolation;
    result["cone_feasible"] = output.coneViolationCount == 0;
    return result;
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
    const std::string& roundingSelection,
    const std::string& roundingProjection,
    const std::string& finalSolve,
    double tolerance, std::int32_t maximumIterations,
    double intermediateTolerance,
    std::int32_t intermediateMaximumIterations,
    double multipleRoundingThreshold,
    std::int32_t pcgCheckInterval,
    const py::array_t<std::int32_t,
        py::array::c_style | py::array::forcecast>& incidenceIndptrArray,
    const py::array_t<std::int32_t,
        py::array::c_style | py::array::forcecast>& incidenceIndicesArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& incidenceCoefficientsArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& baseConesArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& minimumConesArray)
{
    SolverInput input;
    input.rowOffsets = copyArray(rowOffsetsArray);
    input.columnIndices = copyArray(columnIndicesArray);
    input.values = copyArray(valuesArray);
    input.linear = copyArray(linearArray);
    input.integerIndices = copyArray(integerIndicesArray);
    input.periods = copyArray(periodsArray);
    input.initial = copyArray(initialArray);
    input.blockPairs = copyArray(blockPairsArray);
    input.incidenceIndptr = copyArray(incidenceIndptrArray);
    input.incidenceIndices = copyArray(incidenceIndicesArray);
    input.incidenceCoefficients = copyArray(incidenceCoefficientsArray);
    input.baseCones = copyArray(baseConesArray);
    input.minimumCones = copyArray(minimumConesArray);

    const SolverOptions options = makeOptions(roundingSelection,
        roundingProjection, finalSolve, tolerance, maximumIterations,
        intermediateTolerance, intermediateMaximumIterations,
        multipleRoundingThreshold, pcgCheckInterval);

    SolverOutput output;
    {
        py::gil_scoped_release release;
        output = solveLatticeQp(input, options);
    }

    return outputDictionary(std::move(output));
}

std::unique_ptr<PreparedCudaSolver> createPreparedCudaSolver(
    const py::array_t<std::int32_t,
        py::array::c_style | py::array::forcecast>& rowOffsetsArray,
    const py::array_t<std::int32_t,
        py::array::c_style | py::array::forcecast>& columnIndicesArray,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& valuesArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& integerIndicesArray,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& periodsArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& blockPairsArray)
{
    SolverInput input;
    input.rowOffsets = copyArray(rowOffsetsArray);
    input.columnIndices = copyArray(columnIndicesArray);
    input.values = copyArray(valuesArray);
    input.integerIndices = copyArray(integerIndicesArray);
    input.periods = copyArray(periodsArray);
    input.blockPairs = copyArray(blockPairsArray);
    std::unique_ptr<PreparedCudaSolver> prepared;
    {
        py::gil_scoped_release release;
        prepared = std::make_unique<PreparedCudaSolver>(std::move(input));
    }
    return prepared;
}

py::dict solvePreparedCuda(
    PreparedCudaSolver& solver,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& linearArray,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& initialArray,
    const std::string& roundingSelection,
    const std::string& roundingProjection,
    const std::string& finalSolve,
    double tolerance, std::int32_t maximumIterations,
    double intermediateTolerance,
    std::int32_t intermediateMaximumIterations,
    double multipleRoundingThreshold,
    std::int32_t pcgCheckInterval,
    const py::array_t<std::int32_t,
        py::array::c_style | py::array::forcecast>& incidenceIndptrArray,
    const py::array_t<std::int32_t,
        py::array::c_style | py::array::forcecast>& incidenceIndicesArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& incidenceCoefficientsArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& baseConesArray,
    const py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast>& minimumConesArray)
{
    std::vector<double> linear = copyArray(linearArray);
    std::vector<double> initial = copyArray(initialArray);
    std::vector<std::int32_t> incidenceIndptr = copyArray(
        incidenceIndptrArray);
    std::vector<std::int32_t> incidenceIndices = copyArray(
        incidenceIndicesArray);
    std::vector<std::int64_t> incidenceCoefficients = copyArray(
        incidenceCoefficientsArray);
    std::vector<std::int64_t> baseCones = copyArray(baseConesArray);
    std::vector<std::int64_t> minimumCones = copyArray(minimumConesArray);
    const SolverOptions options = makeOptions(roundingSelection,
        roundingProjection, finalSolve, tolerance, maximumIterations,
        intermediateTolerance, intermediateMaximumIterations,
        multipleRoundingThreshold, pcgCheckInterval);
    SolverOutput output;
    {
        py::gil_scoped_release release;
        output = solver.solve(std::move(linear), std::move(initial),
            std::move(incidenceIndptr), std::move(incidenceIndices),
            std::move(incidenceCoefficients), std::move(baseCones),
            std::move(minimumCones), options);
    }
    return outputDictionary(std::move(output));
}

py::dict preparedStats(const PreparedCudaSolver& solver)
{
    const PreparedCudaSolverStats stats = solver.stats();
    py::dict result;
    result["solve_calls"] = stats.solveCalls;
    result["cuda_system_creations"] = stats.cudaSystemCreations;
    result["cuda_system_reuses"] = stats.cudaSystemReuses;
    return result;
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
        py::arg("rounding_selection"), py::arg("rounding_projection"),
        py::arg("final_solve"),
        py::arg("tolerance"), py::arg("maximum_iterations"),
        py::arg("intermediate_tolerance"),
        py::arg("intermediate_maximum_iterations"),
        py::arg("multiple_rounding_threshold"),
        py::arg("pcg_check_interval"),
        py::arg("incidence_indptr"), py::arg("incidence_indices"),
        py::arg("incidence_coefficients"), py::arg("base_cones"),
        py::arg("minimum_cones"));
    py::class_<lattice_qp::PreparedCudaSolver>(module,
        "PreparedCudaSolver")
        .def(py::init(&lattice_qp::createPreparedCudaSolver),
            py::arg("row_offsets"), py::arg("column_indices"),
            py::arg("values"), py::arg("integer_indices"),
            py::arg("lattice_steps"), py::arg("block_pairs"))
        .def("solve", &lattice_qp::solvePreparedCuda,
            py::arg("g"), py::arg("x0"),
            py::arg("rounding_selection"),
            py::arg("rounding_projection"), py::arg("final_solve"),
            py::arg("tolerance"), py::arg("maximum_iterations"),
            py::arg("intermediate_tolerance"),
            py::arg("intermediate_maximum_iterations"),
            py::arg("multiple_rounding_threshold"),
            py::arg("pcg_check_interval"),
            py::arg("incidence_indptr"), py::arg("incidence_indices"),
            py::arg("incidence_coefficients"), py::arg("base_cones"),
            py::arg("minimum_cones"))
        .def("stats", &lattice_qp::preparedStats)
        .def("reset_stats", &lattice_qp::PreparedCudaSolver::resetStats);
}
