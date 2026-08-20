/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "matrix_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace lattice_qp {

CsrMatrixAnalysis::CsrMatrixAnalysis(
    const std::vector<std::int32_t>& rowOffsets,
    const std::vector<std::int32_t>& columnIndices,
    const std::vector<double>& values,
    const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs)
    : dimension_(static_cast<std::int32_t>(rowOffsets.size() - 1)),
      nonzeros_(static_cast<std::int32_t>(values.size())),
      rowOffsets_(rowOffsets), columnIndices_(columnIndices), values_(values)
{
    if (dimension_ <= 0 || nonzeros_ <= 0
        || columnIndices.size() != values.size()
        || rowOffsets.front() != 0 || rowOffsets.back() != nonzeros_) {
        throw std::invalid_argument("invalid CSR matrix");
    }
    preparePreconditioner(explicitPairs);
    prepareColumns();
}

void CsrMatrixAnalysis::subtractFixedColumn(std::int32_t column, double value,
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

void CsrMatrixAnalysis::preparePreconditioner(
    const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs)
{
    inverseDiagonal_.assign(dimension_, 1.0);
    std::vector<double> diagonal(dimension_, 0.0);
    for (std::int32_t row = 0; row < dimension_; ++row) {
        if (rowOffsets_[row] > rowOffsets_[row + 1])
            throw std::invalid_argument("CSR row offsets must be monotone");
        for (std::int32_t entry = rowOffsets_[row];
             entry < rowOffsets_[row + 1]; ++entry) {
            const std::int32_t column = columnIndices_[entry];
            if (column < 0 || column >= dimension_)
                throw std::invalid_argument("CSR column is out of range");
            const double value = values_[entry];
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "CSR matrix contains non-finite values");
            }
            if (column == row)
                diagonal[row] = value;
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
        std::vector<long double> strongestScore(dimension_, 0.0L);
        for (std::int32_t row = 0; row < dimension_; ++row) {
            const double rowDiagonal = diagonal[row];
            if (!(rowDiagonal > 0.0) || !std::isfinite(rowDiagonal))
                continue;
            for (std::int32_t entry = rowOffsets_[row];
                 entry < rowOffsets_[row + 1]; ++entry) {
                const std::int32_t column = columnIndices_[entry];
                if (column == row)
                    continue;
                const double columnDiagonal = diagonal[column];
                const double coupling = values_[entry];
                if (!(columnDiagonal > 0.0)
                    || !std::isfinite(columnDiagonal)
                    || !std::isfinite(coupling)) {
                    continue;
                }
                const long double normalizedCoupling
                    = static_cast<long double>(coupling)
                    / std::sqrt(static_cast<long double>(rowDiagonal))
                    / std::sqrt(static_cast<long double>(columnDiagonal));
                const long double score
                    = normalizedCoupling * normalizedCoupling;
                if (score > strongestScore[row]
                    || (score == strongestScore[row] && score > 0.0
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
            throw std::invalid_argument(
                "block pairs must be valid and disjoint");
        }
        used[first] = used[second] = 1;
        double offDiagonal = 0.0;
        for (std::int32_t entry = rowOffsets_[first];
             entry < rowOffsets_[first + 1]; ++entry) {
            if (columnIndices_[entry] == second) {
                offDiagonal = values_[entry];
                break;
            }
        }
        if (!(diagonal[first] > 0.0) || !(diagonal[second] > 0.0))
            continue;
        const long double scale = std::max({
            static_cast<long double>(diagonal[first]),
            static_cast<long double>(diagonal[second]),
            std::abs(static_cast<long double>(offDiagonal)) });
        const long double scaledFirst
            = static_cast<long double>(diagonal[first]) / scale;
        const long double scaledSecond
            = static_cast<long double>(diagonal[second]) / scale;
        const long double scaledOffDiagonal
            = static_cast<long double>(offDiagonal) / scale;
        const long double scaledProduct = scaledFirst * scaledSecond;
        const long double scaledDeterminant = std::fma(-scaledOffDiagonal,
            scaledOffDiagonal, scaledProduct);
        if (!(scale > 0.0L) || !std::isfinite(scaledDeterminant)
            || !(scaledDeterminant > 1.0e-10L * scaledProduct)) {
            continue;
        }
        const long double inverseScale = 1.0L / scale;
        const long double firstInverse
            = scaledSecond / scaledDeterminant * inverseScale;
        const long double secondInverse
            = scaledFirst / scaledDeterminant * inverseScale;
        const long double inverseOffDiagonal
            = -scaledOffDiagonal / scaledDeterminant * inverseScale;
        if (!std::isfinite(firstInverse) || !std::isfinite(secondInverse)
            || !std::isfinite(inverseOffDiagonal)
            || std::abs(firstInverse) > std::numeric_limits<double>::max()
            || std::abs(secondInverse) > std::numeric_limits<double>::max()
            || std::abs(inverseOffDiagonal)
                > std::numeric_limits<double>::max()) {
            continue;
        }
        blockPartner_[first] = second;
        blockPartner_[second] = first;
        blockInverseDiagonal_[first] = static_cast<double>(firstInverse);
        blockInverseDiagonal_[second] = static_cast<double>(secondInverse);
        blockInverseOffDiagonal_[first]
            = static_cast<double>(inverseOffDiagonal);
        blockInverseOffDiagonal_[second]
            = static_cast<double>(inverseOffDiagonal);
    }
}

void CsrMatrixAnalysis::prepareColumns()
{
    columnOffsets_.assign(static_cast<std::size_t>(dimension_) + 1, 0);
    for (std::int32_t column : columnIndices_)
        ++columnOffsets_[static_cast<std::size_t>(column) + 1];
    std::partial_sum(columnOffsets_.begin(), columnOffsets_.end(),
        columnOffsets_.begin());
    columnRows_.resize(nonzeros_);
    columnValues_.resize(nonzeros_);
    std::vector<std::int32_t> next = columnOffsets_;
    for (std::int32_t row = 0; row < dimension_; ++row) {
        for (std::int32_t entry = rowOffsets_[row];
             entry < rowOffsets_[row + 1]; ++entry) {
            const std::int32_t column = columnIndices_[entry];
            const std::int32_t destination = next[column]++;
            columnRows_[destination] = row;
            columnValues_[destination] = values_[entry];
        }
    }
}

} // namespace lattice_qp
