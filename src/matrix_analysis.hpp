/*
 * Copyright (c) 2026 Hwan Heo <gjghks950@naver.com>. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace lattice_qp {

class CsrMatrixAnalysis final {
public:
    CsrMatrixAnalysis(const std::vector<std::int32_t>& rowOffsets,
        const std::vector<std::int32_t>& columnIndices,
        const std::vector<double>& values,
        const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs);

    std::int32_t dimension() const noexcept { return dimension_; }
    std::int32_t nonzeros() const noexcept { return nonzeros_; }

    const std::vector<std::int32_t>& rowOffsets() const noexcept
    {
        return rowOffsets_;
    }
    const std::vector<std::int32_t>& columnIndices() const noexcept
    {
        return columnIndices_;
    }
    const std::vector<double>& values() const noexcept { return values_; }
    const std::vector<double>& inverseDiagonal() const noexcept
    {
        return inverseDiagonal_;
    }
    const std::vector<std::int32_t>& blockPartner() const noexcept
    {
        return blockPartner_;
    }
    const std::vector<double>& blockInverseDiagonal() const noexcept
    {
        return blockInverseDiagonal_;
    }
    const std::vector<double>& blockInverseOffDiagonal() const noexcept
    {
        return blockInverseOffDiagonal_;
    }

    void subtractFixedColumn(std::int32_t column, double value,
        const std::vector<std::uint8_t>& fixed, std::vector<double>* rhs,
        std::vector<std::uint8_t>* rhsTouched,
        std::vector<std::int32_t>* rhsUpdateIndices) const;

private:
    void preparePreconditioner(
        const std::vector<std::pair<std::int32_t, std::int32_t>>& explicitPairs);
    void prepareColumns();

    std::int32_t dimension_ = 0;
    std::int32_t nonzeros_ = 0;
    std::vector<std::int32_t> rowOffsets_;
    std::vector<std::int32_t> columnIndices_;
    std::vector<double> values_;
    std::vector<double> inverseDiagonal_;
    std::vector<std::int32_t> blockPartner_;
    std::vector<double> blockInverseDiagonal_;
    std::vector<double> blockInverseOffDiagonal_;
    std::vector<std::int32_t> columnOffsets_;
    std::vector<std::int32_t> columnRows_;
    std::vector<double> columnValues_;
};

} // namespace lattice_qp
