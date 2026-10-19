#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Math/Vector4D.h>
#include <Math/VectorUtil.h>
#include <ROOT/RVec.hxx>

#include "FLAF/include/AnalysisTools.h"

namespace hh_bbww {
    // Minimum of delta(ref, b) over the valid b candidates {fatbjet, bjet1, bjet2}; fallback if none is valid.
    template <typename Delta, typename LVector>
    float MinDeltaToBCands(Delta delta,
                           const LVector& ref,
                           const LorentzVectorM& fatbjet_p4,
                           bool fatbjet_isValid,
                           const LorentzVectorM& bjet1_p4,
                           bool bjet1_isValid,
                           const LorentzVectorM& bjet2_p4,
                           bool bjet2_isValid,
                           float fallback) {
        RVecF deltas;
        if (fatbjet_isValid)
            deltas.push_back(delta(ref, fatbjet_p4));
        if (bjet1_isValid)
            deltas.push_back(delta(ref, bjet1_p4));
        if (bjet2_isValid)
            deltas.push_back(delta(ref, bjet2_p4));
        auto it = std::min_element(deltas.begin(), deltas.end());
        if (it != deltas.end())
            return static_cast<float>(*it);
        return fallback;
    }

    template <typename LVector>
    float MinDeltaRToBCands(const LVector& ref,
                            const LorentzVectorM& fatbjet_p4,
                            bool fatbjet_isValid,
                            const LorentzVectorM& bjet1_p4,
                            bool bjet1_isValid,
                            const LorentzVectorM& bjet2_p4,
                            bool bjet2_isValid,
                            float fallback) {
        auto delta = [](const auto& a, const auto& b) { return ROOT::Math::VectorUtil::DeltaR(a, b); };
        return MinDeltaToBCands(
            delta, ref, fatbjet_p4, fatbjet_isValid, bjet1_p4, bjet1_isValid, bjet2_p4, bjet2_isValid, fallback);
    }

    // Smallest |DeltaPhi| to the b candidates.
    template <typename LVector>
    float MinDeltaPhiToBCands(const LVector& ref,
                              const LorentzVectorM& fatbjet_p4,
                              bool fatbjet_isValid,
                              const LorentzVectorM& bjet1_p4,
                              bool bjet1_isValid,
                              const LorentzVectorM& bjet2_p4,
                              bool bjet2_isValid,
                              float fallback) {
        auto delta = [](const auto& a, const auto& b) { return std::abs(ROOT::Math::VectorUtil::DeltaPhi(a, b)); };
        return MinDeltaToBCands(
            delta, ref, fatbjet_p4, fatbjet_isValid, bjet1_p4, bjet1_isValid, bjet2_p4, bjet2_isValid, fallback);
    }

    // True if (l1 b1, l2 b2) has a smaller summed DeltaR than (l1 b2, l2 b1).
    template <typename LVector1, typename LVector2, typename LVector3, typename LVector4>
    bool IsDiagonalLepBPairing(const LVector1& lep1_p4,
                               const LVector2& lep2_p4,
                               const LVector3& bjet1_p4,
                               const LVector4& bjet2_p4) {
        return (ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet1_p4) +
                ROOT::Math::VectorUtil::DeltaR(lep2_p4, bjet2_p4)) <=
               (ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet2_p4) + ROOT::Math::VectorUtil::DeltaR(lep2_p4, bjet1_p4));
    }

    // Index of the largest score.
    inline size_t ArgMax(const std::vector<double>& scores) {
        auto it = std::max_element(scores.begin(), scores.end());
        return it - scores.begin();
    }
}  // namespace hh_bbww
