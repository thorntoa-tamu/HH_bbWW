#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Math/Vector4D.h>
#include <Math/VectorUtil.h>
#include <ROOT/RVec.hxx>

#include "FLAF/include/AnalysisTools.h"

namespace hh_bbww {
    // True if ref is closer in DeltaR to a than to b.
    template <typename LVector1, typename LVector2, typename LVector3>
    bool IsCloserToFirst(const LVector1& ref, const LVector2& a, const LVector3& b) {
        float dr_a = ROOT::Math::VectorUtil::DeltaR(ref, a);
        float dr_b = ROOT::Math::VectorUtil::DeltaR(ref, b);
        return dr_a < dr_b;
    }

    // Indices in bcands of the b candidate of the leptonic top (closest to the lepton) and of the
    // hadronic top (the other candidate closest to the hadronic W).
    inline std::pair<size_t, size_t> AssignTopBCands(const std::vector<LorentzVectorM>& bcands,
                                                     const LorentzVectorM& lep1_p4,
                                                     const LorentzVectorM& hadW_p4) {
        auto dr_cmp_lep = [&lep1_p4](LorentzVectorM const& v1, LorentzVectorM const& v2) {
            return ROOT::Math::VectorUtil::DeltaR(v1, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(v2, lep1_p4);
        };
        auto it = std::min_element(bcands.begin(), bcands.end(), dr_cmp_lep);
        size_t lep_idx = it - bcands.begin();
        size_t had_idx = (lep_idx == 0) ? 1 : 0;
        float min_dr = ROOT::Math::VectorUtil::DeltaR(bcands[had_idx], hadW_p4);
        for (size_t i = 0; i < bcands.size(); ++i) {
            float dr = ROOT::Math::VectorUtil::DeltaR(bcands[i], hadW_p4);
            if (i != lep_idx && dr < min_dr) {
                min_dr = dr;
                had_idx = i;
            }
        }
        return {lep_idx, had_idx};
    }

    // SL top candidates as lists of constituents: [0] hadronic top (b, W jets or fat W),
    // [1] and [2] leptonic top (b, leptonic W) with the positive and negative neutrino pz solution.
    inline RVecVec<LorentzVectorM> BuildTopCandidates(bool resolved,
                                                      bool res2b,
                                                      bool boosted,
                                                      bool DL,
                                                      const LorentzVectorM& lep1_p4,
                                                      const LorentzVectorM& bjet1_p4,
                                                      bool bjet1_isValid,
                                                      const LorentzVectorM& bjet2_p4,
                                                      bool bjet2_isValid,
                                                      const LorentzVectorM& fatbjet_p4,
                                                      bool fatbjet_isValid,
                                                      const LorentzVectorM& wjet1_p4,
                                                      bool wjet1_isValid,
                                                      const LorentzVectorM& wjet2_p4,
                                                      bool wjet2_isValid,
                                                      const LorentzVectorM& fatwjet_p4,
                                                      bool fatwjet_isValid,
                                                      bool WJets_Boosted,
                                                      const LorentzVectorM& lepW_pos_p4,
                                                      const LorentzVectorM& lepW_neg_p4) {
        RVecVec<LorentzVectorM> tops(3, RVecLV{});
        auto set_lep_tops = [&](const LorentzVectorM& b) {
            tops[1] = {b, lepW_pos_p4};
            tops[2] = {b, lepW_neg_p4};
        };
        const bool wjets_resolved = wjet1_isValid && wjet2_isValid && !WJets_Boosted;
        if ((resolved || res2b) && !DL) {
            const bool b1_lep = IsCloserToFirst(lep1_p4, bjet1_p4, bjet2_p4);
            tops[0] = {b1_lep ? bjet2_p4 : bjet1_p4, wjet1_p4, wjet2_p4};
            set_lep_tops(b1_lep ? bjet1_p4 : bjet2_p4);
        } else if (boosted) {
            if (fatwjet_isValid && fatbjet_isValid) {
                if (bjet1_isValid && bjet2_isValid) {
                    const std::vector<LorentzVectorM> bcands = {bjet1_p4, bjet2_p4, fatbjet_p4};
                    const LorentzVectorM hadW_p4 = wjets_resolved ? (wjet1_p4 + wjet2_p4) : fatwjet_p4;
                    const auto [lep_idx, had_idx] = AssignTopBCands(bcands, lep1_p4, hadW_p4);
                    if (wjets_resolved)
                        tops[0] = {bcands[had_idx], wjet1_p4, wjet2_p4};
                    else
                        tops[0] = {bcands[had_idx], fatwjet_p4};
                    set_lep_tops(bcands[lep_idx]);
                } else if (bjet1_isValid || bjet2_isValid) {
                    const LorentzVectorM bjet_p4 = bjet1_isValid ? bjet1_p4 : bjet2_p4;
                    const bool bjet_lep = IsCloserToFirst(lep1_p4, bjet_p4, fatbjet_p4);
                    const LorentzVectorM& had_b = bjet_lep ? fatbjet_p4 : bjet_p4;
                    if (wjets_resolved)
                        tops[0] = {had_b, wjet1_p4, wjet2_p4};
                    else
                        tops[0] = {had_b, fatwjet_p4};
                    set_lep_tops(bjet_lep ? bjet_p4 : fatbjet_p4);
                }
            } else if (!fatwjet_isValid && fatbjet_isValid) {
                if (bjet1_isValid && bjet2_isValid) {
                    if (!wjet1_isValid || !wjet2_isValid)
                        return tops;
                    const std::vector<LorentzVectorM> bcands = {bjet1_p4, bjet2_p4, fatbjet_p4};
                    const auto [lep_idx, had_idx] = AssignTopBCands(bcands, lep1_p4, wjet1_p4 + wjet2_p4);
                    tops[0] = {bcands[had_idx], wjet1_p4, wjet2_p4};
                    set_lep_tops(bcands[lep_idx]);
                } else if ((bjet1_isValid || bjet2_isValid) && wjet1_isValid && wjet2_isValid) {
                    const LorentzVectorM bjet_p4 = bjet1_isValid ? bjet1_p4 : bjet2_p4;
                    const bool bjet_lep = IsCloserToFirst(lep1_p4, bjet_p4, fatbjet_p4);
                    tops[0] = {bjet_lep ? fatbjet_p4 : bjet_p4, wjet1_p4, wjet2_p4};
                    set_lep_tops(bjet_lep ? bjet_p4 : fatbjet_p4);
                }
            } else if (fatwjet_isValid && !fatbjet_isValid) {
                if (bjet1_isValid && bjet2_isValid) {
                    const bool b1_lep = IsCloserToFirst(lep1_p4, bjet1_p4, bjet2_p4);
                    tops[0] = {b1_lep ? bjet2_p4 : bjet1_p4, fatwjet_p4};
                    set_lep_tops(b1_lep ? bjet1_p4 : bjet2_p4);
                }
            }
        }
        return tops;
    }

    template <typename LVector>
    LVector SumP4(const ROOT::VecOps::RVec<LVector>& p4s) {
        LVector res;
        for (auto const& v : p4s)
            res += v;
        return res;
    }

    // True if the positive solution is further from ref in |DeltaPhi| than the negative one.
    template <typename LVector1, typename LVector2, typename LVector3>
    bool PickPositiveSolution(const LVector1& ref, const LVector2& pos, const LVector3& neg) {
        float dphi_pos = std::abs(ROOT::Math::VectorUtil::DeltaPhi(ref, pos));
        float dphi_neg = std::abs(ROOT::Math::VectorUtil::DeltaPhi(ref, neg));
        return dphi_pos > dphi_neg;
    }

    // Transverse mass of the leptonic top (neutrino + lepton + b) for the chosen neutrino solution.
    template <typename LVector1, typename LVector2, typename LVector3>
    float LepTopMT(bool top_solution_tag,
                   const LVector1& nu_pos_p4,
                   const LVector2& nu_neg_p4,
                   const RVecVec<LorentzVectorM>& tops,
                   const LVector3& lep1_p4) {
        VectorXY<float> nu_t;
        LorentzVectorM bjet_p4;
        if (top_solution_tag) {
            nu_t = VectorXY<float>(nu_pos_p4.Px(), nu_pos_p4.Py());
            bjet_p4 = tops[1].empty() ? LorentzVectorM() : tops[1][0];
        } else {
            nu_t = VectorXY<float>(nu_neg_p4.Px(), nu_neg_p4.Py());
            bjet_p4 = tops[2].empty() ? LorentzVectorM() : tops[2][0];
        }
        VectorXY<float> lep1_t = VectorXY<float>(lep1_p4.Px(), lep1_p4.Py());
        VectorXY<float> bjet_t = VectorXY<float>(bjet_p4.Px(), bjet_p4.Py());
        VectorXY<float> total_transverse_momentum = nu_t + lep1_t + bjet_t;

        float total_transverse_energy =
            std::sqrt(nu_t.Mag2()) + std::sqrt(lep1_t.Mag2() + lep1_p4.M2()) + std::sqrt(bjet_t.Mag2() + bjet_p4.M2());

        float mt_square = total_transverse_energy * total_transverse_energy - total_transverse_momentum.Mag2();
        return static_cast<float>(mt_square > 0.0f ? std::sqrt(mt_square) : -1.0f);
    }

    // pT of the hadronic top over the scalar sum of its constituents' pT.
    template <typename LVector>
    float HadTopConstituentPtFrac(const RVecLV& constituents, const LVector& top_p4) {
        if (constituents.empty())
            return 0.0f;
        float sum_pt = 0.0f;
        for (auto const& p : constituents)
            sum_pt += p.Pt();
        return static_cast<float>(top_p4.Pt() / sum_pt);
    }

    // pT of the leptonic top over pT(b) + MET + pT(lepton), for the chosen neutrino solution.
    template <typename LVector, typename MetPt, typename LepPt>
    float LepTopConstituentPtFrac(bool top_solution_tag,
                                  const RVecVec<LorentzVectorM>& tops,
                                  const LVector& lepT_p4,
                                  MetPt met_pt,
                                  LepPt lep1_pt) {
        if (top_solution_tag)
            return static_cast<float>(tops[1].empty() ? 0.0f : lepT_p4.Pt() / (tops[1][0].Pt() + met_pt + lep1_pt));
        else
            return static_cast<float>(tops[2].empty() ? 0.0f : lepT_p4.Pt() / (tops[2][0].Pt() + met_pt + lep1_pt));
    }

    // Invariant mass of the leptonic top's b candidate and the lepton, for the chosen neutrino solution.
    template <typename LVector>
    float LepTopBLepMass(bool top_solution_tag, const RVecVec<LorentzVectorM>& tops, const LVector& lep1_p4) {
        if (top_solution_tag)
            return static_cast<float>(tops[1].empty() ? -1.0f : (tops[1][0] + lep1_p4).M());
        else
            return static_cast<float>(tops[2].empty() ? -1.0f : (tops[2][0] + lep1_p4).M());
    }
}  // namespace hh_bbww
