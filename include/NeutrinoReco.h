#pragma once

#include <cmath>

#include <Math/Vector4D.h>

#include "FLAF/include/AnalysisTools.h"

namespace hh_bbww {
    // Both neutrino pz solutions of a mass constraint; with a negative discriminant both are its real part.
    template <typename Disc>
    struct NuPzSolutions {
        Disc disc_sqr;
        float pz_pos;
        float pz_neg;
    };

    // Neutrino pz from the on-shell W constraint m(lep + nu) = mW, with the neutrino pT taken from MET.
    template <typename LVector1, typename LVector2, typename MetPt>
    NuPzSolutions<float> SolveNuPzFromW(const LVector1& lep_p4, const LVector2& met_p4, MetPt met_pt) {
        float mw = 80.1f;
        float lep_pt = lep_p4.Pt();
        float lep_E = lep_p4.E();
        float lep_pz = lep_p4.Pz();
        float lambda = mw * mw / 2 + met_p4.Px() * lep_p4.Px() + met_p4.Py() * lep_p4.Py();
        float disc_sqr = lambda * lambda * lep_pz * lep_pz / (lep_pt * lep_pt * lep_pt * lep_pt) -
                         (lep_E * lep_E * met_pt * met_pt - lambda * lambda) / (lep_pt * lep_pt);
        NuPzSolutions<float> sol{disc_sqr, 0.f, 0.f};
        if (disc_sqr > 0) {
            sol.pz_pos = static_cast<float>(lambda * lep_pz / (lep_pt * lep_pt) + std::sqrt(disc_sqr));
            sol.pz_neg = static_cast<float>(lambda * lep_pz / (lep_pt * lep_pt) - std::sqrt(disc_sqr));
        } else {
            sol.pz_pos = static_cast<float>(lambda * lep_pz / (lep_pt * lep_pt));
            sol.pz_neg = static_cast<float>(lambda * lep_pz / (lep_pt * lep_pt));
        }
        return sol;
    }

    // Neutrino pz from the constraint m(vis + nu) = mH, where vis is the hadronic W plus the lepton.
    template <typename LVector1, typename LVector2, typename MetPt>
    auto SolveNuPzFromH(const LVector1& vis_p4, const LVector2& met_p4, MetPt met_pt) {
        float mh = 125.0f;
        float mVis = vis_p4.M();
        float lambda = (mh * mh - mVis * mVis) / 2 + met_p4.Px() * vis_p4.Px() + met_p4.Py() * vis_p4.Py();
        auto a = vis_p4.Pz() * vis_p4.Pz() - vis_p4.E() * vis_p4.E();
        auto b = 2 * lambda * vis_p4.Pz();
        auto c = lambda * lambda - vis_p4.E() * vis_p4.E() * met_pt * met_pt;
        auto disc_sqr = b * b - 4 * a * c;
        NuPzSolutions<decltype(disc_sqr)> sol{disc_sqr, 0.f, 0.f};
        float const eps = 1e-6f;
        if (std::abs(a) < eps) {
            sol.pz_pos = (std::abs(b) > eps) ? static_cast<float>(-c / b) : 0.0f;
            sol.pz_neg = sol.pz_pos;
        } else if (disc_sqr > 0) {
            sol.pz_pos = static_cast<float>(-b / (2 * a) + std::sqrt(disc_sqr) / (2 * a));
            sol.pz_neg = static_cast<float>(-b / (2 * a) - std::sqrt(disc_sqr) / (2 * a));
        } else {
            sol.pz_pos = static_cast<float>(-b / (2 * a));
            sol.pz_neg = sol.pz_pos;
        }
        return sol;
    }

    // Massless neutrino p4 with the transverse momentum of MET and the given pz.
    template <typename LVector, typename MetPt>
    LorentzVectorXYZ NeutrinoP4(const LVector& met_p4, MetPt met_pt, float pz) {
        auto E = std::sqrt(met_pt * met_pt + pz * pz);
        return LorentzVectorXYZ(met_p4.Px(), met_p4.Py(), pz, E);
    }
}  // namespace hh_bbww
