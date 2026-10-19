#pragma once

#include <cmath>

#include <Math/Vector3D.h>
#include <Math/Vector4D.h>
#include <Math/VectorUtil.h>
#include <TVector2.h>

// HH-system variables ported from hh-italian-group/AnalysisTools, Core/include/AnalysisMath.h @ 2e5ab9a.
namespace hh_bbww {
    // Projection of the dilepton pT plus MET on the bisector zeta of the two lepton directions in the transverse plane.
    template <typename LVector1, typename LVector2, typename LVector3>
    double Calculate_Pzeta(const LVector1& l1_p4, const LVector2& l2_p4, const LVector3& met_p4) {
        const auto ll_p4 = l1_p4 + l2_p4;
        const TVector2 ll_p2(ll_p4.Px(), ll_p4.Py());
        const TVector2 met_p2(met_p4.Px(), met_p4.Py());
        const TVector2 ll_s = ll_p2 + met_p2;
        const TVector2 l1_u(std::cos(l1_p4.Phi()), std::sin(l1_p4.Phi()));
        const TVector2 l2_u(std::cos(l2_p4.Phi()), std::sin(l2_p4.Phi()));
        const TVector2 ll_u = l1_u + l2_u;
        const double ll_u_met = ll_s * ll_u;
        const double ll_mod = ll_u.Mod();
        return ll_u_met / ll_mod;
    }

    // Projection of the dilepton pT on zeta, without MET.
    template <typename LVector1, typename LVector2>
    double Calculate_visiblePzeta(const LVector1& l1_p4, const LVector2& l2_p4) {
        const auto ll_p4 = l1_p4 + l2_p4;
        const TVector2 ll_p2(ll_p4.Px(), ll_p4.Py());
        const TVector2 l1_u(std::cos(l1_p4.Phi()), std::sin(l1_p4.Phi()));
        const TVector2 l2_u(std::cos(l2_p4.Phi()), std::sin(l2_p4.Phi()));
        const TVector2 ll_u = l1_u + l2_u;
        const double ll_p2u = ll_p2 * ll_u;
        const double ll_mod = ll_u.Mod();
        return ll_p2u / ll_mod;
    }

    // Cosine of the angle between h1 and the beam axis in the HH rest frame; odd under h1 <-> h2.
    template <typename LVector1, typename LVector2>
    double Calculate_cosThetaStar(const LVector1& h1, const LVector2& h2) {
        const auto H = h2 + h1;
        const auto boosted_h1 = ROOT::Math::VectorUtil::boost(h1, H.BoostToCM());
        return ROOT::Math::VectorUtil::CosTheta(boosted_h1, ROOT::Math::Cartesian3D<>(0, 0, 1));
    }

    // Cosine of the angle between a daughter in its parent's rest frame and the parent's flight direction.
    template <typename LVector1, typename LVector2>
    double Calculate_cosTheta_2bodies(const LVector1& object1, const LVector2& h) {
        const auto boosted_object1 = ROOT::Math::VectorUtil::boost(object1, h.BoostToCM());
        return ROOT::Math::VectorUtil::CosTheta(boosted_object1, h);
    }

    // DeltaR between two daughters in their parent's rest frame.
    template <typename LVector1, typename LVector2, typename LVector3>
    double Calculate_dR_boosted(const LVector1& particle_1, const LVector2& particle_2, const LVector3& h) {
        const auto boosted_1 = ROOT::Math::VectorUtil::boost(particle_1, h.BoostToCM());
        const auto boosted_2 = ROOT::Math::VectorUtil::boost(particle_2, h.BoostToCM());
        return ROOT::Math::VectorUtil::DeltaR(boosted_1, boosted_2);
    }

    // Angle between the decay planes (a1, a2) of h_a and (b1, b2) of h_b in the HH rest frame.
    template <typename LVector1, typename LVector2, typename LVector3, typename LVector4, typename LVector5, typename LVector6>
    double Calculate_phi(const LVector1& a1,
                         const LVector2& a2,
                         const LVector3& b1,
                         const LVector4& b2,
                         const LVector5& h_a,
                         const LVector6& h_b) {
        const auto H = h_b + h_a;
        const auto boosted_a1 = ROOT::Math::VectorUtil::boost(a1, H.BoostToCM());
        const auto boosted_a2 = ROOT::Math::VectorUtil::boost(a2, H.BoostToCM());
        const auto boosted_b1 = ROOT::Math::VectorUtil::boost(b1, H.BoostToCM());
        const auto boosted_b2 = ROOT::Math::VectorUtil::boost(b2, H.BoostToCM());
        const auto n1 = boosted_a1.Vect().Cross(boosted_a2.Vect());
        const auto n2 = boosted_b1.Vect().Cross(boosted_b2.Vect());
        return ROOT::Math::VectorUtil::Angle(n1, n2);
    }

    // Angle between the decay plane (a1, a2) of h_a and the production plane (h_a, beam axis) in the HH rest frame.
    template <typename LVector1, typename LVector2, typename LVector3, typename LVector4>
    double Calculate_phi1(const LVector1& a1, const LVector2& a2, const LVector3& h_a, const LVector4& h_b) {
        const auto H = h_b + h_a;
        const auto boosted_1 = ROOT::Math::VectorUtil::boost(a1, H.BoostToCM());
        const auto boosted_2 = ROOT::Math::VectorUtil::boost(a2, H.BoostToCM());
        const auto boosted_h = ROOT::Math::VectorUtil::boost(h_a, H.BoostToCM());
        ROOT::Math::Cartesian3D<> z_axis(0, 0, 1);
        const auto n1 = boosted_1.Vect().Cross(boosted_2.Vect());
        const auto n3 = boosted_h.Vect().Cross(z_axis);
        return ROOT::Math::VectorUtil::Angle(n1, n3);
    }

    // Reduced mass m(h1 + h2) - m(h1) - m(h2) + 250 GeV (Calculate_MX with h1 = bb and h2 = ll + MET).
    template <typename LVector1, typename LVector2>
    double Calculate_MX_reduced(const LVector1& h1, const LVector2& h2) {
        static constexpr double shift = 250.;
        return (h1 + h2).M() - h2.M() - h1.M() + shift;
    }
}  // namespace hh_bbww
