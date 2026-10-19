import uproot, numpy as np

base = "/eos/user/t/thorntoa/DNNDatasets/ResolvedDataset_v2605a_Disco/Dataset"
mass = 600

ev = uproot.open(f"{base}/nParity0_Merged.root")["Events"].arrays(
    ["class_value", "X_mass", "weight_Central"], library="np")
w = uproot.open(f"{base}/nParity0_Merged_weight_m{mass}.root")["weight_tree"].arrays(library="np")

# --- training weight sums (class_weight from the per-mass weight file) ---
sig = w["class_target"] == 0
print(f"--- weight file m{mass} ---")
print("sum sig cw:", w["class_weight"][sig].sum())
print("sum bkg cw:", w["class_weight"][~sig].sum())
print("N events:  ", len(sig))

oth = sig & (ev["X_mass"] != mass)
print("other-mass sig events:", oth.sum())
print("nonzero cw among them:", (w["class_weight"][oth] != 0).sum())

# --- dataset composition (physics yields from weight_Central) ---
print("\n--- composition (weight_Central) ---")
names = {0: "Signal", 1: "TT", 2: "DY", 3: "Other"}
bkg_yield = ev["weight_Central"][ev["class_value"] != 0].sum()
for c in [0, 1, 2, 3]:
    m = ev["class_value"] == c
    y = ev["weight_Central"][m].sum()
    share = f"  ({100*y/bkg_yield:.1f}% of bkg)" if c != 0 else ""
    print(f"{names[c]:7s} n={m.sum():>9,}  yield={y:>10.1f}{share}")
