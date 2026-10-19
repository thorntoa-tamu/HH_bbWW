from Studies.HME.new.hmeVariables import GetHMEVariables


class HMEProducer:
    def __init__(self, cfg, payload_name, era):
        self.cfg = cfg
        self.payload_name = payload_name
        self.era = era

    def run(self, dfw):
        if "ncentralJet" not in dfw.df.GetColumnNames():
            dfw.Define("ncentralJet", "return centralJet_pt.size();")

        ch = self.cfg["channel"]
        if ch == "DL":
            dfw.Define(
                "has_necessary_inputs",
                "ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0",
            )
        elif ch == "SL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 4 && lep1_pt > 0.0")
        else:
            raise RuntimeError(f"Illegal channel in config: {ch}")

        dfw.df = GetHMEVariables(dfw.df, ch)
        for col in self.cfg["columns"]:
            if col != "valid":
                dfw.DefineAndAppend(
                    f"{self.payload_name}_{col}",
                    f"return hme_output[static_cast<size_t>(HME::EstimOut::{col})];",
                )
        if "valid" in self.cfg["columns"]:
            dfw.DefineAndAppend(
                f"{self.payload_name}_valid", f"return {self.payload_name}_mass > 0.0;"
            )
        return dfw
