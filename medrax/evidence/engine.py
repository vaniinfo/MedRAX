"""Evidence Engine v1.0 runtime (specification: EVIDENCE_ENGINE_V1.md).

Loads the frozen parameter bundle, verifies it against its checksums, and turns a CheXagent
p_present into an evidence state. It never fits or updates anything.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

HERE = Path(__file__).parent
BUNDLE = HERE / "evidence_engine_v1.0.json"
CHECKSUMS = HERE / "evidence_engine_v1.0.sha256"

SUPPORTING, UNCERTAIN, REFUTING, INSUFFICIENT = (
    "SUPPORTING", "UNCERTAIN", "REFUTING", "INSUFFICIENT DATA")


@dataclass
class Evidence:
    """CheXagent's evidence about one finding."""

    finding: str
    state: str
    reason: str
    p_present: Optional[float]
    lr: Optional[float] = None
    lr_ci: Optional[Tuple[float, float]] = None
    engine_id: str = ""
    engine: str = ""
    rule: str = ""
    prior: Optional[float] = None
    posterior: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify_checksums(directory: Path = HERE, checksums: Path = CHECKSUMS) -> None:
    """Fail loudly if the specification or parameters differ from the frozen version."""
    for line in checksums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        actual = hashlib.sha256((directory / name.lstrip("*")).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Evidence Engine file {name} does not match its frozen checksum")


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class EvidenceEngine:
    """Maps p_present to SUPPORTING / UNCERTAIN / REFUTING / INSUFFICIENT DATA."""

    def __init__(self, engine: str = "primary", rule: Optional[str] = None,
                 bundle: Path = BUNDLE, verify: bool = True):
        """
        Args:
            engine: "primary" (NIH-derived, production) or "comparator" (Indiana-derived)
            rule: state rule; defaults to the primary rule "primary 5 / 0.2"
            bundle: path to the frozen parameter bundle
            verify: check the bundle and specification against their checksums
        """
        if verify:
            verify_checksums()
        data = json.loads(Path(bundle).read_text(encoding="utf-8"))
        self.engine_id = data["engine_id"]
        self.engine = engine
        self.engine_name = data["engines"][engine]["name"]
        self.params = data["engines"][engine]["findings"]
        self.rule = rule or data["rules"]["primary"]
        self.min_dev_positives = data["min_dev_positives"]
        g = data["grid"]
        self.grid = np.linspace(g["start"], g["stop"], g["num"])
        self.band_edges = np.array(data["band_edges"])

    @property
    def findings(self):
        return list(self.params)

    def assess(self, finding: str, p_present: float, prior: Optional[float] = None) -> Evidence:
        """Evidence state for one finding.

        Args:
            finding: finding name; only the validated findings get an evidence state
            p_present: CheXagent's Yes/(Yes+No) score for "Is there {finding}?"
            prior: optional deployment prevalence; if given, a post-test probability is added
        """
        key = finding.strip().lower()
        base = {"finding": key, "p_present": p_present, "engine_id": self.engine_id,
                "engine": self.engine_name, "rule": self.rule}
        if key not in self.params:
            return Evidence(state=INSUFFICIENT, reason="FINDING_NOT_VALIDATED", **base)

        par = self.params[key]
        p = float(p_present)
        z = _logit(p)
        log_lr = par["platt"]["coef"] * z + par["platt"]["intercept"] - math.log(
            par["dev_prevalence"] / (1 - par["dev_prevalence"]))
        lr = math.exp(min(max(log_lr, -50.0), 50.0))
        ci = (math.exp(float(np.interp(z, self.grid, par["log_lr_ci"]["lower"]))),
              math.exp(float(np.interp(z, self.grid, par["log_lr_ci"]["upper"]))))
        cut = par["cutpoints"][self.rule]

        if par["dev_positives"] < self.min_dev_positives:
            state, reason = INSUFFICIENT, "FEW_DEV_POSITIVES"
        elif not par["support"]["p_min"] <= p <= par["support"]["p_max"]:
            state, reason = INSUFFICIENT, "OUTSIDE_SUPPORT"
        elif cut["p_support"] is not None and p >= cut["p_support"]:
            state, reason = SUPPORTING, "SUPPORTING"
        elif cut["p_refute"] is not None and p <= cut["p_refute"]:
            # REFUTING needs the empirical band to confirm what the Platt LR indicates
            band = int(np.clip(np.searchsorted(self.band_edges, p, side="right") - 1,
                               0, len(self.band_edges) - 2))
            uci = par["empirical_bands"][band]["uci"]
            if uci is not None and uci <= cut["refute_lr"]:
                state, reason = REFUTING, "REFUTING"
            else:
                state, reason = UNCERTAIN, "REFUTING_NOT_EMPIRICALLY_CONFIRMED"
        else:
            state, reason = UNCERTAIN, "WEAK_EVIDENCE"

        evidence = Evidence(state=state, reason=reason, lr=lr, lr_ci=ci, **base)
        if prior is not None and state != INSUFFICIENT:
            posterior_odds = prior / (1 - prior) * lr
            evidence.prior = prior
            evidence.posterior = posterior_odds / (1 + posterior_odds)
        return evidence
