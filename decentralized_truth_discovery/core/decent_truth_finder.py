"""
Enhanced TruthFinder — DECENTRUTH (IEEE INFOCOM 2023)

Implements the CBI-TD (Composite Batch Incremental Truth Discovery) algorithm.

Reference: "A Decentralized Truth Discovery Approach to the Blockchain Oracle Problem"
           IEEE INFOCOM 2023

Core formulas (continuous/numeric data only, per Section III-A of the paper):
  Eq.(3):  x̃_i  = Σ r_s · x_{s,i} / Σ r_s           (local truth estimation)
  Eq.(4):  ε_s   = Σ (x_{s,i} − x̂_i)² / count        (batch MSE per source)
  Eq.(5):  κ_s   = erfc(β|ε_s − ε̂_s|)(1−α) + κ̂_s·α  (consistency measure)
  Eq.(6):  r_s   = κ_s / ε_s                            (reliability update)
  Eq.(8):  w_n   = 1 / d(x̃_n, x̂)                     (node weight by distance)
  Eq.(9):  x̂_i  = Σ w_n · x̃_n / Σ w_n               (global truth)

Data structure: DataFrame with columns
  - website : source identifier
  - fact    : reported numeric value
  - object  : object identifier
"""
import numpy as np
from scipy.special import erfc
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .zk_proof import ZKProofEngine, attach_grouped_proofs


class EnhancedTruthFinder:
    """
    CBI-TD truth discovery per DECENTRUTH (IEEE INFOCOM 2023).

    Input:  DataFrame with columns  website (str), fact (numeric), object (str)
    Output: same DataFrame with added columns:
              source_reliability, error_metric, consistency_metric, global_truth
              [, proof, proof_id, proof_verified]  (when enable_zk_proof=True)
    """

    def __init__(self,
                 alpha: float = 0.9,
                 beta: float = 100.0,
                 initial_reliability: float = 0.5,
                 enable_zk_proof: bool = True,
                 zk_proof_secret: Optional[str] = None):
        """
        Args:
            alpha:                 smoothing factor α ∈ (0,1) for κ update (Eq.5)
            beta:                  scale factor β for erfc consistency measure (Eq.5)
            initial_reliability:   r₀ assigned to each source on first encounter
            enable_zk_proof:       whether to attach ZK proofs to output
            zk_proof_secret:       seed for ZKProofEngine (optional)
        """
        self.alpha = alpha
        self.beta = beta
        self.initial_reliability = initial_reliability
        self.enable_zk_proof = enable_zk_proof
        self.zk_proof_engine = ZKProofEngine(
            prover_id="EnhancedTruthFinder",
            secret_seed=zk_proof_secret,
        )

        # {website: {'r': reliability, 'e': error, 'k': consistency}}
        self.source_reliability_history: Dict[str, Dict[str, float]] = {}
        # {epoch: {object: global_truth_value}}
        self.global_truth_history: Dict[int, Dict[str, float]] = {}
        self.last_iteration_count: int = 0

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_source_reliability(self, dataframe: pd.DataFrame) -> None:
        """Set r=initial_reliability, e=0, k=1 for sources not yet seen."""
        for website in dataframe['website'].unique():
            if website not in self.source_reliability_history:
                self.source_reliability_history[website] = {
                    'r': self.initial_reliability,
                    'e': 0.0,
                    'k': 1.0,
                }

    # ------------------------------------------------------------------
    # Online truth estimate
    # ------------------------------------------------------------------

    def _build_local_estimates(self, dataframe: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Collect each source's reported values: {website: {object: value}}."""
        local_estimates: Dict[str, Dict[str, float]] = {}
        for _, row in dataframe.iterrows():
            if pd.isna(row['fact']):
                continue
            local_estimates.setdefault(str(row['website']), {})[str(row['object'])] = float(row['fact'])
        return local_estimates

    def _initialize_global_truth(
        self, local_estimates: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Eq.(3): x̃_i = Σ r_s · x_{s,i} / Σ r_s
        Reliability-weighted average as the one-shot online truth estimate.
        """
        epsilon = 1e-10
        weighted: Dict[str, List[Tuple[float, float]]] = {}
        for website, object_values in local_estimates.items():
            r = max(epsilon, float(
                self.source_reliability_history.get(website, {}).get('r', self.initial_reliability)
            ))
            for object_id, value in object_values.items():
                weighted.setdefault(object_id, []).append((value, r))

        x_hat: Dict[str, float] = {}
        for object_id, value_weights in weighted.items():
            total_w = sum(w for _, w in value_weights)
            x_hat[object_id] = (
                sum(v * w for v, w in value_weights) / total_w
                if total_w > 0
                else float(np.mean([v for v, _ in value_weights]))
            )
        return x_hat

    def _run_global_td(self, dataframe: pd.DataFrame) -> Dict[str, float]:
        """
        Online one-shot truth estimate for the current batch.
        This intentionally avoids any within-batch fixed-point loop and only
        uses source reliability carried from previous batches.
        """
        local_estimates = self._build_local_estimates(dataframe)
        if not local_estimates:
            return {}
        return self._initialize_global_truth(local_estimates)

    # ------------------------------------------------------------------
    # Reliability update — Eqs.(4)(5)(6)
    # ------------------------------------------------------------------

    def _update_source_reliability(
        self,
        dataframe: pd.DataFrame,
        global_truth: Dict[str, float],
    ) -> None:
        """
        Eqs.(4)(5)(6): update ε, κ, r for every source.
          ε_s  = Σ (x_{s,i} − x̂_i)² / count            (Eq.4 — batch MSE)
          κ_s  = erfc(β|ε_s − ε̂_s|)(1−α) + κ̂_s·α      (Eq.5 — consistency)
          r_s  = κ_s / ε_s                                (Eq.6 — reliability)
        """
        epsilon = 1e-10
        for website in dataframe['website'].unique():
            if website not in self.source_reliability_history:
                self.source_reliability_history[website] = {
                    'r': self.initial_reliability, 'e': 0.0, 'k': 1.0,
                }

            source_data = dataframe[dataframe['website'] == website]
            errors = [
                (float(row['fact']) - float(global_truth[str(row['object'])])) ** 2
                for _, row in source_data.iterrows()
                if str(row['object']) in global_truth
            ]
            if not errors:
                continue

            prev_e = self.source_reliability_history[website]['e']
            prev_k = self.source_reliability_history[website]['k']

            new_e = float(np.mean(errors))                          # Eq.(4)
            new_k = erfc(self.beta * abs(new_e - prev_e))           # erfc part of Eq.(5)
            k_updated = self.alpha * prev_k + (1 - self.alpha) * new_k  # Eq.(5)
            new_r = k_updated / (new_e + epsilon)                   # Eq.(6)

            self.source_reliability_history[website]['e'] = new_e
            self.source_reliability_history[website]['k'] = k_updated
            self.source_reliability_history[website]['r'] = new_r

    # ------------------------------------------------------------------
    # DataFrame output
    # ------------------------------------------------------------------

    def _update_dataframe(
        self,
        dataframe: pd.DataFrame,
        global_truth: Dict[str, float],
    ) -> None:
        """Write result columns: source_reliability, error_metric, consistency_metric, global_truth."""
        dataframe['source_reliability'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('r', self.initial_reliability)
        )
        dataframe['error_metric'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('e', 0.0)
        )
        dataframe['consistency_metric'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('k', 1.0)
        )
        dataframe['global_truth'] = dataframe['object'].map(global_truth)

    # ------------------------------------------------------------------
    # Online batch update
    # ------------------------------------------------------------------

    def iteration(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """One online update: estimate truth once, then update reliability and outputs."""
        global_truth = self._run_global_td(dataframe)
        self._update_source_reliability(dataframe, global_truth)
        self._update_dataframe(dataframe, global_truth)
        return dataframe

    def process_batch(
        self,
        dataframe: pd.DataFrame,
        epoch: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Incremental (one-shot) processing of a single data batch.
        Source reliability state is carried over from previous batches.

        Args:
            dataframe: current batch — columns website, fact (numeric), object
            epoch:     optional epoch index; global truth is saved to self.global_truth_history
        Returns:
            DataFrame with added result columns.
        """
        dataframe = dataframe.copy()
        dataframe['fact'] = pd.to_numeric(dataframe['fact'], errors='raise')

        self._initialize_source_reliability(dataframe)
        dataframe = self.iteration(dataframe)
        self.last_iteration_count = 1

        global_truth = (
            dataframe[["object", "global_truth"]]
            .drop_duplicates(subset=["object"])
            .set_index("object")["global_truth"]
            .astype(float)
            .to_dict()
        )
        if epoch is not None:
            self.global_truth_history[epoch] = global_truth

        if self.enable_zk_proof:
            source_scores = {w: float(v['r']) for w, v in self.source_reliability_history.items()}
            dataframe = attach_grouped_proofs(
                dataframe,
                method_name="EnhancedTruthFinder",
                source_scores=source_scores,
                group_columns=("object",),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
            )

        return dataframe

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:
        """
        Return reliability state for one or all sources.
        Returns dict {'r': float, 'e': float, 'k': float} for a single source,
        or {website: {...}} for all sources.
        """
        if website is not None:
            return self.source_reliability_history.get(website, {}).copy()
        return {w: v.copy() for w, v in self.source_reliability_history.items()}

    def reset_history(self) -> None:
        """Clear all state. Call before processing an independent dataset."""
        self.source_reliability_history = {}
        self.global_truth_history = {}
        self.last_iteration_count = 0
