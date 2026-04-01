"""
SenFEED Composite Truth Discovery (CompositeTD)

严格按论文「SenFEED: Dynamic Decentralized Oracle Services for Accurate and Real-Time Sensor Data」
中的真值发现部分（CompositeTD）复现核心公式：

- Eq. (2): 对象真值的加权聚合
- Eq. (3): 迭代式来源权重更新（单调递减函数 f）
- Eq. (6): 增量式 MAP 权重更新
- Eq. (7): 使用历史真值平滑的加权聚合

说明：
1. 该实现仅覆盖“真值发现”部分，不覆盖 DPS-BFT 共识协议。
2. 输入数据默认是数值型传感器数据（论文真值发现实验的主要设定）。
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm

from .zk_proof import ZKProofEngine, attach_grouped_proofs


ModeType = Literal["composite", "iterative", "incremental"]


class SenFeedTruthDiscovery:
    """
    SenFEED 论文真值发现算法（CompositeTD）复现实现。

    Args:
        mode:
            - "composite": CompositeTD（默认，按论文组合策略）
            - "iterative": 仅迭代式（每个时间戳都执行 Eq.2 + Eq.3 至收敛）
            - "incremental": 仅增量式（每个时间戳执行 Eq.6 的 MAP 更新）
        gamma: 论文中的误差尺度系数 γ（用于 Eq.6 分母）
        map_shape_a: MAP 的 Gamma 先验 shape 参数 a（Eq.6）
        map_scale_beta: MAP 的 Gamma 先验 scale 参数 β（Eq.6）
        history_truth_weight: 论文 Eq.7 中历史真值虚拟源权重 w
        evolution_threshold: CompositeTD 的权重波动阈值 ε
        convergence_threshold: 迭代收敛阈值
        max_iterations: 迭代最大轮数
        initial_weight: 来源初始权重
        weight_floor: 权重下限，避免数值不稳定
    """

    def __init__(
        self,
        mode: ModeType = "composite",
        gamma: float = 1.0,
        map_shape_a: float = 2.0,
        map_scale_beta: float = 1.0,
        history_truth_weight: float = 1.0,
        evolution_threshold: float = 0.05,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 200,
        initial_weight: float = 1.0,
        weight_floor: float = 1e-12,
        enable_zk_proof: bool = True,
        zk_proof_secret: Optional[str] = None,
    ):
        assert mode in {"composite", "iterative", "incremental"}, "invalid mode"
        assert gamma > 0, "gamma must be > 0"
        assert map_shape_a > 1.0, "map_shape_a should be > 1 to keep numerator positive"
        assert map_scale_beta > 0, "map_scale_beta must be > 0"
        assert history_truth_weight >= 0, "history_truth_weight must be >= 0"
        assert evolution_threshold >= 0, "evolution_threshold must be >= 0"
        assert convergence_threshold > 0, "convergence_threshold must be > 0"
        assert max_iterations > 0, "max_iterations must be > 0"
        assert initial_weight > 0, "initial_weight must be > 0"
        assert weight_floor > 0, "weight_floor must be > 0"

        self.mode = mode
        self.gamma = gamma
        self.map_shape_a = map_shape_a
        self.map_scale_beta = map_scale_beta
        self.history_truth_weight = history_truth_weight
        self.evolution_threshold = evolution_threshold
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.initial_weight = initial_weight
        self.weight_floor = weight_floor
        self.enable_zk_proof = enable_zk_proof

        # 来源权重 w_{t,s}
        self.source_weights: Dict[str, float] = {}
        # 历史真值 v_{t-1,o}（用于 Eq.7）
        self.previous_truth: Dict[str, float] = {}
        # 增量 MAP 所需累计误差项 Σ(Δv)^2（用于 Eq.6 分母）
        self.cumulative_squared_error: Dict[str, float] = {}
        # 增量 MAP 所需累计观测数 Σ_{t=1}^T Σ_o 𝟙_{s,o}（用于 Eq.6 分子）
        self.cumulative_obs_count: Dict[str, int] = {}
        # 记录最近一次迭代使用的轮数
        self.last_iteration_count: int = 0
        self.zk_proof_engine = ZKProofEngine(
            prover_id="SenFeedTruthDiscovery",
            secret_seed=zk_proof_secret,
        )

    def reset_history(self):
        """重置内部状态。"""
        self.source_weights = {}
        self.previous_truth = {}
        self.cumulative_squared_error = {}
        self.cumulative_obs_count = {}
        self.last_iteration_count = 0

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:
        """获取来源权重。"""
        if website is not None:
            return {"weight": self.source_weights.get(website, self.initial_weight)}
        return {k: {"weight": v} for k, v in self.source_weights.items()}

    def train(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        训练/推理入口，按时间戳顺序处理数据。

        输入列要求：
        - website: 来源
        - object: 对象
        - fact: 数值观测
        - timestamp: 可选；若缺失则视为单个时间戳
        """
        df = self._prepare_dataframe(dataframe)
        self._initialize_sources(df)

        result_chunks = []
        unique_timestamps = sorted(df["timestamp"].unique())

        for i, timestamp in enumerate(unique_timestamps):
            current_df = df[df["timestamp"] == timestamp].copy()
            old_weights = self.source_weights.copy()

            if self.mode == "iterative":
                truth_map = self._run_iterative_truth_discovery(current_df)
                self._accumulate_squared_error(current_df, truth_map)
                td_mode = "iterative"
                weight_evolution = self._compute_weight_evolution(old_weights, self.source_weights)
            elif self.mode == "incremental":
                truth_map, _, _ = self._run_incremental_map_update(current_df)
                td_mode = "incremental"
                weight_evolution = self._compute_weight_evolution(old_weights, self.source_weights)
            else:
                # CompositeTD 策略：
                # 1) t=1 时先用迭代式得到稳定初值
                # 2) 后续先尝试增量式（Eq.6），若权重波动过大则回退到迭代式
                if i == 0:
                    truth_map = self._run_iterative_truth_discovery(current_df)
                    self._accumulate_squared_error(current_df, truth_map)
                    td_mode = "iterative"
                    weight_evolution = self._compute_weight_evolution(old_weights, self.source_weights)
                else:
                    truth_inc, sq_error_this_step, obs_count_this_step = self._run_incremental_map_update(current_df)
                    weight_evolution = self._compute_weight_evolution(old_weights, self.source_weights)

                    # 根据论文中的权重波动思想触发“增量->迭代”切换：
                    # 当来源权重演化量超过阈值 ε 时，回退到 Eq.2 + Eq.3 的迭代式求解。
                    need_iterative = any(
                        delta > self.evolution_threshold
                        for delta in weight_evolution.values()
                    )

                    if need_iterative:
                        # 回滚增量步骤（因为该时刻改用迭代法）
                        self.source_weights = old_weights.copy()
                        for s, e2 in sq_error_this_step.items():
                            self.cumulative_squared_error[s] = max(
                                0.0, self.cumulative_squared_error.get(s, 0.0) - e2
                            )
                        for s, cnt in obs_count_this_step.items():
                            self.cumulative_obs_count[s] = max(
                                0, self.cumulative_obs_count.get(s, 0) - cnt
                            )

                        truth_map = self._run_iterative_truth_discovery(current_df)
                        self._accumulate_squared_error(current_df, truth_map)
                        td_mode = "iterative"
                        weight_evolution = self._compute_weight_evolution(old_weights, self.source_weights)
                    else:
                        truth_map = truth_inc
                        td_mode = "incremental"

            # 更新历史真值（供 Eq.7 使用）
            self.previous_truth = truth_map.copy()

            current_df["global_truth"] = current_df["object"].map(truth_map)
            current_df["source_weight"] = current_df["website"].map(
                lambda s: self.source_weights.get(s, self.initial_weight)
            )
            current_df["td_mode"] = td_mode
            current_df["weight_evolution"] = current_df["website"].map(
                lambda s: weight_evolution.get(s, 0.0)
            )
            result_chunks.append(current_df)

        result_df = pd.concat(result_chunks, ignore_index=True)

        if self.enable_zk_proof:
            result_df = attach_grouped_proofs(
                result_df,
                method_name=f"SenFeedTruthDiscovery[{self.mode}]",
                source_scores=self.source_weights,
                group_columns=("timestamp", "object"),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
                extra_public_data={"mode": self.mode},
            )

        return result_df

    def _prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        required = {"website", "fact", "object"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        df = dataframe.copy()
        if "timestamp" not in df.columns:
            df["timestamp"] = 0

        # 论文真值发现部分以数值传感器数据为主，这里强制数值化。
        df["fact"] = pd.to_numeric(df["fact"], errors="coerce")
        if df["fact"].isna().any():
            bad_rows = df[df["fact"].isna()].head(5)
            raise ValueError(
                "fact column must be numeric for SenFEED truth discovery.\n"
                f"sample invalid rows:\n{bad_rows}"
            )

        df = df.sort_values(["timestamp", "object", "website"]).reset_index(drop=True)
        return df

    def _initialize_sources(self, df: pd.DataFrame):
        for source in df["website"].unique():
            if source not in self.source_weights:
                self.source_weights[source] = self.initial_weight
            if source not in self.cumulative_squared_error:
                self.cumulative_squared_error[source] = 0.0
            if source not in self.cumulative_obs_count:
                self.cumulative_obs_count[source] = 0

    def _compute_weighted_truth(
        self, object_df: pd.DataFrame, include_history_truth: bool
    ) -> float:
        """
        Eq. (2) 与 Eq. (7)：

        Eq. (2):
            v_hat_{t,o} = Σ_s I_{s,o} * w_{t,s} * v_{t,s,o} / Σ_s I_{s,o} * w_{t,s}

        Eq. (7):
            v_hat_{t,o} = (Σ_s I_{s,o} * w_{t,s} * v_{t,s,o} + w * v_hat_{t-1,o})
                         / (Σ_s I_{s,o} * w_{t,s} + w)
        """
        object_id = str(object_df["object"].iloc[0])
        numerator = 0.0
        denominator = 0.0

        for _, row in object_df.iterrows():
            source = str(row["website"])
            value = float(row["fact"])
            w_ts = max(self.weight_floor, self.source_weights.get(source, self.initial_weight))

            numerator += w_ts * value
            denominator += w_ts

        if include_history_truth and object_id in self.previous_truth:
            numerator += self.history_truth_weight * float(self.previous_truth[object_id])
            denominator += self.history_truth_weight

        if denominator <= self.weight_floor:
            return float(object_df["fact"].mean())
        return float(numerator / denominator)

    def _estimate_truth_map(self, df_t: pd.DataFrame, include_history_truth: bool) -> Dict[str, float]:
        truth_map: Dict[str, float] = {}
        for object_id in df_t["object"].unique():
            obj_df = df_t[df_t["object"] == object_id]
            truth_map[str(object_id)] = self._compute_weighted_truth(obj_df, include_history_truth)
        return truth_map

    def _collect_error_statistics(
        self, df_t: pd.DataFrame, truth_map: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        abs_error_sum: Dict[str, float] = {}
        squared_error_sum: Dict[str, float] = {}
        observation_count: Dict[str, int] = {}

        for _, row in df_t.iterrows():
            source = str(row["website"])
            object_id = str(row["object"])
            value = float(row["fact"])
            truth = float(truth_map[object_id])
            diff = value - truth

            abs_error_sum[source] = abs_error_sum.get(source, 0.0) + abs(diff)
            squared_error_sum[source] = squared_error_sum.get(source, 0.0) + diff * diff
            observation_count[source] = observation_count.get(source, 0) + 1

        return abs_error_sum, squared_error_sum, observation_count

    def _run_iterative_truth_discovery(
        self,
        df_t: pd.DataFrame,
        include_history_truth: bool = False,
    ) -> Dict[str, float]:
        """
        迭代式真值发现（Eq.2 + Eq.3）。

        Eq. (3) 仅要求 f(·) 为单调递减函数：
            w_{t,s} = f( Σ_o d(v_{t,s,o}, v_hat_{t,o}) )
        这里采用经典可解释实例：
            f(x) = 1 / (x + eps)
        """
        sources = [str(s) for s in df_t["website"].unique()]
        prev = np.array([self.source_weights[s] for s in sources], dtype=float)
        truth_map: Dict[str, float] = {}

        for iteration in range(self.max_iterations):
            truth_map = self._estimate_truth_map(
                df_t,
                include_history_truth=include_history_truth,
            )
            abs_err, _, _ = self._collect_error_statistics(df_t, truth_map)

            new_weights: Dict[str, float] = {}
            for s in sources:
                err = abs_err.get(s, 0.0)
                new_weights[s] = 1.0 / (err + self.weight_floor)

            # 归一化到均值为 1，避免尺度漂移影响收敛判定。
            mean_w = float(np.mean(list(new_weights.values()))) if new_weights else 1.0
            if mean_w > self.weight_floor:
                for s in new_weights:
                    new_weights[s] /= mean_w

            current = np.array([new_weights[s] for s in sources], dtype=float)
            delta = norm(current - prev)

            for s in sources:
                self.source_weights[s] = max(self.weight_floor, float(new_weights[s]))

            self.last_iteration_count = iteration + 1
            if delta < self.convergence_threshold:
                break
            prev = current

        return truth_map

    def _run_incremental_map_update(
        self, df_t: pd.DataFrame
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        增量 MAP 更新（Eq.6），并使用 Eq.7 计算当前真值。

        Eq. (6):
            w_{t,s} = (2a - 2 + Σ_{τ<=t}Σ_o I_{τ,s,o}) / (2β + γ*Σ_{τ<=t}Σ_o (Δv_{τ,s,o})^2)
        """
        truth_map = self._estimate_truth_map(df_t, include_history_truth=True)
        _, squared_err, obs_count = self._collect_error_statistics(df_t, truth_map)

        for source in [str(s) for s in df_t["website"].unique()]:
            self.cumulative_squared_error[source] = (
                self.cumulative_squared_error.get(source, 0.0) + squared_err.get(source, 0.0)
            )
            self.cumulative_obs_count[source] = (
                self.cumulative_obs_count.get(source, 0) + obs_count.get(source, 0)
            )

            numerator = (2.0 * self.map_shape_a - 2.0) + float(self.cumulative_obs_count[source])
            denominator = (2.0 * self.map_scale_beta) + self.gamma * self.cumulative_squared_error[source]

            if denominator <= self.weight_floor:
                new_weight = self.weight_floor
            else:
                new_weight = numerator / denominator

            self.source_weights[source] = max(self.weight_floor, float(new_weight))

        self.last_iteration_count = 1
        return truth_map, squared_err, obs_count

    def _accumulate_squared_error(self, df_t: pd.DataFrame, truth_map: Dict[str, float]):
        """把当前时间戳误差累计到 Eq.6 的历史项。"""
        _, squared_err, _ = self._collect_error_statistics(df_t, truth_map)
        for source, e2 in squared_err.items():
            self.cumulative_squared_error[source] = self.cumulative_squared_error.get(source, 0.0) + e2

    @staticmethod
    def _compute_weight_evolution(old_w: Dict[str, float], new_w: Dict[str, float]) -> Dict[str, float]:
        evolution: Dict[str, float] = {}
        all_sources = set(old_w.keys()) | set(new_w.keys())
        for source in all_sources:
            evolution[source] = abs(float(new_w.get(source, 0.0)) - float(old_w.get(source, 0.0)))
        return evolution


class SenFeedCompositeTD(SenFeedTruthDiscovery):
    """CompositeTD 模式快捷类。"""

    def __init__(self, **kwargs):
        super().__init__(mode="composite", **kwargs)


class SenFeedIterativeTD(SenFeedTruthDiscovery):
    """迭代模式快捷类（每轮都执行 Eq.2 + Eq.3）。"""

    def __init__(self, **kwargs):
        super().__init__(mode="iterative", **kwargs)


class SenFeedIncrementalTD(SenFeedTruthDiscovery):
    """增量模式快捷类（每轮都执行 Eq.6 + Eq.7）。"""

    def __init__(self, **kwargs):
        super().__init__(mode="incremental", **kwargs)
