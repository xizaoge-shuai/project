from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TextLike = Union[str, Sequence[str]]


@dataclass
class VerifierConfig:
    """
    轻量 PCE 配置。
    第一版使用 TF-IDF + LogisticRegression。
    """

    max_features: int = 4000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 1.0
    lowercase: bool = True
    sublinear_tf: bool = True

    # LogisticRegression
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = None
    solver: str = "liblinear"
    random_state: int = 42


class VerifierPCE:
    """
    Prefix Confidence Estimator (PCE) 的正式第一版实现。

    当前实现：
    - 文本输入：question + context + prefix + current_answer + prefix_len
    - 输出：success probability（当前 prefix 最终走向正确答案的概率）

    主要接口：
    - fit(texts, labels)
    - predict_proba_texts(texts)
    - predict(text) / predict(question, text)
    - predict_batch(texts)
    """

    def __init__(
        self,
        max_features: int = 4000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        lowercase: bool = True,
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = None,
        solver: str = "liblinear",
        random_state: int = 42,
    ):
        self.config = VerifierConfig(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
            sublinear_tf=sublinear_tf,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
            random_state=random_state,
        )

        self.pipeline: Pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.config.max_features,
                        ngram_range=self.config.ngram_range,
                        min_df=self.config.min_df,
                        max_df=self.config.max_df,
                        lowercase=self.config.lowercase,
                        sublinear_tf=self.config.sublinear_tf,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=self.config.C,
                        max_iter=self.config.max_iter,
                        class_weight=self.config.class_weight,
                        solver=self.config.solver,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

        self.is_fitted: bool = False
        self.label_positive: int = 1
        self.label_negative: int = 0

    def info(self) -> Dict[str, Any]:
        return {
            "name": "VerifierPCE",
            "type": "tfidf_logreg",
            "config": asdict(self.config),
            "is_fitted": self.is_fitted,
        }

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "VerifierPCE":
        texts = [self._safe_text(t) for t in texts]
        labels = [int(x) for x in labels]

        if len(texts) == 0:
            raise ValueError("Empty training texts.")
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        if len(set(labels)) < 2:
            raise ValueError("Training labels must contain at least two classes.")

        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        return self

    def predict_proba_texts(self, texts: Sequence[str]) -> List[float]:
        self._check_fitted()
        texts = [self._safe_text(t) for t in texts]

        probs = self.pipeline.predict_proba(texts)
        # sklearn 返回 shape = [N, num_classes]
        # 默认正类取 label=1 对应的那一列
        class_to_index = {
            int(cls): idx
            for idx, cls in enumerate(self.pipeline.named_steps["clf"].classes_)
        }
        if self.label_positive not in class_to_index:
            raise ValueError(
                f"Positive label {self.label_positive} not found in classifier classes."
            )

        pos_idx = class_to_index[self.label_positive]
        return [float(x) for x in probs[:, pos_idx]]

    def predict_batch(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        probs = self.predict_proba_texts(texts)
        results: List[Dict[str, Any]] = []
        for p in probs:
            pred = 1 if p >= 0.5 else 0
            results.append(
                {
                    "success_prob": float(p),
                    "pred_label": int(pred),
                }
            )
        return results

    def predict_proba(self, texts: TextLike) -> Union[List[float], np.ndarray]:
        """
        兼容部分旧代码风格：
        - 输入单个 str -> 返回 [prob]
        - 输入多个 texts -> 返回 List[prob]
        """
        if isinstance(texts, str):
            return self.predict_proba_texts([texts])
        return self.predict_proba_texts(texts)

    def predict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        兼容多种旧接口调用方式：

        1) predict(text)
        2) predict(question, text)   # 忽略 question，仅使用 text
        3) predict(text="...")
        """
        self._check_fitted()

        text: Optional[str] = None

        if "text" in kwargs:
            text = kwargs["text"]
        elif len(args) == 1:
            text = args[0]
        elif len(args) >= 2:
            # 兼容旧写法 predict(question, txt)
            text = args[1]

        if text is None:
            raise ValueError("predict() requires a text input.")

        prob = self.predict_proba_texts([text])[0]
        pred = 1 if prob >= 0.5 else 0

        return {
            "success_prob": float(prob),
            "pred_label": int(pred),
        }

    def decision_function(self, texts: Sequence[str]) -> List[float]:
        """
        返回分类 margin，可用于后续分析或自定义校准。
        """
        self._check_fitted()
        texts = [self._safe_text(t) for t in texts]
        clf = self.pipeline.named_steps["clf"]
        feats = self.pipeline.named_steps["tfidf"].transform(texts)

        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(feats)
            scores = np.asarray(scores).reshape(-1)
            return [float(x) for x in scores]

        # 如果没有 decision_function，就退化成 logit(prob)
        probs = np.asarray(self.predict_proba_texts(texts))
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        logits = np.log(probs / (1 - probs))
        return [float(x) for x in logits]

    def get_feature_names(self) -> List[str]:
        self._check_fitted()
        vec = self.pipeline.named_steps["tfidf"]
        if hasattr(vec, "get_feature_names_out"):
            return list(vec.get_feature_names_out())
        return []

    def top_positive_features(self, k: int = 30) -> List[Dict[str, Any]]:
        """
        查看对“正确前缀”最有利的特征。
        """
        self._check_fitted()
        clf = self.pipeline.named_steps["clf"]
        vec = self.pipeline.named_steps["tfidf"]

        if not hasattr(clf, "coef_"):
            return []

        coef = clf.coef_[0]
        feats = vec.get_feature_names_out()
        order = np.argsort(-coef)[:k]

        return [{"feature": str(feats[i]), "weight": float(coef[i])} for i in order]

    def top_negative_features(self, k: int = 30) -> List[Dict[str, Any]]:
        """
        查看对“错误前缀”更敏感的特征。
        """
        self._check_fitted()
        clf = self.pipeline.named_steps["clf"]
        vec = self.pipeline.named_steps["tfidf"]

        if not hasattr(clf, "coef_"):
            return []

        coef = clf.coef_[0]
        feats = vec.get_feature_names_out()
        order = np.argsort(coef)[:k]

        return [{"feature": str(feats[i]), "weight": float(coef[i])} for i in order]

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("VerifierPCE is not fitted yet.")

    @staticmethod
    def _safe_text(text: Any) -> str:
        if text is None:
            return ""
        return str(text).strip()
