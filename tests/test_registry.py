"""Tests for the model/task registry."""
from __future__ import annotations


from yeastbench.adapters.protocols import (
    IntegratedPromoterPanelPredictor,
    VariantEffectScorer,
)
from yeastbench.benchmarks.base import Benchmark
from yeastbench.benchmarks.eqtl import EQTLClassificationBenchmark
from yeastbench.benchmarks.ytk_promoter import YTKPromoterBenchmark
from yeastbench.registry import MODELS, TASKS


class TestRegistry:
    def test_caudal_eqtl_registered(self):
        assert "caudal_eqtl" in TASKS

    def test_shorkie_registered(self):
        assert "shorkie" in MODELS

    def test_yorzoi_registered(self):
        assert "yorzoi" in MODELS

    def test_ytk_promoter_registered(self):
        assert "ytk_promoter" in TASKS

    def test_ytk_promoter_protocol(self):
        assert (
            YTKPromoterBenchmark.adapter_protocol
            is IntegratedPromoterPanelPredictor
        )

    def test_caudal_factory_produces_benchmark(self, synthetic_distribution):
        task = TASKS["caudal_eqtl"](
            distribution_dir=synthetic_distribution,
            fasta_path=synthetic_distribution / "reference" / "R64-1-1.fa",
            gtf_path=synthetic_distribution / "reference" / "R64-1-1.115.gtf",
        )
        assert isinstance(task, Benchmark)
        assert isinstance(task, EQTLClassificationBenchmark)

    def test_caudal_adapter_protocol(self, synthetic_distribution):
        task = TASKS["caudal_eqtl"](
            distribution_dir=synthetic_distribution,
            fasta_path=synthetic_distribution / "reference" / "R64-1-1.fa",
            gtf_path=synthetic_distribution / "reference" / "R64-1-1.115.gtf",
        )
        assert task.adapter_protocol is VariantEffectScorer
