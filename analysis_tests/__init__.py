"""Deprecated compatibility namespace for analysis.evaluation."""
from analysis.evaluation.evaluator import Evaluator
from analysis.evaluation.visualizer import ForceAnalyzer, LossPlotter

__all__ = ["Evaluator", "LossPlotter", "ForceAnalyzer"]
