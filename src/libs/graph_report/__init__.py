"""Graph-report library object: standalone HTML visualization of one question.

Renders the scientific debugging artifact described in the research plan section 17:
stacked chain-of-thought lanes (with branch/join coloring and terminal outcomes),
the consolidated DAG, degree/cluster histograms, and a merge-debug table. It
consumes only plain serializable data assembled by the orchestrator, so it stays
isolated from the generation and reasoning-graph libraries.
"""
from .graph_html_report import GraphHtmlReport
from .tuning_comparison_report import TuningComparisonReport
from .tuning_config_report import TuningConfigReport

__all__ = ["GraphHtmlReport", "TuningComparisonReport", "TuningConfigReport"]
