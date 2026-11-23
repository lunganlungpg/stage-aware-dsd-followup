"""Stage-aware follow-up designs for definitive screening experiments.

Companion code for Lunganlung P G and Prabhakar Verma, "Stage-aware follow-up
designs for definitive screening experiments: exact criteria, rank completion
and pooled-criterion bias" (preprint).
"""
from .bounds import relaxation_bound, trace_bound
from .criteria import (FollowUpProblem, cube_moments, f_matrix, minimum_followup_runs,
                       model_exponents, term_names, useful_run_counts, z_matrix)
from .design import dsd, paley_conference, projection
from .exchange import ExchangeResult, coordinate_exchange, rank_two_update, updated_value

__version__ = "1.0.0"

__all__ = [
    "FollowUpProblem", "dsd", "paley_conference", "projection", "coordinate_exchange",
    "ExchangeResult", "rank_two_update", "updated_value", "trace_bound", "relaxation_bound",
    "cube_moments", "f_matrix", "z_matrix", "model_exponents", "term_names",
    "minimum_followup_runs", "useful_run_counts",
]
