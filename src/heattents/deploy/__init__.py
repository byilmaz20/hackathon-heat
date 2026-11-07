"""Daily decision: where to put today's tents and which doctors staff them in which shift (Gurobi MILP)."""
from .milp import solve, synthetic_request
from .model import DeploymentPlan, DeploymentRequest, Doctor, Shift, Site

__all__ = ["solve", "synthetic_request", "DeploymentPlan", "DeploymentRequest", "Doctor", "Shift", "Site"]
