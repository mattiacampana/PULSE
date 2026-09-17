from typing import Any

from .oppavg import OppAVG
from .local import LocalOnlyNode
from .opportunistic_fl import OpportunisticFLNode
from .optimist import OPTIMISTNode
from .pulse import PULSENode

from opportunistic_simulator.utils.config import Config
from .base import Node

def create_node(
    config: Config,
    node_kwargs: dict[str, Any],
    *,
    node_index: int,
    num_nodes: int,
    seed: int
) -> Node:
    """Instantiate the node selected by ``algorithm``.

    Parameters
    ----------
    algorithm:
        Typed algorithm section of the experiment configuration.
    node_kwargs:
        Arguments required by every node implementation, such as ``node_id``,
        local data, encoder, classifier, device, and training configuration.
    node_index, num_nodes:
        Position and total number of nodes.  They are only used by OPTIMIST to
        assign a subnetwork deterministically.
    """
    if config.algorithm.name == "local":
        return LocalOnlyNode(**node_kwargs, seed=seed)
    
    if config.algorithm.name == "oppavg":
        return OppAVG(**node_kwargs, seed=seed)
    
    if config.algorithm.name == "oppfl":
        return OpportunisticFLNode(
            **node_kwargs,
            similarity_threshold=config.opportunistic_fl.similarity_threshold,
            lambda_weight=config.opportunistic_fl.lambda_weight,
            encounter_rounds=config.opportunistic_fl.encounter_rounds,
            aggregation=config.opportunistic_fl.aggregation,
            seed=seed
        )
    
    if config.algorithm.name == "optimist":
        return OPTIMISTNode(
            **node_kwargs,
            num_subnetworks=num_nodes,
            subnet_id=node_index % num_nodes,
            seed=seed
        )

    if config.algorithm.name == "pulse":
        return PULSENode(
            **node_kwargs,
            alpha_max=config.pulse.alpha_max,
            utility_momentum=config.pulse.utility_momentum,
            initial_utility=config.pulse.initial_utility,
            ucb_exploration=config.pulse.ucb_exploration,
            norm_epsilon=config.pulse.norm_epsilon,
            seed=seed
        )
    
    raise ValueError(
        "Unknown algorithm. Available: local, oppavg, "
        "oppfl, optimist, pulse."
    )


__all__ = [
    "OppAVG",
    "LocalOnlyNode",
    "OpportunisticFLNode",
    "OPTIMISTNode",
    "PULSENode",
    "create_node",
]
