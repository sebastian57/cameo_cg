"""Energy models and topology utilities for CG protein simulations."""

from importlib import import_module

__all__ = [
    # Topology
    "TopologyBuilder",
    "precompute_chain_topology",
    "precompute_dihedrals",
    "precompute_repulsive_pairs",
    "filter_neighbors_by_mask",
    # Energy models
    "PriorEnergy",
    "AllegroModel",
    "AllegroModelCuEq",
    "MACEModel",
    "PaiNNModel",
    "CombinedModel",
]

_LAZY_SYMBOLS = {
    # Topology
    "TopologyBuilder": ("models.topology", "TopologyBuilder"),
    "precompute_chain_topology": ("models.topology", "precompute_chain_topology"),
    "precompute_dihedrals": ("models.topology", "precompute_dihedrals"),
    "precompute_repulsive_pairs": ("models.topology", "precompute_repulsive_pairs"),
    "filter_neighbors_by_mask": ("models.topology", "filter_neighbors_by_mask"),
    # Models
    "PriorEnergy": ("models.prior_energy", "PriorEnergy"),
    "AllegroModel": ("models.allegro_model", "AllegroModel"),
    "AllegroModelCuEq": ("models.allegro_cueq_model", "AllegroModelCuEq"),
    "MACEModel": ("models.mace_model", "MACEModel"),
    "PaiNNModel": ("models.painn_model", "PaiNNModel"),
    "CombinedModel": ("models.combined_model", "CombinedModel"),
}


def __getattr__(name):
    if name in _LAZY_SYMBOLS:
        module_name, symbol_name = _LAZY_SYMBOLS[name]
        module = import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'models' has no attribute '{name}'")
