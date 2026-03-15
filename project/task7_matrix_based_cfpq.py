from collections import defaultdict
from itertools import product
from typing import Set

import networkx as nx
import pyformlang.cfg
from pyformlang.cfg import Variable, Terminal
from scipy.sparse import csr_array

from project.task6_hellings_based_cfpq import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    node_to_idx: dict[int, int] = {int(n): idx for idx, n in enumerate(graph.nodes)}
    n = len(node_to_idx)

    start_nodes = start_nodes if start_nodes is not None else set(node_to_idx.keys())
    final_nodes = final_nodes if final_nodes is not None else set(node_to_idx.keys())

    cfg = cfg_to_weak_normal_form(cfg)

    terminal_to_vars: dict[Terminal, set[Variable]] = defaultdict(set)
    binary_bodies: dict[tuple[Variable, Variable], set[Variable]] = defaultdict(set)

    for prod in cfg.productions:
        body = prod.body
        if len(body) == 1 and isinstance(body[0], Terminal):
            terminal_to_vars[body[0]].add(prod.head)
        elif len(body) == 2:
            binary_bodies[(body[0], body[1])].add(prod.head)

    def empty_matrix() -> csr_array:
        return csr_array((n, n), dtype=bool)

    matrices: dict[Variable, csr_array] = defaultdict(empty_matrix)

    for v1, v2, lbl in graph.edges(data="label"):
        for var in terminal_to_vars.get(Terminal(lbl), set()):
            m = matrices[var].tolil()
            m[node_to_idx[int(v1)], node_to_idx[int(v2)]] = True
            matrices[var] = csr_array(m)

    for sym in cfg.get_nullable_symbols():
        var = sym if isinstance(sym, Variable) else Variable(sym.value)
        m = matrices[var].tolil()
        m.setdiag(True)
        matrices[var] = csr_array(m)

    updated: set[Variable] = set(cfg.variables)

    while updated:
        next_updated: set[Variable] = set()

        for (b1, b2), heads in binary_bodies.items():
            if b1 not in updated and b2 not in updated:
                continue

            delta: csr_array = matrices[b1] @ matrices[b2]
            if delta.nnz == 0:
                continue

            for head in heads:
                old_nnz = matrices[head].nnz
                matrices[head] = csr_array(matrices[head] + delta, dtype=bool)
                if matrices[head].nnz > old_nnz:
                    next_updated.add(head)

        updated = next_updated

    start_sym = cfg.start_symbol
    if start_sym not in matrices:
        return set()

    reach = matrices[start_sym]
    return {
        (s, f)
        for s, f in product(start_nodes, final_nodes)
        if reach[node_to_idx[s], node_to_idx[f]]
    }
