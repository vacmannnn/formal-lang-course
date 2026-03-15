import itertools

import numpy as np
from networkx import MultiDiGraph
from scipy.sparse import lil_array, vstack

from project.task2_regex import regex_to_dfa, graph_to_nfa
from project.task3_adjacency_matrix import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    all_nodes = {int(n) for n in graph.nodes}
    start_nodes = start_nodes if start_nodes else all_nodes
    final_nodes = final_nodes if final_nodes else all_nodes

    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex))

    starts = list(start_nodes)
    dfa_start = next(iter(dfa.start_states))

    front_st = []
    for st in starts:
        fr = lil_array((dfa.states_count, nfa.states_count), dtype=np.bool_)
        fr[dfa_start, nfa.states[st]] = True
        front_st.append(fr)

    front = vstack(front_st, format="csr")
    visited = front.copy()

    symbols = nfa.matricies.keys() & dfa.matricies.keys()
    dfa_transposed = {sym: dfa.matricies[sym].T.tocsr() for sym in symbols}

    while front.nnz:
        sym_fronts = []
        for sym in symbols:
            new_front = front @ nfa.matricies[sym]
            sym_fronts.append(
                vstack(
                    [
                        dfa_transposed[sym]
                        @ new_front[dfa.states_count * i : dfa.states_count * (i + 1)]
                        for i in range(len(starts))
                    ],
                    format="csr",
                )
            )

        if not sym_fronts:
            break

        front = sum(sym_fronts) > visited
        visited = visited + front

    result = set()
    for st_idx, st in enumerate(starts):
        visited_st = visited[
            dfa.states_count * st_idx : dfa.states_count * (st_idx + 1)
        ]
        for dfa_fi, fi in itertools.product(dfa.final_states, final_nodes):
            if fi in nfa.states and visited_st[dfa_fi, nfa.states[fi]]:
                result.add((st, fi))

    return result
