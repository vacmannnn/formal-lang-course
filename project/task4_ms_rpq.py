import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import Symbol
from functools import reduce
from scipy.sparse import csr_matrix
from project.task2_regex import regex_to_dfa, graph_to_nfa
from project.task3_adjacency_matrix import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    adj_by_regex = AdjacencyMatrixFA(regex_to_dfa(regex))
    adj_matrix_nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    transposed_matricies: dict[Symbol, csr_matrix] = {}
    for symbol, matrix in adj_by_regex.matricies.items():
        transposed_matricies[symbol] = matrix.transpose()

    dfa_total_states = adj_by_regex.states_count
    dfa_initial_state = list(adj_by_regex.start_states)[0]
    nfa_initial_states = adj_matrix_nfa.start_states
    nfa_initial_count = len(nfa_initial_states)

    initial_data = np.ones(nfa_initial_count, dtype=bool)
    row_indices = [
        dfa_initial_state + dfa_total_states * i for i in range(nfa_initial_count)
    ]
    col_indices = [state for state in nfa_initial_states]
    front = csr_matrix(
        (initial_data, (row_indices, col_indices)),
        shape=(dfa_total_states * nfa_initial_count, adj_matrix_nfa.states_count),
        dtype=bool,
    )
    visited_states = front
    common_symbols = adj_by_regex.matricies.keys() & adj_matrix_nfa.matricies.keys()

    while front.count_nonzero() != 0:
        next_fronts = {}
        for symbol in common_symbols:
            next_fronts[symbol] = front @ adj_matrix_nfa.matricies[symbol]

            for idx in range(len(nfa_initial_states)):
                start_idx = idx * dfa_total_states
                end_idx = (idx + 1) * dfa_total_states

                next_fronts[symbol][start_idx:end_idx] = (
                    transposed_matricies[symbol]
                    @ next_fronts[symbol][start_idx:end_idx]
                )

        front = reduce(lambda x, y: x + y, next_fronts.values(), front)
        front = front > visited_states
        visited_states += front

    reversed_nfa_states = {v: k for k, v in adj_matrix_nfa.states.items()}
    res = set()

    for dfa_final_state in adj_by_regex.final_states:
        for idx, nfa_start in enumerate(nfa_initial_states):
            reachable_states = visited_states.getrow(
                dfa_total_states * idx + dfa_final_state
            ).indices
            for reached_state in reachable_states:
                if reached_state in adj_matrix_nfa.final_states:
                    res.add(
                        (
                            reversed_nfa_states[nfa_start],
                            reversed_nfa_states[reached_state],
                        )
                    )

    return res
