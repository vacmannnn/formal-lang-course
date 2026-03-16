from dataclasses import dataclass
from itertools import product
from typing import Optional

import networkx as nx
import pyformlang.cfg
import pyformlang.rsa
import scipy.sparse as sp
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.rsa import RecursiveAutomaton

from project.task2_regex import graph_to_nfa
from project.task3_adjacency_matrix import AdjacencyMatrixFA, intersect_automata


@dataclass(frozen=True)
class _RSMState:
    sym: Symbol
    state: State


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def _rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()
    for sym, box in rsm.boxes.items():
        dfa = box.dfa
        for st in dfa.start_states:
            nfa.add_start_state(State(_RSMState(sym, st)))
        for st in dfa.final_states:
            nfa.add_final_state(State(_RSMState(sym, st)))
        for st1, lbl, st2 in dfa._transition_function.get_edges():
            nfa.add_transition(
                State(_RSMState(sym, st1)), lbl, State(_RSMState(sym, st2))
            )
    return nfa


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    all_nodes = {int(n) for n in graph.nodes}
    start_nodes = start_nodes if start_nodes is not None else all_nodes
    final_nodes = final_nodes if final_nodes is not None else all_nodes

    graph_mfa = AdjacencyMatrixFA(graph_to_nfa(nx.MultiDiGraph(graph), set(), set()))
    rsm_nfa = _rsm_to_nfa(rsm)
    rsm_mfa = AdjacencyMatrixFA(rsm_nfa)

    rsm_start_states = rsm_nfa.start_states
    rsm_final_states = rsm_nfa.final_states
    prev_nnz = 0
    while True:
        inter = intersect_automata(graph_mfa, rsm_mfa)
        tc = inter.transitive_closure()

        if tc.nnz <= prev_nnz:
            break
        prev_nnz = tc.nnz

        idx_to_pair = {v: k for k, v in inter.states.items()}

        for i, j in zip(*tc.nonzero()):
            g1, rsm_st1 = idx_to_pair[i]
            g2, rsm_st2 = idx_to_pair[j]

            if rsm_st1 not in rsm_start_states or rsm_st2 not in rsm_final_states:
                continue

            rsm_state1 = rsm_st1.value
            rsm_state2 = rsm_st2.value

            if rsm_state1.sym != rsm_state2.sym:
                continue

            sym = rsm_state1.sym
            if sym not in graph_mfa.matricies:
                graph_mfa.matricies[sym] = sp.csr_matrix(
                    (graph_mfa.states_count, graph_mfa.states_count), dtype=bool
                )
            graph_mfa.matricies[sym][graph_mfa.states[g1], graph_mfa.states[g2]] = True

    initial_label = rsm.initial_label
    if initial_label not in graph_mfa.matricies:
        return set()

    reach = graph_mfa.matricies[initial_label]
    return {
        (s, f)
        for s, f in product(start_nodes, final_nodes)
        if s in graph_mfa.states
        and f in graph_mfa.states
        and reach[graph_mfa.states[s], graph_mfa.states[f]]
    }
