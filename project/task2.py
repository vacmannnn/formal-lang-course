from typing import Set

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    is_start_empty = len(start_states) == 0
    is_final_empty = len(final_states) == 0
    if is_final_empty or is_start_empty:
        for node in graph.nodes:
            if is_start_empty:
                start_states.add(node)
            if is_final_empty:
                final_states.add(node)

    nfa = NondeterministicFiniteAutomaton.from_networkx(graph=graph)
    for st in start_states:
        nfa.add_start_state(st)
    for st in final_states:
        nfa.add_final_state(st)

    return nfa
