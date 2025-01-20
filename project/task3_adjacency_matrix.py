from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from project.task2_regex import regex_to_dfa, graph_to_nfa
import scipy.sparse as sp
from collections.abc import Iterable
from pyformlang.finite_automaton import Symbol
from functools import reduce
from networkx import MultiDiGraph
import itertools


class AdjacencyMatrixFA:
    def __init__(self, automation: NondeterministicFiniteAutomaton = None):
        self.matricies = {}

        if automation is None:
            self.states = {}
            self.alphabet = set()
            self.start_states = set()
            self.final_states = set()
            return

        self.states = {st: i for (i, st) in enumerate(automation.states)}
        self.states_count = len(self.states)
        self.alphabet = automation.symbols

        graph = automation.to_networkx()

        self.matricies.update(
            {
                symbol: sp.csr_matrix(
                    (self.states_count, self.states_count), dtype=bool
                )
                for symbol in self.alphabet
            }
        )

        for u, v, label in graph.edges(data="label"):
            if not (str(u).startswith("starting_") or str(v).startswith("starting_")):
                self.matricies[label][self.states[u], self.states[v]] = True

        self.start_states = {self.states[key] for key in automation.start_states}
        self.final_states = {self.states[key] for key in automation.final_states}

    def accepts(self, word: Iterable[Symbol]) -> bool:
        symbols = list(word)

        configs = [(symbols, st) for st in self.start_states]

        while len(configs) > 0:
            rest, state = configs.pop()

            if len(rest) == 0 and state in self.final_states:
                return True

            for assume_next in self.states.values():
                if self.matricies[rest[0]][state, assume_next]:
                    configs.append((rest[1:], assume_next))

        return False

    def is_empty(self) -> bool:
        tr_clos = self.transitive_closure()

        for st, fn in itertools.product(self.start_states, self.final_states):
            if tr_clos[st, fn]:
                return False

        return True

    def transitive_closure(self):
        reach = sp.csr_matrix((self.states_count, self.states_count), dtype=bool)
        reach.setdiag(True)

        if not self.matricies:
            return reach

        reach: sp.csr_matrix = reach + reduce(
            lambda x, y: x + y, self.matricies.values()
        )

        for k in range(self.states_count):
            for i in range(self.states_count):
                for j in range(self.states_count):
                    reach[i, j] = reach[i, j] or (reach[i, k] and reach[k, j])

        return reach


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    A1, A2 = automaton1.matricies, automaton2.matricies

    intersect = AdjacencyMatrixFA()

    intersect.states_count = automaton1.states_count * automaton2.states_count

    for k in A1.keys():
        if A2.get(k) is None:
            continue
        intersect.matricies[k] = sp.kron(A1[k], A2[k], format="csr")

    intersect.states = {}
    for state1 in automaton1.states:
        for state2 in automaton2.states:
            intersect.states[(state1, state2)] = (
                automaton1.states[state1] * automaton2.states_count
                + automaton2.states[state2]
            )

    intersect.start_states = []
    for start1 in automaton1.start_states:
        for start2 in automaton2.start_states:
            intersect.start_states.append(start1 * automaton2.states_count + start2)

    intersect.final_states = []
    for final1 in automaton1.final_states:
        for final2 in automaton2.final_states:
            intersect.final_states.append(final1 * automaton2.states_count + final2)

    intersect.alphabet = automaton1.alphabet.union(automaton2.alphabet)

    return intersect


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    adj_matrix_by_reg = AdjacencyMatrixFA(regex_to_dfa(regex))
    adj_matrix_by_graph = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes)
    )

    intersect = intersect_automata(adj_matrix_by_reg, adj_matrix_by_graph)

    tr_cl = intersect.transitive_closure()

    reg_raw_start_states = []
    for key, value in adj_matrix_by_reg.states.items():
        if value in adj_matrix_by_reg.start_states:
            reg_raw_start_states.append(key)

    reg_raw_final_states = []
    for key, value in adj_matrix_by_reg.states.items():
        if value in adj_matrix_by_reg.final_states:
            reg_raw_final_states.append(key)

    result = set()
    for st in start_nodes:
        for fn in final_nodes:
            for st_reg in reg_raw_start_states:
                for fn_reg in reg_raw_final_states:
                    if tr_cl[
                        intersect.states[(st_reg, st)],
                        intersect.states[(fn_reg, fn)],
                    ]:
                        result.add((st, fn))
    return result
