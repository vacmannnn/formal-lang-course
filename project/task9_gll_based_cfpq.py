from dataclasses import dataclass
from typing import Optional

import networkx as nx
from pyformlang.finite_automaton import State, Symbol
from pyformlang.rsa import RecursiveAutomaton


@dataclass(frozen=True)
class _RSMState:
    sym: Symbol
    state: State


@dataclass(frozen=True)
class _GSSNode:
    rsm: _RSMState
    node: int


@dataclass(frozen=True)
class _Conf:
    rsm: _RSMState
    gss: _GSSNode
    node: int


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    all_nodes = {int(n) for n in graph.nodes}
    start_nodes = start_nodes if start_nodes is not None else all_nodes
    final_nodes = final_nodes if final_nodes is not None else all_nodes

    gss = nx.MultiDiGraph()
    paths: set[tuple[int, int]] = set()

    def process_term(conf: _Conf, rtransitions: dict) -> set[_Conf]:
        return {
            _Conf(
                rsm=_RSMState(sym=conf.rsm.sym, state=rtransitions[term]),
                gss=conf.gss,
                node=gnext,
            )
            for _, gnext, lbl in graph.out_edges(conf.node, data="label")
            if (term := Symbol(lbl)) in rtransitions
        }

    def process_nonterm(conf: _Conf, rtransitions: dict) -> set[_Conf]:
        result: set[_Conf] = set()

        for nonterm, rreturn_state in rtransitions.items():
            if nonterm not in rsm.labels:
                continue

            rnext = _RSMState(sym=nonterm, state=rsm.boxes[nonterm].dfa.start_state)
            gsnext = _GSSNode(rsm=rnext, node=conf.node)

            gss.add_node(gsnext)
            rreturn = _RSMState(sym=conf.rsm.sym, state=rreturn_state)
            gss.add_edge(gsnext, conf.gss, label=rreturn)

            if returns := gss.nodes[gsnext].get("returns"):
                result |= {
                    _Conf(rsm=rreturn, gss=conf.gss, node=greturn)
                    for greturn in returns
                }
                continue

            result.add(_Conf(rsm=rnext, gss=gsnext, node=conf.node))

        return result

    def process_return(conf: _Conf) -> set[_Conf]:
        if conf.rsm.state not in rsm.boxes[conf.rsm.sym].dfa.final_states:
            return set()

        gss.nodes[conf.gss].setdefault("returns", set()).add(conf.node)

        if conf.gss.rsm.sym == rsm.initial_label:
            paths.add((conf.gss.node, conf.node))

        return {
            _Conf(rsm=rreturn, gss=gsreturn, node=conf.node)
            for _, gsreturn, rreturn in gss.out_edges(conf.gss, data="label")
        }

    processed: set[_Conf] = set()
    pending: set[_Conf] = set()

    for start in start_nodes:
        rstate = _RSMState(
            sym=rsm.initial_label,
            state=rsm.boxes[rsm.initial_label].dfa.start_state,
        )
        gsnode = _GSSNode(rsm=rstate, node=start)
        gss.add_node(gsnode)
        pending.add(_Conf(rsm=rstate, gss=gsnode, node=start))

    while pending:
        conf = pending.pop()
        processed.add(conf)

        rtransitions = rsm.boxes[
            conf.rsm.sym
        ].dfa._transition_function._transitions.setdefault(conf.rsm.state, {})

        new = process_term(conf, rtransitions)
        new |= process_nonterm(conf, rtransitions)
        new |= process_return(conf)

        pending |= new - processed

    return {(v1, v2) for v1, v2 in paths if v1 in start_nodes and v2 in final_nodes}
