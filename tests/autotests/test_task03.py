# This file contains test cases that you need to pass to get a grade
# You MUST NOT touch anything here except ONE block below
# You CAN modify this file IF AND ONLY IF you have found a bug and are willing to fix it
# Otherwise, please report it
from pyformlang.regular_expression import Regex
from pyformlang.cfg import CFG
import pytest
import random
import itertools
from grammars_constants import REGEXES
from rpq_concrete_cases import CASES_RPQ, CaseRPQ

# Fix import statements in try block to run tests
try:
    from project.task2_regex import regex_to_dfa
    from project.task3_adjacency_matrix import (
        intersect_automata,
        AdjacencyMatrixFA,
        tensor_based_rpq,
    )
except ImportError:
    pytestmark = pytest.mark.skip("Task 3 is not ready to test!")


class TestAdjacencyMatrixFAIntersection:
    @pytest.mark.parametrize(
        "regex_str1, regex_str2", itertools.combinations(REGEXES, 2)
    )
    def test(self, regex_str1: str, regex_str2: str) -> None:
        dfa1 = AdjacencyMatrixFA(regex_to_dfa(regex_str1))
        dfa2 = AdjacencyMatrixFA(regex_to_dfa(regex_str2))
        intersect_fa = intersect_automata(dfa1, dfa2)

        regex1: Regex = Regex(regex_str1)
        regex2: Regex = Regex(regex_str2)
        cfg_of_regex1: CFG = regex1.to_cfg()
        intersect_cfg: CFG = cfg_of_regex1.intersection(regex2)
        words = intersect_cfg.get_words()
        if intersect_cfg.is_finite():
            all_word_parts = list(words)
            if len(all_word_parts) == 0:
                assert intersect_fa.is_empty()
                return
            word_parts = random.choice(all_word_parts)
        else:
            index = random.randint(0, 2**9)
            word_parts = next(itertools.islice(words, index, None))

        word = map(lambda x: x.value, word_parts)

        assert intersect_fa.accepts(word)


class TestTensorBasedRPQ:
    @pytest.mark.parametrize("case", CASES_RPQ)
    def test_concrete_cases(self, case: CaseRPQ):
        case.check_answer_regex(tensor_based_rpq)
