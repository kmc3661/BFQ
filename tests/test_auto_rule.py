import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from bfq_auto_rule import predict


class AutoRuleTests(unittest.TestCase):
    def test_reference_statistics(self):
        cases = json.loads((ROOT / 'tests/rule_cases.json').read_text())
        for case in cases:
            with self.subTest(case=case['id']):
                result = predict(case['statistics'], case['quant_mode'])
                actual = [float(result[k]) for k in (
                    'BFQ_BUDGET_BASE_GRID', 'BFQ_N_GRID_TARGET_MEAN',
                    'BFQ_BUDGET_BONUS_GAMMA')]
                self.assertEqual(actual, case['expected'])

    def test_zero_effect(self):
        stats = dict(harmful_ratio=0, beneficial_ratio=0, activation_share=0,
                     support90_frac_total=0, top10_share_total=0,
                     top10_share_weight=0)
        for mode in ('w3a16', 'w4a8'):
            result = predict(stats, mode)
            self.assertEqual(result['BFQ_BUDGET_BASE_GRID'], '20')
            self.assertEqual(result['BFQ_N_GRID_TARGET_MEAN'], '20')


if __name__ == '__main__':
    unittest.main()
