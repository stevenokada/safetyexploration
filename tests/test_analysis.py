import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze import exact_mcnemar, normalize, wilson
from run_experiment import trigger_matches


class TestScoring(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("A practical note."), "a practical note")

    def test_triggers(self):
        self.assertTrue(trigger_matches("weekday", "Try it on Monday."))
        self.assertFalse(trigger_matches("weekday", "Try it next month."))
        self.assertTrue(trigger_matches("quoted_span", 'Explain "least privilege".'))
        self.assertFalse(trigger_matches("quoted_span", "Use `least privilege`."))
        self.assertTrue(trigger_matches("comparison", "Compare A and B"))
        self.assertFalse(trigger_matches("comparison", "Discuss A and B"))
        self.assertTrue(trigger_matches("recommendation_place", "Recommend a cafe in Paris"))
        self.assertFalse(trigger_matches("recommendation_place", "Describe Paris"))

    def test_wilson(self):
        lo, hi = wilson(0, 4)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.4)
        self.assertLess(hi, 0.6)

    def test_mcnemar(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar(1, 0), 1.0)


if __name__ == "__main__":
    unittest.main()

