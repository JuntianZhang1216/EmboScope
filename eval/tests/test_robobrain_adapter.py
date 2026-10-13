"""Serialization adaptation must never turn malformed prose into a choice."""
import unittest
from embodied_eval.answer_adapters import adapt_answer


class RoboBrainAdapterTests(unittest.TestCase):
    def test_final_channel_preserves_content(self):
        cases = {
            '<think></think><answer>A</answer>': 'A',
            '<think>Compare both observations.</think>\n<answer>Yes</answer>': 'Yes',
            '<answer>Pick up the cup, then place it down.</answer>': 'Pick up the cup, then place it down.',
            '<answer>B because the robot moves.</answer>': 'B because the robot moves.',
            '<answer></answer>': '',
        }
        for raw, final in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(adapt_answer(raw, 'robobrain_answer_v1'), final)
                self.assertEqual(adapt_answer(raw, 'hy_answer_v1'), final)

    def test_malformed_and_unwrapped_remain_unchanged(self):
        for raw in ['A', 'The answer is A.', '<answer>A',
                    '<answer>A</answer> explanation',
                    'prefix <answer>A</answer>',
                    '<answer>A</answer><answer>B</answer>',
                    '<answer><answer>A</answer></answer>',
                    '<think>unfinished<answer>A</answer>',
                    '<think>x</think><think>y</think><answer>A</answer>',
                    '<answer><think>x</think>A</answer>']:
            with self.subTest(raw=raw):
                self.assertEqual(adapt_answer(raw, 'robobrain_answer_v1'), raw)
                self.assertEqual(adapt_answer(raw, 'hy_answer_v1'), raw)

    def test_model_answer_envelopes(self):
        raw = '<think></think><answer>A</answer>'
        self.assertEqual(adapt_answer(raw), raw)
        self.assertEqual(adapt_answer(raw, 'glm_box_v1'), raw)
        self.assertEqual(adapt_answer('<|begin_of_box|>A<|end_of_box|>', 'glm_box_v1'), 'A')

    def test_mimo_reasoning_boundary_and_truncation(self):
        self.assertEqual(adapt_answer('<think>Compare images.</think>\nA', 'mimo_think_v1'), 'A')
        self.assertEqual(adapt_answer('<think>Unfinished reasoning A', 'mimo_think_v1'), '')
        self.assertEqual(adapt_answer('<think>Done.</think>', 'mimo_think_v1'), '')
        self.assertEqual(adapt_answer('<think>Done.</think>B because...', 'mimo_think_v1'), 'B because...')
        for raw in ['A', 'Answer: B', 'prefix <think>x</think>A', '<think>x</think><think>y</think>A', '<think>x</think>A</think>']:
            self.assertEqual(adapt_answer(raw, 'mimo_think_v1'), raw)
