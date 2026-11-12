"""Fault injection using fake wrappers; no model, simulator or cloud."""
import unittest
from types import SimpleNamespace

from failure_guards import (
    CandidateGuardError, InteractivePlacementFailure, ResetBudget,
    reject_interactive_placement, reset_once_zero_noise,
)


class FailureGuards(unittest.TestCase):
    def call(self, lower, budget=None, noise=0, check=lambda: None):
        events = []
        budget = budget or ResetBudget()
        result = reset_once_zero_noise(
            SimpleNamespace(noise=noise, env=SimpleNamespace(reset=lower)),
            budget, role="synthetic", emit=lambda *x: events.append(x), check=check,
        )
        return result, events, budget

    def test_success_returns_original_observation_without_second_call(self):
        result, calls = object(), []
        out, events, budget = self.call(lambda: calls.append(1) or result)
        self.assertIs(out, result)
        self.assertEqual(calls, [1])
        self.assertEqual((budget.attempted, budget.successful), (1, 1))
        self.assertEqual([x[0] for x in events], ["RESET_ATTEMPT", "RESET_SUCCEEDED"])

    def test_all_failure_classes_propagate_with_one_attempt(self):
        for error in (RuntimeError("randomization"), KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(kind=type(error).__name__):
                calls, events = [], []
                budget = ResetBudget()
                def lower():
                    calls.append(1)
                    raise error
                with self.assertRaises(type(error)):
                    reset_once_zero_noise(
                        SimpleNamespace(noise=0, env=SimpleNamespace(reset=lower)),
                        budget, role="synthetic", emit=lambda *x: events.append(x),
                        check=lambda: None,
                    )
                self.assertEqual(calls, [1])
                self.assertEqual((budget.attempted, budget.successful), (1, 0))
                self.assertEqual(events[-1][0], "RESET_FAILED")

    def test_budget_stops_before_an_eleventh_attempt(self):
        calls = []
        budget = ResetBudget()
        for _ in range(10):
            self.call(lambda: calls.append(1), budget)
        with self.assertRaisesRegex(CandidateGuardError, "reset_attempt_cap"):
            self.call(lambda: calls.append(1), budget)
        self.assertEqual(len(calls), 10)

    def test_unsupported_noise_stops_before_reset(self):
        for noise in (1, -1, 0.0, False):
            calls = []
            with self.assertRaises(CandidateGuardError):
                self.call(lambda: calls.append(1), noise=noise)
            self.assertEqual(calls, [])

    def test_expired_budget_stops_before_reset(self):
        calls = []
        def expired():
            raise TimeoutError()
        with self.assertRaises(TimeoutError):
            self.call(lambda: calls.append(1), check=expired)
        self.assertEqual(calls, [])

    def test_placement_debugger_fails_without_interaction(self):
        with self.assertRaises(InteractivePlacementFailure):
            reject_interactive_placement()


if __name__ == "__main__":
    unittest.main()
