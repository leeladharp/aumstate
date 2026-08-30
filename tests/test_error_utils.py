import unittest

from error_utils import format_technical_error, format_user_error


class ErrorUtilsTests(unittest.TestCase):
    def test_format_user_error_preserves_explicit_message(self) -> None:
        error = ValueError("missing visual_prompt")
        self.assertEqual(format_user_error(error, "Unknown storyboard error"), "missing visual_prompt")

    def test_format_user_error_uses_exception_class_for_empty_message(self) -> None:
        error = Exception()
        self.assertEqual(format_user_error(error, "Unknown storyboard error"), "Unknown storyboard error (Exception)")

    def test_format_user_error_preserves_wrapped_cause(self) -> None:
        try:
            try:
                raise ValueError("missing visual_prompt")
            except ValueError as inner:
                raise RuntimeError("Storyboard validation failed") from inner
        except RuntimeError as error:
            message = format_user_error(error, "Unknown storyboard error")

        self.assertIn("Storyboard validation failed", message)
        self.assertIn("missing visual_prompt", message)

    def test_format_technical_error_includes_exception_chain(self) -> None:
        try:
            try:
                raise ValueError("bad json")
            except ValueError as inner:
                raise RuntimeError("Storyboard generation failed during JSON parsing") from inner
        except RuntimeError as error:
            details = format_technical_error(error)

        self.assertIn("RuntimeError: Storyboard generation failed during JSON parsing", details)
        self.assertIn("ValueError: bad json", details)


if __name__ == "__main__":
    unittest.main()
