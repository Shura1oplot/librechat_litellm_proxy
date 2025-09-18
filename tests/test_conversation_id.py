import unittest

from custom_handler import OpenAIResponsesBridge


class ConversationIdTests(unittest.TestCase):
    def test_get_conversation_id_basic(self) -> None:
        messages = [
            {"role": "user", "content": "<conv_id=abc123>\nHello"},
        ]

        self.assertEqual(
            OpenAIResponsesBridge._get_conversation_id(messages),
            "abc123",
        )

    def test_get_conversation_id_with_inline_gt_characters(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "<conv_id=abc123> I'm curious why 5 > 3 holds.",
            },
        ]

        self.assertEqual(
            OpenAIResponsesBridge._get_conversation_id(messages),
            "abc123",
        )


if __name__ == "__main__":
    unittest.main()
