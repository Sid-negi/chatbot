import unittest
from chatbot import preprocess_text, get_answer

class TestChatbot(unittest.TestCase):

    def test_preprocess_text_lowercasing(self):
        """Test that preprocessing converts text to lowercase."""
        result = preprocess_text("HOW DO I RESET MY PASSWORD")
        self.assertEqual(result, "reset password")

    def test_preprocess_text_punctuation_removal(self):
        """Test that punctuation is removed during preprocessing."""
        result = preprocess_text("hello!!!")
        self.assertEqual(result, "hello")

    def test_preprocess_text_stopwords(self):
        """Test that common English stopwords are removed."""
        result = preprocess_text("How do I update my billing info")
        # 'how', 'do', 'i', 'my' should be removed
        self.assertNotIn("how", result.split())
        self.assertNotIn("i", result.split())
        self.assertIn("update", result.split())
        self.assertIn("billing", result.split())

    def test_get_answer_valid_match(self):
        """Test that a highly matching question returns the correct FAQ answer."""
        # "How do I reset my password?" is a defined question in chatbot.py
        answer = get_answer("How can I reset my password?", history=[])
        self.assertIn("To reset your password, go to the login page", answer)

    def test_get_answer_invalid_or_unrelated(self):
        """Test that an unrelated question returns the default fallback answer."""
        answer = get_answer("What is the meaning of life?", history=[])
        self.assertEqual(answer, "I don't have an answer for that. Please try rephrasing or contact support.")

    def test_get_answer_empty_input(self):
        """Test behavior when user types nothing."""
        answer = get_answer("", history=[])
        self.assertEqual(answer, "Please ask a valid question.")

if __name__ == "__main__":
    unittest.main()
