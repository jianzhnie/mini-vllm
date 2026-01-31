import unittest

from minivllm.llm import LLM


class TestLLM(unittest.TestCase):

    def test_llm_imports(self):
        self.assertTrue(issubclass(LLM, object))


if __name__ == '__main__':
    unittest.main()
