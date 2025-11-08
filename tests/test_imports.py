import unittest
import cv2
import numpy as np

class TestImports(unittest.TestCase):
    def test_imports(self):
        self.assertTrue(hasattr(cv2, 'imread'))
        self.assertTrue(hasattr(np, 'array'))

if __name__ == "__main__":
    unittest.main()
