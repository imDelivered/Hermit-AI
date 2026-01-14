
import unittest
from unittest.mock import MagicMock, patch
from chatbot.joints.entity_extractor import EntityExtractorJoint

class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.joint = EntityExtractorJoint(model="mock-model")

    @patch('chatbot.joints.entity_extractor.local_inference')
    def test_extract_simple_query(self, mock_inference):
        # Mock LLM response
        mock_inference.return_value = '{"is_comparison": false, "entities": [{"name": "Python (programming language)", "type": "technology", "aliases": ["Python"]}], "action": "identify creator", "answer_type": "inventor"}'
        
        result = self.joint.extract("Who created Python?")
        
        self.assertEqual(len(result['entities']), 1)
        self.assertEqual(result['entities'][0]['name'], "Python (programming language)")
        self.assertFalse(result['is_comparison'])

    @patch('chatbot.joints.entity_extractor.local_inference')
    def test_extract_comparison(self, mock_inference):
        mock_inference.return_value = '{"is_comparison": true, "entities": [{"name": "Tesla"}, {"name": "Edison"}], "action": "compare", "comparison_dimension": "quantity"}'
        
        result = self.joint.extract("Tesla vs Edison")
        
        self.assertTrue(result['is_comparison'])
        self.assertEqual(len(result['entities']), 2)

if __name__ == '__main__':
    unittest.main()
