"""
Unit tests for evaluation metrics.
"""
import unittest
from workspace.evaluation.metrics import (
    normalize_numberish,
    exact_match_norm,
    token_f1,
    rouge_l,
    answer_in_evidence,
    recall_at_k,
    mrr_at_k
)


class TestNormalization(unittest.TestCase):
    """Test text normalization functions"""
    
    def test_normalize_numberish(self):
        """Test number normalization"""
        self.assertEqual(normalize_numberish("$1,080"), "1080")
        self.assertEqual(normalize_numberish("12.8 percent"), "12.8%")
        self.assertEqual(normalize_numberish("USD 1000"), "1000")
        
    def test_normalize_case_insensitive(self):
        """Test case insensitivity"""
        self.assertEqual(normalize_numberish("Apple Inc"), normalize_numberish("apple inc"))


class TestAnswerMetrics(unittest.TestCase):
    """Test answer quality metrics"""
    
    def test_exact_match_norm(self):
        """Test exact match scoring"""
        self.assertEqual(exact_match_norm("925 million", "$925 million"), 1.0)
        self.assertEqual(exact_match_norm("apple", "orange"), 0.0)
        
    def test_exact_match_numberish(self):
        """Test number-aware exact match"""
        self.assertEqual(exact_match_norm("1,080", "1080"), 1.0)
        self.assertEqual(exact_match_norm("12.8 percent", "12.8%"), 1.0)
        
    def test_token_f1_perfect(self):
        """Test perfect token F1"""
        self.assertEqual(token_f1("hello world", "hello world"), 1.0)
        
    def test_token_f1_partial(self):
        """Test partial token F1"""
        score = token_f1("hello world", "hello there")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        
    def test_token_f1_empty(self):
        """Test empty string handling"""
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("hello", ""), 0.0)
        
    def test_rouge_l(self):
        """Test ROUGE-L score"""
        self.assertEqual(rouge_l("the cat sat", "the cat sat"), 1.0)
        score = rouge_l("the cat sat on mat", "the cat sat")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class TestGroundingMetrics(unittest.TestCase):
    """Test answer grounding metrics"""
    
    def test_answer_in_evidence_present(self):
        """Test when answer appears in evidence"""
        pred = "925 million"
        evidence = "The company had $925 million in cash reserves."
        self.assertEqual(answer_in_evidence(pred, evidence), 1.0)
        
    def test_answer_in_evidence_absent(self):
        """Test when answer not in evidence"""
        pred = "100 million"
        evidence = "The company had 50 million in cash."
        self.assertEqual(answer_in_evidence(pred, evidence), 0.0)


class TestRetrievalMetrics(unittest.TestCase):
    """Test retrieval quality metrics"""
    
    def test_recall_at_k_found(self):
        """Test recall when gold doc found"""
        hits = [
            {"doc_id": "doc1"},
            {"doc_id": "doc2"},
            {"doc_id": "doc3"}
        ]
        self.assertEqual(recall_at_k(hits, "doc2", k=5), 1.0)
        
    def test_recall_at_k_not_found(self):
        """Test recall when gold doc not found"""
        hits = [{"doc_id": "doc1"}, {"doc_id": "doc2"}]
        self.assertEqual(recall_at_k(hits, "doc3", k=5), 0.0)
        
    def test_recall_at_k_beyond_k(self):
        """Test recall when gold doc beyond k"""
        hits = [
            {"doc_id": f"doc{i}"} for i in range(10)
        ]
        self.assertEqual(recall_at_k(hits, "doc8", k=5), 0.0)
        
    def test_mrr_at_k_rank_1(self):
        """Test MRR with gold at rank 1"""
        hits = [
            {"doc_id": "gold"},
            {"doc_id": "doc2"}
        ]
        self.assertEqual(mrr_at_k(hits, "gold", k=5), 1.0)
        
    def test_mrr_at_k_rank_3(self):
        """Test MRR with gold at rank 3"""
        hits = [
            {"doc_id": "doc1"},
            {"doc_id": "doc2"},
            {"doc_id": "gold"}
        ]
        self.assertAlmostEqual(mrr_at_k(hits, "gold", k=5), 1.0/3.0)


if __name__ == "__main__":
    unittest.main()
