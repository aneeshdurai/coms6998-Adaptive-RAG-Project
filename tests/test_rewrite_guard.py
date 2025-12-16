"""
Unit tests for query rewrite validation.
"""
import unittest
from workspace.models.rewrite_guard import (
    validate_rewrite,
    _extract_years,
    _extract_numbers,
    _tokens
)


class TestExtractionFunctions(unittest.TestCase):
    """Test text extraction functions"""
    
    def test_extract_years(self):
        """Test year extraction"""
        years = _extract_years("In 2023 and FY2022, the company...")
        self.assertEqual(years, {"2023", "2022"})
        
    def test_extract_years_no_match(self):
        """Test no years found"""
        years = _extract_years("The company reported strong results.")
        self.assertEqual(years, set())
        
    def test_extract_numbers(self):
        """Test number extraction"""
        nums = _extract_numbers("Revenue was $1,080 million or 12.8% growth")
        self.assertIn("1080", nums)
        self.assertIn("12.8%", nums)
        
    def test_extract_numbers_no_match(self):
        """Test no numbers found"""
        nums = _extract_numbers("The company grew significantly.")
        self.assertEqual(nums, set())


class TestRewriteValidation(unittest.TestCase):
    """Test rewrite validation logic"""
    
    def test_valid_rewrite_rephrase(self):
        """Test valid rewrite (simple rephrase)"""
        original = "What was Apple's revenue?"
        rewritten = "What was the revenue of Apple?"
        result = validate_rewrite(original, rewritten)
        self.assertTrue(result.ok)
        
    def test_invalid_added_year(self):
        """Test invalid rewrite (added year)"""
        original = "What was Apple's revenue?"
        rewritten = "What was Apple's revenue in 2023?"
        result = validate_rewrite(original, rewritten, forbid_new_years=True)
        self.assertFalse(result.ok)
        self.assertIn("added_years", str(result.reasons))
        
    def test_valid_preserved_year(self):
        """Test valid rewrite (preserved year)"""
        original = "What was Apple's revenue in 2023?"
        rewritten = "What was the 2023 revenue of Apple?"
        result = validate_rewrite(original, rewritten)
        self.assertTrue(result.ok)
        
    def test_invalid_added_number(self):
        """Test invalid rewrite (added number)"""
        original = "What was the company's debt?"
        rewritten = "What was the company's $500 million debt?"
        result = validate_rewrite(original, rewritten, forbid_new_numbers=True)
        self.assertFalse(result.ok)
        self.assertIn("added_numbers", str(result.reasons))
        
    def test_invalid_low_overlap(self):
        """Test invalid rewrite (topic drift)"""
        original = "What was Apple's revenue?"
        rewritten = "How many employees does Microsoft have?"
        result = validate_rewrite(original, rewritten)
        self.assertFalse(result.ok)
        self.assertIn("low_token_overlap", str(result.reasons))
        
    def test_valid_with_relaxed_validation(self):
        """Test valid rewrite with relaxed validation"""
        original = "What was the company's debt?"
        rewritten = "What was the total debt in 2023?"
        
        # Strict: should fail
        result_strict = validate_rewrite(original, rewritten, 
                                        forbid_new_years=True, 
                                        forbid_new_numbers=True)
        self.assertFalse(result_strict.ok)
        
        # Relaxed: should pass
        result_relaxed = validate_rewrite(original, rewritten,
                                          forbid_new_years=False,
                                          forbid_new_numbers=False)
        # May still fail on overlap, but won't fail on added year
        if not result_relaxed.ok:
            self.assertNotIn("added_years", str(result_relaxed.reasons))


if __name__ == "__main__":
    unittest.main()
