#!/usr/bin/env python3
"""
Test answer extraction logic
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "evaluation", "vlnce_baselines"))

from mindcraft_metrics import extract_answer_from_response, check_answer_correctness

# Test cases
test_cases = [
    # Case 1: Letter answer for multiple choice
    {
        "response": "A",
        "options": ["before", "after", "at the same time"],
        "ground_truth": "before",
        "should_be_correct": True,
        "description": "Simple letter A for 'before'"
    },
    {
        "response": "B",
        "options": ["before", "after", "at the same time"],
        "ground_truth": "after",
        "should_be_correct": True,
        "description": "Simple letter B for 'after'"
    },
    {
        "response": "C",
        "options": ["before", "after", "at the same time"],
        "ground_truth": "at the same time",
        "should_be_correct": True,
        "description": "Simple letter C for 'at the same time'"
    },
    
    # Case 2: Letter with punctuation
    {
        "response": "A)",
        "options": ["bedroom", "kitchen", "bathroom"],
        "ground_truth": "bedroom",
        "should_be_correct": True,
        "description": "Letter with parenthesis A) for 'bedroom'"
    },
    {
        "response": "(B)",
        "options": ["bedroom", "kitchen", "bathroom"],
        "ground_truth": "kitchen",
        "should_be_correct": True,
        "description": "Letter with parentheses (B) for 'kitchen'"
    },
    
    # Case 3: Full word answers
    {
        "response": "before",
        "options": ["before", "after", "at the same time"],
        "ground_truth": "before",
        "should_be_correct": True,
        "description": "Full word 'before'"
    },
    {
        "response": "The answer is before",
        "options": ["before", "after", "at the same time"],
        "ground_truth": "before",
        "should_be_correct": True,
        "description": "Sentence containing 'before'"
    },
    
    # Case 4: Left/Right questions
    {
        "response": "left",
        "options": ["left", "right"],
        "ground_truth": "left",
        "should_be_correct": True,
        "description": "Simple 'left'"
    },
    {
        "response": "It was on the left side",
        "options": ["left", "right"],
        "ground_truth": "left",
        "should_be_correct": True,
        "description": "Sentence with 'left'"
    },
    
    # Case 5: Yes/No questions
    {
        "response": "yes",
        "options": ["yes", "no"],
        "ground_truth": "yes",
        "should_be_correct": True,
        "description": "Simple 'yes'"
    },
    {
        "response": "No, it is not",
        "options": ["yes", "no"],
        "ground_truth": "no",
        "should_be_correct": True,
        "description": "Sentence with 'no'"
    },
    
    # Case 6: Room names
    {
        "response": "bedroom",
        "options": ["bedroom", "kitchen", "bathroom"],
        "ground_truth": "bedroom",
        "should_be_correct": True,
        "description": "Room name 'bedroom'"
    },
    {
        "response": "I think it's the kitchen",
        "options": ["bedroom", "kitchen", "bathroom"],
        "ground_truth": "kitchen",
        "should_be_correct": True,
        "description": "Sentence containing 'kitchen'"
    },
]

print("="*80)
print("Testing Answer Extraction Logic")
print("="*80)

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    response = test["response"]
    options = test["options"]
    ground_truth = test["ground_truth"]
    should_be_correct = test["should_be_correct"]
    description = test["description"]
    
    # Extract answer
    extracted = extract_answer_from_response(response, options, ground_truth)
    
    # Check correctness
    is_correct = check_answer_correctness(response, ground_truth, options)
    
    # Determine if test passed
    test_passed = (is_correct == should_be_correct)
    
    if test_passed:
        status = "✓ PASS"
        passed += 1
    else:
        status = "✗ FAIL"
        failed += 1
    
    print(f"\nTest {i}: {status}")
    print(f"  Description: {description}")
    print(f"  Response: '{response}'")
    print(f"  Options: {options}")
    print(f"  Ground Truth: '{ground_truth}'")
    print(f"  Extracted: '{extracted}'")
    print(f"  Is Correct: {is_correct} (expected: {should_be_correct})")

print("\n" + "="*80)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("="*80)

if __name__ == "__main__":
    if failed > 0:
        sys.exit(1)
    print("\n✓ All tests passed!")
    sys.exit(0)


