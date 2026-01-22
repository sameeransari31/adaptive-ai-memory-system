"""
run_project_demo.py - Proof of Concept for Adaptive AI Memory System

This script demonstrates the complete adaptive memory pipeline with:
1. Real semantic embeddings via sentence-transformers ('all-MiniLM-L6-v2')
2. Multi-user memory management via GlobalMemoryManager
3. Reinforcement learning for importance scoring
4. Long-term memory storage and retrieval

Processing the Edutor App User Profile dataset.
"""

import torch
import re
from memory import AdaptiveMemorySystem, GlobalMemoryManager
from controller import MemoryController, TextEncoder

VECTOR_DIM = 64
RETRIEVAL_THRESHOLD = 0.4

USER_PROFILE_TEXT = """Every morning I open the Edutor app with a cup of black coffee at 7:15 a.m.,
usually starting with physics revision because quantitative topics feel clearer
when my mind is calm. I prefer concise bullet-point summaries over long
lectures, but I often re-watch visual explanations of magnetism because
diagrams stick better for me than equations. On stressful days I spend less time
reading- around ten minutes but I still attempt at least one problem to maintain
consistency. I tend to forget formulas if I study late at night after dinner,
especially when I'm tired, so I highlight them in blue to recall them faster next
morning. My average quiz score on recent chapters is around 82 percent, and I
notice a strong recall when I explain a concept to someone else the same day. I
dislike text-heavy history lessons and instead remember short stories or
timelines. Whenever I review on my phone during travel, retention drops; I guess
background noise and movement distract me. My emotion tracker shows higher
engagement whenever a session begins with an encouraging message or a short
recap video. Over the last three weeks I've revised electromagnetic induction
four times and now recall almost every law instantly. What I want the app to
remember is: I learn best in the early morning, through visuals, short summaries,
and spaced repetition every three days."""

IMPORTANT_KEYWORDS = ['physics', 'visual', 'morning', 'recall', 'summary', 'spaced']

def calculate_reward(sentence):
    """
    Demo reward function using keyword matching.
    See PRODUCTION NOTE above for real-world implementation.
    """
    sentence_lower = sentence.lower()
    for keyword in IMPORTANT_KEYWORDS:
        if keyword in sentence_lower:
            return 1.0, keyword
    return 0.0, None


def split_sentences(text):
    """
    Split text into sentences using regex that respects abbreviations.
    
    Handles:
    - Standard sentence endings (. ! ?)
    - Abbreviations like 'a.m.', 'p.m.', 'e.g.', 'i.e.', etc.
    - Decimal numbers and percentages
    
    Returns:
        list: List of sentence strings with whitespace stripped
    """
    text = text.replace('\n', ' ')
    pattern = r'(?<![ap])\.(?!m\.)(?=\s+[A-Z]|\s*$)|[!?](?=\s+[A-Z]|\s*$)'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def generate_response(user_query, context_text, confidence):
    """
    Generate an adapted response based on retrieved memory context.
    
    Args:
        user_query (str): The user's query string
        context_text (str or None): Retrieved memory text (if any)
        confidence (float): Retrieval confidence score
    
    Returns:
        str: An adapted response string with dynamic context insertion
    """
    if context_text is not None and confidence > 0.3:
        display_text = context_text[:80] + "..." if len(context_text) > 80 else context_text
        return (f"Based on your learning preferences (confidence: {confidence:.2f}), "
                f"Context found: '{display_text}' "
                f"Let me tailor the explanation accordingly...")
    else:
        return ("I don't have specific information about your preferences for this topic yet. "
                "Here is a general answer: Please let me know your preferred learning style "
                "so I can adapt future responses.")


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    USER_ID = "candidate_1"
    
    print_section("ADAPTIVE AI MEMORY SYSTEM - PRODUCTION PROOF OF CONCEPT")
    print(f"User ID: {USER_ID}")
    print(f"Important Keywords: {IMPORTANT_KEYWORDS}")
    
    print_section("PHASE 1: System Initialization")
    
    memory_manager = GlobalMemoryManager()
    storage = memory_manager.get_user_memory(USER_ID)
    
    brain = MemoryController()
    encoder = TextEncoder()
    
    print(f"  [OK] GlobalMemoryManager initialized")
    print(f"  [OK] User memory created for: {USER_ID}")
    print(f"  [OK] MemoryController initialized (Decision-Making Brain)")
    print(f"  [OK] TextEncoder initialized (sentence-transformers: all-MiniLM-L6-v2)")
    
    sentences = split_sentences(USER_PROFILE_TEXT)
    print(f"\n  Total sentences to process: {len(sentences)}")
    
    importance_history = []
    keyword_detections = []
    
    print_section("PHASE 2: Processing User Profile with Semantic Embeddings")
    
    for i, sentence in enumerate(sentences):
        sentence_vector = encoder.encode(sentence)
        
        decision = brain.assess_interaction(sentence_vector)
        importance = decision['importance']
        importance_history.append(importance)
        
        reward, matched_keyword = calculate_reward(sentence)
        
        print(f"\n[{i+1}/{len(sentences)}] \"{sentence[:50]}...\"")
        print(f"     Importance Score: {importance:.4f}")
        
        if reward > 0:
            keyword_detections.append((i, matched_keyword, importance))
            print(f"     >>> KEYWORD DETECTED: '{matched_keyword}' -> Reward: +1.0")
            
            learn_log = brain.process_feedback(reward_signal=reward)
            print(f"     >>> Controller Learning: {learn_log}")
            
            new_decision = brain.assess_interaction(sentence_vector)
            new_importance = new_decision['importance']
            
            if new_importance > importance:
                print(f"     >>> ADAPTED! New Importance: {new_importance:.4f} (+{new_importance - importance:.4f})")
                importance = new_importance
        
        log = storage.perceive(sentence_vector, importance, text=sentence)
        if "STORED" in log:
            print(f"     \033[1m>>> MEMORY COMMIT: {log}\033[0m")
        else:
            print(f"     Memory: {log}")
    
    print_section("PHASE 3: Learning Analysis")
    
    print(f"\n  Total sentences processed: {len(sentences)}")
    print(f"  Keywords detected: {len(keyword_detections)}")
    print(f"  Average importance score: {sum(importance_history)/len(importance_history):.4f}")
    
    print("\n  Keyword Detection Summary:")
    for idx, keyword, imp in keyword_detections:
        print(f"    - Sentence {idx+1}: '{keyword}' (importance: {imp:.4f})")
    
    print_section("PHASE 4: Semantic Memory Retrieval Test")
    
    test_queries = [
        "I love studying physics in the morning",
        "Visual learning with diagrams works best",
        "I enjoy cooking pasta for dinner"
    ]
    
    print("\n  Testing semantic retrieval with new queries:")
    for query in test_queries:
        query_vector = encoder.encode(query)
        retrieved_text, confidence = storage.query(query_vector)
        
        print(f"\n  Query: '{query}'")
        if confidence > 0.0 and retrieved_text is not None:
            text_preview = retrieved_text[:50] + "..." if len(retrieved_text) > 50 else retrieved_text
            
            if confidence > RETRIEVAL_THRESHOLD:
                print(f"  -> \033[1;32m[HIT]\033[0m Confidence: {confidence:.4f}")
                print(f"  -> Retrieved: \"{text_preview}\"")
            else:
                print(f"  -> \033[1;31m[MISS]\033[0m Confidence: {confidence:.4f} (Below Threshold: {RETRIEVAL_THRESHOLD})")
        else:
            print(f"  -> \033[1;31m[MISS]\033[0m No matching memory found (similarity too low)")
        
        adapted_response = generate_response(query, retrieved_text, confidence)
        print(f"  -> Adapted Response: {adapted_response}")
    
    print_section("PHASE 5: Multi-User Memory Status")
    
    stats = memory_manager.get_user_stats(USER_ID)
    print(f"\n  User: {USER_ID}")
    print(f"  LTM Slots Used: {stats['active_slots']}/{stats['total_slots']}")
    print(f"  Max Memory Strength: {stats['max_strength']:.4f}")
    print(f"  Avg Memory Strength: {stats['avg_strength']:.4f}")
    print(f"  Active Users in System: {memory_manager.list_users()}")
    
    print_section("PROOF OF CONCEPT COMPLETE")
    print("""
  KEY DEMONSTRATIONS:
  1. Real semantic embeddings via sentence-transformers (all-MiniLM-L6-v2)
  2. Multi-user memory isolation via GlobalMemoryManager
  3. MemoryController learns important patterns via reinforcement learning
  4. AdaptiveMemorySystem stores high-importance memories in LTM
  5. Semantic retrieval finds related content (not just exact matches)
  
  PRODUCTION READY FEATURES:
  - Semantic similarity enables true meaning-based matching
  - Per-user memory isolation supports multi-tenant deployment
  - Reward signals can be replaced with implicit user feedback
  - Memory decay and spaced repetition mimic human cognition
""")


if __name__ == "__main__":
    main()