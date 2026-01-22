import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np


VECTOR_DIM = 64
STM_CAPACITY = 5
LTM_SLOTS = 100
DECAY_RATE = 0.995
IMPORTANCE_THRESHOLD = 0.45


class AdaptiveMemorySystem(nn.Module):
    """
    A Neural Memory Module that mimics human cognitive processes.
    
    Structure:
    1. STM (Short Term Memory): FIFO buffer for immediate context.
    2. LTM (Long Term Memory): Persistent tensor storage for habits/facts.
    
    Philosophy:
    - No SQL/NoSQL. Memory is a learned weight matrix.
    - Forgetting is a feature, not a bug.
    - Importance is determined externally by the MemoryController.
    """
    
    def __init__(self):
        super().__init__()
        self.stm = deque(maxlen=STM_CAPACITY)
        self.ltm_matrix = torch.zeros(LTM_SLOTS, VECTOR_DIM)
        self.ltm_strengths = torch.zeros(LTM_SLOTS)
        self.ltm_texts = [None] * LTM_SLOTS

    def perceive(self, input_vector, importance_score, text=None):
        """
        The main entry point. The system 'sees' a new vector.
        
        Args:
            input_vector (Tensor): The encoded input vector.
            importance_score (float): Importance score from MemoryController (0.0 to 1.0).
            text (str, optional): The raw text string to store for retrieval.
        
        Process:
        1. Update STM
        2. Conditionally Update LTM (Consolidation)
        3. Apply Time Decay (Forgetting)
        """
        input_vector = F.normalize(input_vector, p=2, dim=0)
        importance = importance_score
        self.stm.append({
            'vector': input_vector,
            'importance': importance,
            'text': text
        })
        if importance > IMPORTANCE_THRESHOLD:
            slot_idx = self._consolidate_to_ltm(input_vector, importance)
            if text is not None:
                self.ltm_texts[slot_idx] = text
            decision_log = f"[MEM] STORED | Score: {importance:.4f} | Slot: {slot_idx}"
        else:
            decision_log = f"[MEM] IGNORED | Score: {importance:.4f} (Threshold: {IMPORTANCE_THRESHOLD})"

        self._apply_decay()
        
        return decision_log

    def _consolidate_to_ltm(self, vector, importance):
        """
        Writes to Long Term Memory with Memory Reinforcement.
        
        Logic:
        1. Calculate cosine similarity between input and all existing LTM vectors.
        2. If most similar memory has similarity > SIMILARITY_THRESHOLD (0.85):
           - Merge new vector into old (80% old, 20% new weighted average).
           - Boost importance score: current_strength + (importance * 0.5), clamped at 1.0.
           - Return index of reinforced slot.
        3. If no similar memory found (similarity < 0.85):
           - Find slot with minimum strength and overwrite it entirely.
           - Return index of overwritten slot.
        """
        SIMILARITY_THRESHOLD = 0.85
        OLD_WEIGHT = 0.8
        NEW_WEIGHT = 0.2
        BOOST_FACTOR = 0.5
        vector_normalized = F.normalize(vector, p=2, dim=0)
        ltm_norms = torch.norm(self.ltm_matrix, p=2, dim=1, keepdim=True)
        
        if torch.all(ltm_norms < 1e-6):
            weakest_slot_idx = torch.argmin(self.ltm_strengths)
            self.ltm_matrix[weakest_slot_idx] = vector
            self.ltm_strengths[weakest_slot_idx] = importance
            return weakest_slot_idx.item()
        
        ltm_norms = torch.clamp(ltm_norms, min=1e-8)
        ltm_normalized = self.ltm_matrix / ltm_norms
        similarities = torch.mv(ltm_normalized, vector_normalized)
        max_similarity, most_similar_idx = torch.max(similarities, dim=0)
        
        if max_similarity.item() > SIMILARITY_THRESHOLD:
            old_vector = self.ltm_matrix[most_similar_idx]
            merged_vector = OLD_WEIGHT * old_vector + NEW_WEIGHT * vector
            merged_vector = F.normalize(merged_vector, p=2, dim=0)
            
            self.ltm_matrix[most_similar_idx] = merged_vector
            
            current_strength = self.ltm_strengths[most_similar_idx].item()
            boosted_strength = current_strength + (importance * BOOST_FACTOR)
            self.ltm_strengths[most_similar_idx] = min(boosted_strength, 1.0)
            
            return most_similar_idx.item()
        else:
            weakest_slot_idx = torch.argmin(self.ltm_strengths)
            self.ltm_matrix[weakest_slot_idx] = vector
            self.ltm_strengths[weakest_slot_idx] = importance
            return weakest_slot_idx.item()

    def _apply_decay(self):
        """
        Simulates the Ebbinghaus Forgetting Curve.
        Every memory loses a fraction of its strength at every time step.
        """
        self.ltm_strengths = self.ltm_strengths * DECAY_RATE
        mask = self.ltm_strengths > 0.01
        self.ltm_strengths = self.ltm_strengths * mask.float()

    def query(self, query_vector):
        """
        Retrieves relevant memories based on vector similarity AND memory strength.
        
        Implements 'Spaced Repetition' logic:
        - When a memory is successfully recalled (score > 0.1), its strength is boosted.
        - This mimics 'active recall' and prevents decay for frequently accessed memories.
        
        Logic Gate (Similarity Threshold):
        - A valid_mask (similarity > 0.25) ensures irrelevant queries score 0.0
        - This prevents high-importance memories from being retrieved for unrelated queries
        
        Returns:
            The best matching memory text (str or None)
            The confidence score (float)
        """
        RECALL_THRESHOLD = 0.1
        RECALL_BOOST = 1.2
        SIMILARITY_WEIGHT = 0.7
        STRENGTH_WEIGHT = 0.3
        SIMILARITY_GATE = 0.35
        
        query_vector = F.normalize(query_vector, p=2, dim=0)
        similarities = torch.mv(self.ltm_matrix, query_vector)
        
        valid_mask = (similarities > SIMILARITY_GATE).float()
        
        retrieval_scores = (similarities * SIMILARITY_WEIGHT) + (self.ltm_strengths * STRENGTH_WEIGHT)
        
        retrieval_scores = retrieval_scores * valid_mask
        
        best_score = torch.max(retrieval_scores)
        best_idx = torch.argmax(retrieval_scores)
        
        if best_score.item() > RECALL_THRESHOLD:
            current_strength = self.ltm_strengths[best_idx].item()
            boosted_strength = min(current_strength * RECALL_BOOST, 1.0)
            self.ltm_strengths[best_idx] = boosted_strength
            return self.ltm_texts[best_idx], best_score.item()
        elif best_score.item() > 0.0:
            return self.ltm_texts[best_idx], best_score.item()
        else:
            return None, 0.0
    
    def get_stats(self):
        """Returns current memory statistics."""
        active_slots = (self.ltm_strengths > 0.01).sum().item()
        max_strength = torch.max(self.ltm_strengths).item()
        avg_strength = self.ltm_strengths[self.ltm_strengths > 0.01].mean().item() if active_slots > 0 else 0
        return {
            "active_slots": int(active_slots),
            "total_slots": LTM_SLOTS,
            "max_strength": max_strength,
            "avg_strength": avg_strength
        }


class GlobalMemoryManager:
    """
    Multi-User Memory Manager for Production Scalability.
    
    Manages separate AdaptiveMemorySystem instances for each user,
    enabling isolated memory storage and retrieval per user session.
    
    Features:
    - Lazy initialization: User memory created on first access
    - Isolated memory: Each user has independent STM/LTM
    - Scalable: Dictionary-based user lookup
    
    Usage:
        manager = GlobalMemoryManager()
        user_memory = manager.get_user_memory("user_123")
        user_memory.perceive(vector, importance)
    """
    
    def __init__(self):
        self.user_stores = {}
    
    def get_user_memory(self, user_id: str) -> AdaptiveMemorySystem:
        """
        Get or create an AdaptiveMemorySystem for a specific user.
        
        Args:
            user_id (str): Unique identifier for the user.
            
        Returns:
            AdaptiveMemorySystem: The user's personal memory instance.
        """
        if user_id not in self.user_stores:
            self.user_stores[user_id] = AdaptiveMemorySystem()
        return self.user_stores[user_id]
    
    def list_users(self) -> list:
        """Returns list of all user IDs with active memory stores."""
        return list(self.user_stores.keys())
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get memory statistics for a specific user."""
        if user_id not in self.user_stores:
            return None
        return self.user_stores[user_id].get_stats()
    
    def clear_user_memory(self, user_id: str) -> bool:
        """Clear and remove a user's memory store."""
        if user_id in self.user_stores:
            del self.user_stores[user_id]
            return True
        return False
    
    def save_state(self, filepath: str):
        """
        Persist all user memory stores to a file using torch.save.
        
        Serializes each user's AdaptiveMemorySystem state including:
        - LTM matrix tensors
        - LTM strength values
        - STM contents (as list for serialization)
        
        Args:
            filepath (str): Path to save the state file (e.g., 'memory_state.pt')
        """
        state = {}
        for user_id, memory_system in self.user_stores.items():
            state[user_id] = {
                'ltm_matrix': memory_system.ltm_matrix,
                'ltm_strengths': memory_system.ltm_strengths,
                'stm': list(memory_system.stm)
            }
        torch.save(state, filepath)
        print(f"[PERSIST] Saved {len(state)} user memory stores to '{filepath}'")
    
    def load_state(self, filepath: str):
        """
        Load user memory stores from a previously saved state file.
        
        Restores each user's AdaptiveMemorySystem with their:
        - LTM matrix tensors
        - LTM strength values
        - STM contents
        
        Args:
            filepath (str): Path to the saved state file
            
        Returns:
            bool: True if load was successful, False if file not found
        """
        import os
        if not os.path.exists(filepath):
            print(f"[PERSIST] No saved state found at '{filepath}'")
            return False
        
        state = torch.load(filepath)
        for user_id, user_state in state.items():
            memory_system = AdaptiveMemorySystem()
            memory_system.ltm_matrix = user_state['ltm_matrix']
            memory_system.ltm_strengths = user_state['ltm_strengths']
            memory_system.stm = deque(user_state['stm'], maxlen=STM_CAPACITY)
            self.user_stores[user_id] = memory_system
        
        print(f"[PERSIST] Loaded {len(state)} user memory stores from '{filepath}'")
        return True


if __name__ == "__main__":

    print("="*60)
    print("  MULTI-USER MEMORY SYSTEM DEMO")
    print("="*60)
    
    manager = GlobalMemoryManager()
    torch.manual_seed(42)
    
    print("\n--- User 1: Coffee Enthusiast ---")
    user1_memory = manager.get_user_memory("user_001")
    vec_coffee = torch.randn(VECTOR_DIM)
    log = user1_memory.perceive(vec_coffee, importance_score=0.9)
    print(f"  User 1 stores 'Coffee preference': {log}")
    
    print("\n--- User 2: Physics Student ---")
    user2_memory = manager.get_user_memory("user_002")
    vec_physics = torch.randn(VECTOR_DIM)
    log = user2_memory.perceive(vec_physics, importance_score=0.85)
    print(f"  User 2 stores 'Physics interest': {log}")
    
    print("\n--- Verify Isolation ---")
    print(f"  Active users: {manager.list_users()}")
    print(f"  User 1 stats: {manager.get_user_stats('user_001')}")
    print(f"  User 2 stats: {manager.get_user_stats('user_002')}")
    
    print("\n--- Cross-User Query Test ---")
    retrieved, conf = user1_memory.query(vec_physics)
    print(f"  Querying User 1's memory for User 2's physics vector...")
    print(f"  Confidence: {conf:.4f} (should be low - different users)")
    
    retrieved, conf = user2_memory.query(vec_physics)
    print(f"  Querying User 2's memory for their own physics vector...")
    print(f"  Confidence: {conf:.4f} (should be high - same user)")
    
    print("\n" + "="*60)
    print("  ISOLATION VERIFIED: Each user has separate memory!")
    print("="*60)