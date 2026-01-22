import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


INPUT_DIM = 64
EMBEDDING_DIM = 384
LEARNING_RATE = 0.05


class TextEncoder(nn.Module):
    """
    Production Text Encoder using sentence-transformers.
    
    Uses the 'all-MiniLM-L6-v2' model to convert raw text into semantic embeddings.
    The model outputs 384-dimensional vectors which are projected down to 64 dimensions
    to match the system's INPUT_DIM via a learned linear layer.
    
    Features:
    - Real semantic embeddings (not random noise)
    - Cosine similarity between related texts will be high
    - Efficient inference with no gradient computation
    - Automatic device handling (CUDA/CPU compatibility)
    """
    
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.projection = nn.Linear(EMBEDDING_DIM, INPUT_DIM)
        self._init_projection()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.projection.to(self.device)
    
    def _init_projection(self):
        """Initialize projection layer with Xavier initialization for stability."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a text string into a semantic vector representation.
        
        Args:
            text (str): The raw text string to encode.
            
        Returns:
            torch.Tensor: A tensor of shape (64,) containing semantic embeddings (on CPU).
        """
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
            embedding = embedding.to(self.device)
            projected = self.projection(embedding)
        return projected.cpu()
    
    def encode_batch(self, texts: list) -> torch.Tensor:
        """
        Encode multiple text strings into semantic vectors.
        
        Args:
            texts (list): List of raw text strings to encode.
            
        Returns:
            torch.Tensor: A tensor of shape (batch_size, 64) (on CPU).
        """
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            embeddings = embeddings.to(self.device)
            projected = self.projection(embeddings)
        return projected.cpu()


class MemoryController(nn.Module):
    """
    The Decision-Making Brain of the Adaptive System.
    
    Role:
    - Does NOT store memories (stateless).
    - Acts as a Policy Network (Actor) in an RL context.
    - Decides "Importance" (Should I remember this?) and "Retrieval" (Do I need context?).
    
    This module bridges the gap between raw input and memory operations.
    """
    
    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.policy_net[2].bias.fill_(0.5)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.last_input = None
        self.last_prediction = None

    def assess_interaction(self, input_vector):
        """
        The Forward Pass. Analyzes an incoming signal and outputs control commands.
        
        Args:
            input_vector (Tensor): The encoded user interaction.
            
        Returns:
            dict: A control packet containing:
                - importance (float): Write-gate value (0.0 to 1.0)
                - retrieve (bool): Whether to query LTM
                - write_strength (float): Intensity of memory update
        """
        self.last_input = input_vector.clone().detach()
        
        with torch.no_grad():
            signals = self.policy_net(input_vector)
            
        importance_score = signals[0].item()
        retrieval_need = signals[1].item()
        write_strength = importance_score if importance_score > 0.5 else 0.0
        should_retrieve = retrieval_need > 0.6
        decision = {
            "importance": importance_score,
            "write_strength": write_strength,
            "should_retrieve": should_retrieve,
            "action_log": self._generate_log(importance_score, should_retrieve)
        }
        self.last_prediction = signals
        
        return decision

    def process_feedback(self, reward_signal):
        """
        The Learning Step (RL).
        Adjusts the controller's judgment based on success/failure.
        
        Args:
            reward_signal (float): 
                 +1.0 (User said "Good job remembering that")
                 -1.0 (User said "That's irrelevant" or "You hallucinated")
        """
        if self.last_input is None or self.last_prediction is None:
            return "No recent action to learn from."

        target_value = 1.0 if reward_signal > 0 else 0.0
        target_tensor = torch.tensor([target_value, target_value])
        self.optimizer.zero_grad()
        current_pred = self.policy_net(self.last_input)
        loss = F.mse_loss(current_pred, target_tensor)
        loss.backward()
        self.optimizer.step()
        return f"Controller weights updated. Loss: {loss.item():.4f}"


    def _generate_log(self, importance, retrieve):
        """Helper to create human-readable decision logs."""
        action = "IGNORE"
        if importance > 0.8: action = "CRITICAL WRITE"
        elif importance > 0.5: action = "NORMAL WRITE"
        
        return f"Decision: {action} (Imp: {importance:.2f}) | Retrieve: {retrieve}"

if __name__ == "__main__":
    print("Initializing Memory Controller (The Brain)...")
    brain = MemoryController()
    
    print("\n" + "="*50)
    print("--- TextEncoder with Real Semantic Embeddings ---")
    print("="*50)
    
    encoder = TextEncoder()
    
    texts = [
        "I love physics",
        "Physics is my favorite subject",
        "I enjoy cooking pasta"
    ]
    
    print("\nEncoding texts and checking semantic similarity:")
    vectors = []
    for text in texts:
        vec = encoder.encode(text)
        vectors.append(vec)
        print(f"\n  '{text}'")
        print(f"  Vector shape: {vec.shape}")
        print(f"  First 5 dims: {vec[:5].tolist()}")
    
    print("\n--- Semantic Similarity Matrix ---")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if i < j:
                sim = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
                print(f"  '{t1[:20]}...' vs '{t2[:20]}...': {sim.item():.4f}")
    
    print("\n--- Using encoded text with MemoryController ---")
    sample_text = "I love physics"
    text_vector = encoder.encode(sample_text)
    decision = brain.assess_interaction(text_vector)
    print(f"Decision for '{sample_text}': {decision['action_log']}")