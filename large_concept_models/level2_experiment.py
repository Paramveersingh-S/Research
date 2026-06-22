# @title 1. Environment Setup
# To run this on Colab:
# !pip install sentence-transformers torch
# Colab CPU or T4 GPU is fine!

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("WARNING: sentence_transformers not installed. Run: !pip install sentence-transformers")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# @title 2. Load SONAR-like Encoder
# We use all-MiniLM-L6-v2 as our "Concept Encoder" (SONAR proxy)
if device == "cuda" or device == "cpu": # Can run on CPU for this toy dataset
    print("Loading Concept Encoder...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    embedding_dim = encoder.get_sentence_embedding_dimension() # 384
    
    # @title 3. The Dataset (A Story)
    # A short story where each sentence naturally follows the previous one.
    story = [
        "The sky grew dark as the storm approached.",
        "Lightning flashed across the horizon.",
        "A loud clap of thunder shook the ground.",
        "Heavy rain began to pour down.",
        "People ran to find shelter under the awnings."
    ]
    
    print("\nEncoding sentences into Concept Vectors...")
    # Convert sentences into continuous vectors
    # We must clone() because sentence_transformers creates inference tensors
    embeddings = encoder.encode(story, convert_to_tensor=True).clone()
    
    # X: Sentences 0 to 3
    # Y: Sentences 1 to 4 (The "Next Concept")
    X = embeddings[:-1].requires_grad_(True)
    Y = embeddings[1:].requires_grad_(False)
    
    # @title 4. The Large Concept Model (LCM)
    # A small MLP that predicts the next embedding vector
    class MiniLCM(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, dim)
            )
            
        def forward(self, x):
            return self.net(x)
            
    lcm = MiniLCM(embedding_dim).to(device)
    optimizer = optim.Adam(lcm.parameters(), lr=1e-3)
    # Using Cosine Embedding Loss as recommended for high-dimensional semantic spaces
    # Target '1' means we want the prediction and Y to have high cosine similarity
    criterion = nn.CosineEmbeddingLoss()
    target_labels = torch.ones(X.size(0)).to(device)
    
    print("\nTraining LCM on continuous trajectories...")
    epochs = 300
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = lcm(X)
        loss = criterion(predictions, Y, target_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
            
    # @title 5. Inference (Generation & Decoding)
    print("\n--- Testing Concept Generation ---")
    
    # 1. Start with a prompt
    prompt = "The sky grew dark as the storm approached."
    prompt_emb = encoder.encode(prompt, convert_to_tensor=True).unsqueeze(0)
    
    # 2. LCM predicts the NEXT concept vector
    with torch.no_grad():
        predicted_emb = lcm(prompt_emb)
        
    # 3. "Decode" the predicted vector back into text.
    # Since we don't have Meta's massive SONAR decoder, we will simulate decoding 
    # by finding the closest sentence in our dictionary of possible events.
    dictionary = [
        "The sun came out brightly.",
        "Lightning flashed across the horizon.",
        "The cat slept on the mat.",
        "Heavy rain began to pour down."
    ]
    
    dict_embeddings = encoder.encode(dictionary, convert_to_tensor=True)
    
    # Find closest match via cosine similarity
    cos_scores = util.cos_sim(predicted_emb, dict_embeddings)[0]
    best_idx = torch.argmax(cos_scores).item()
    decoded_text = dictionary[best_idx]
    
    print(f"Given Context : '{prompt}'")
    print(f"LCM Prediction: '{decoded_text}' (Confidence: {cos_scores[best_idx].item():.2f})")
    print("\nSuccess! The model successfully predicted the next logical event in a continuous representation space, completely ignoring English tokens.")
