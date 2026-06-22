import torch
import torch.nn as nn
import torch.optim as optim

print("--- Large Concept Models: Continuous Sequence Prediction (Level 1 POC) ---\n")

# Suppose we have a vocabulary of 3 "Ideas" (Concepts)
# Idea 0: "Wake up"
# Idea 1: "Eat breakfast"
# Idea 2: "Go to work"

# Instead of tokens [0, 1, 2], our SONAR-like encoder embeds them as continuous vectors
concept_embeddings = {
    0: torch.tensor([0.9, 0.1]),  # Vector for "Wake up"
    1: torch.tensor([0.5, 0.5]),  # Vector for "Eat breakfast"
    2: torch.tensor([0.1, 0.9])   # Vector for "Go to work"
}

# The sequence we want the model to learn: 0 -> 1 -> 2
X = torch.stack([concept_embeddings[0], concept_embeddings[1]]) # Inputs: [Wake up, Eat breakfast]
Y = torch.stack([concept_embeddings[1], concept_embeddings[2]]) # Targets: [Eat breakfast, Go to work]

# A tiny "Concept Model" (Instead of predicting probabilities over vocabulary, 
# it outputs a continuous vector!)
class TinyConceptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
        
    def forward(self, x):
        return self.layer(x)

model = TinyConceptModel()

# Standard LLMs use CrossEntropyLoss. LCMs use continuous loss like MSE.
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Training the Concept Model (MSE Loss on Vectors)...")
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, Y)
    loss.backward()
    optimizer.step()

print(f"Final Loss: {loss.item():.4f}\n")

# Inference (Generation in Concept Space)
print("--- Generation in Concept Space ---")
input_concept = concept_embeddings[0] # Start with "Wake up"
print(f"Input Concept  : {input_concept.tolist()}  (Meaning: Wake up)")

predicted_concept_1 = model(input_concept)
print(f"Predicted Step 1: {predicted_concept_1.detach().tolist()} (Should be close to [0.5, 0.5])")

predicted_concept_2 = model(predicted_concept_1)
print(f"Predicted Step 2: {predicted_concept_2.detach().tolist()} (Should be close to [0.1, 0.9])\n")

print("Conclusion:")
print("We successfully predicted the next 'idea' entirely in continuous vector space,")
print("without ever using discrete tokens or a Softmax layer!")
