import random
import math

corpus = ["I like NLP", "I like learning", "NLP is fun"]

# Prepare the vocabulary
words = " ".join(corpus).split()
word2Index = {word: idx for idx, word in enumerate(words)}
index2Word = {idx: word for idx, word in enumerate(words)}
vocab_size = len(words)
embedding_size = 5
epochs = 1000
negative_samples = 2
learning_rate = 0.03

def random_vector(size):
    return [random.uniform(-1, 1) for _ in range(size)]

# Initialize embeddings
w_input = {word: random_vector(embedding_size) for word in words}
w_output = {word: random_vector(embedding_size) for word in words}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dot(a, b):
    return sum([x * y for x, y in zip(a, b)])

def update_gradient(center_word, context_word, label, learning_rate):
    v_c = w_input[center_word]
    u_o = w_output[context_word]

    dotp = dot(v_c, u_o)
    predicted = sigmoid(dotp)
    loss = -math.log(predicted) if label == 1 else -math.log(1 - predicted)
    grad = predicted - label

    for i in range(embedding_size):
        v_c[i] -= learning_rate * grad * u_o[i]
    for i in range(embedding_size):
        u_o[i] -= learning_rate * grad * v_c[i]

    return loss

# Training pairs (positive samples)
training_pairs = [("I", "like"), ("like", "I"), ("like", "NLP"), ("I", "learning"), ("NLP", "is"), ("is", "fun")]

# Training loop
for epoch in range(epochs):
    total_loss = 0

    for center_word, context_word in training_pairs:
        # Positive sample
        total_loss += update_gradient(center_word, context_word, 1, learning_rate)

        # Negative samples
        for _ in range(negative_samples):
            negative_word = random.choice(words)
            while negative_word == context_word:
                negative_word = random.choice(words)

            total_loss += update_gradient(center_word, negative_word, 0, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {total_loss}")

# Final word embeddings
print("\nFinal Word Embeddings:")
for word in words:
    print(f"{word}: {w_input[word]}")

# Cosine similarity
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)


# Nearest neighbors
def find_nearest_neighbors(word, top_k=3):
    similarities = {}

    for candidate in words:
        if candidate != word:
            similarity = cosine_similarity(w_input[word], w_input[candidate])  # Same embedding space
            similarities[candidate] = similarity

    sorted_candidates = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_candidates[:top_k]

# Example usage
target_word = "I"
neighbors = find_nearest_neighbors(target_word)

print(f"\nTop neighbors for '{target_word}':")
for neighbor, score in neighbors:
    print(f"{neighbor} with similarity {score:.4f}")
