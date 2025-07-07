from collections import defaultdict

# Step 1: Setup Dataset
train_data = [
    ("Free money offer", "Spam"),
    ("Win money now", "Spam"),
    ("Call me later", "NotSpam"),
    ("Let’s have lunch", "NotSpam")
]

# Step 2: Initialize structures
word_counts = defaultdict(lambda: defaultdict(int))
class_counts = defaultdict(int)
label_counts = defaultdict(int)
vocab = set()

# Step 3: Build word counts per label
for message, label in train_data:
    words = message.lower().split()
    label_counts[label] += 1  # Count how many messages per label
    class_counts[label] += len(words)  # Total words in this label
    for word in words:
        word_counts[label][word] += 1
        vocab.add(word)

vocab_size = len(vocab)
total_messages = len(train_data)

# Step 4: Calculate Priors
priors = {}
for label in label_counts:
    priors[label] = label_counts[label] / total_messages

print("=== Priors ===")
for label in priors:
    print(f"P({label}) = {priors[label]:.4f}")

# Step 5: Calculate Likelihoods with Laplace Smoothing
likelihoods = defaultdict(lambda: defaultdict(float))

for label in word_counts:
    total_words_in_label = class_counts[label]
    for word in vocab:
        count = word_counts[label][word]
        likelihoods[label][word] = (count + 1) / (total_words_in_label + vocab_size)

print("\n=== Likelihoods ===")
for label in likelihoods:
    print(f"\nGiven {label}:")
    for word in vocab:
        print(f"P({word} | {label}) = {likelihoods[label][word]:.4f}")
