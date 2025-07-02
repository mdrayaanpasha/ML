import random 
import math

corpus = ["I like NLP", "I like learning", "NLP is fun"]

"""
create a word2index adn indextoword with the help of the corpus
"""

words = " ".join(corpus).split()
word2Index = {word:idx for word,idx in enumerate(words)}
index2Word = {idx:word for word,idx in enumerate(words)}
vocab_size = len(corpus)
embedding_size = 5
epochs = 1000
negative_samples = 2
learning_rate=0.03

def random_vector(size):
  return [random.uniform(-1,1) for _ in range(size)]

w_input = {word:random_vector(embedding_size) for word in words}
w_output = {word:random_vector(embedding_size) for word in words}

def sigmoid(x):
  return 1 / (1+math.exp(-x))


def dot(a,b):
  return sum([a * b for a,b in zip(a,b)])

def update_gradient(center_word,context_word,learning_rate,label):
  v_c = w_input[center_word]
  u_o = w_output[context_word]

  dotp = dot(v_c,u_o)

  predicted = sigmoid(dotp)

  loss = -math.log(predicted) if label ==  1 else - math.log(1-predicted)

  grad = predicted - label

  for i in range(embedding_size):
    v_c[i] -= learning_rate * grad * u_o[i]

  for i in range(embedding_size):
    u_o[i] -= learning_rate * grad * v_c[i]

  return loss


training_pairs = [("I", "like"), ("like", "I"), ("like", "NLP"), ("I", "learning"), ("NLP", "is"), ("is", "fun")]

for epoch in range(epochs):
  total_loss = 0

  for center_word,context_word in training_pairs:

    total_loss+= update_gradient(center_word,context_word,1,0.03)


    negative_word = random.choice(list(words))
    while negative_word == context_word:
      negative_word = random.choice(list(words))

    total_loss += update_gradient(center_word,context_word,0.03,1)

  
  if epoch % 100 == 0:
    print(f"Epoch {epoch} Loss: ${total_loss}")




# 8. Final Word Embeddings
print("\nFinal Word Embeddings:")
for word in words:
    print(f"{word}: {w_input[word]}")

