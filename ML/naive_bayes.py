from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample training data
texts = ["win free lottery now", "meeting scheduled for Monday", "claim your free prize", "project update meeting"]
labels = ["spam", "ham", "spam", "ham"] # corresponding labels

# Create a pipeline: first convert text to word counts, then apply Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(texts, labels)

# Predict a new text
new_email = "free lottery meeting"
prediction = model.predict([new_email])

print(f"The email '{new_email}' is classified as: {prediction[0]}")
# Output: The email 'free lottery meeting' is classified as: spam