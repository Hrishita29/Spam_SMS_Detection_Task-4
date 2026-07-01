# Spam SMS Detection using Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

print(df.head())
print(df.info())


# Keep required columns
df = df[['v1','v2']]

# Rename columns
df.columns = ['label','message']


# Convert labels
df['label'] = df['label'].map({
    'ham':0,
    'spam':1
})


print(df['label'].value_counts())


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)


# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)


X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)



# Train Model
model = MultinomialNB()

model.fit(
    X_train,
    y_train
)


# Prediction
y_pred = model.predict(X_test)



# Evaluation

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)


print(
    classification_report(
        y_test,
        y_pred
    )
)



# Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred
)


sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Spam Detection Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()



# Test Custom Message

message = [
    "Congratulations! You won a free lottery prize. Call now!"
]


message_vector = tfidf.transform(message)


result = model.predict(message_vector)


if result[0] == 1:
    print("Spam Message")
else:
    print("Legitimate Message")