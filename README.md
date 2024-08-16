# Text-Classification-From-Scratch




## Introduction

Text classification is a crucial task in Natural Language Processing (NLP) that involves categorizing text into predefined labels. This project focuses on building a text classification model from scratch using the IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset. Our goal is to classify text data based on emotional content, exploring the depths of emotion recognition through deep learning.

The IEMOCAP dataset is particularly rich in emotional annotations, making it an excellent resource for this type of task. In this project, we guide you through the entire process—ranging from data preprocessing and model development to evaluation—using the Keras framework.
## Libraries Used

 ● Pandas

● Numpy

● Os

 ● Tensorflow

 ● Matplotlib

● Seaborn
# Instructions

## 1.Import Libraries 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support



# 2.Extraction


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    return text

dataset_path = r"D:\New folder\IEMOCAP_full_release"

texts = []
labels = []
for session in range(1, 6):
    session_path = os.path.join(dataset_path, f'Session{session}', 'dialog', 'transcriptions')
    for root, _, files in os.walk(session_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ')
                        label = parts[0]
                        text = ' '.join(parts[1:])
                        texts.append(clean_text(text))
                        labels.append(label)
df = pd.DataFrame({'text': texts, 'label': labels})
csv_file_path = "cleaned.csv"
df.to_csv(csv_file_path, index=False)


# 3.Labelling

file_path = r"C:\Users\SAKSHEE\Downloads\cleaned.csv"
data = pd.read_csv(file_path)

happy_keywords = ["good","know","im","love","like","mean","fine","happy","going","get","excited","thats","okay","youre","theyre","right","yeah","spot","thing","hey"]
sad_keywords = ['sad', 'unhappy', 'miserable','hate','oh','help', 'sorrow', 'upset', 'depressed', 'downcast', 'gloomy','dont','cant']

def label_text(text):
    text_lower = text.lower()
    happy_count = sum(keyword in text_lower for keyword in happy_keywords)
    sad_count = sum(keyword in text_lower for keyword in sad_keywords)
    
    if happy_count > sad_count:
        return 'happy'
    elif sad_count > happy_count:
        return 'sad'
    else:
        return 'neutral'

data['label'] = data['text'].apply(label_text)

output_file_path = r"C:\Users\SAKSHEE\Downloads\relabelled_data5.csv"
data.to_csv(output_file_path, index=False)

label_distribution = data['label'].value_counts()
print(label_distribution)


# 4.Preprocessing
df = pd.read_csv(r"C:\Users\Satyam Sangar\Downloads\relabelled_data5.csv")

texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

emotion_to_index = {emotion: idx for idx, emotion in enumerate(set(labels))}
index_to_emotion = {idx: emotion for emotion, idx in emotion_to_index.items()}

labels_indices = [emotion_to_index[label] for label in labels]

num_classes = len(emotion_to_index)
labels_one_hot = to_categorical(labels_indices, num_classes=num_classes)

texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels_one_hot, test_size=0.2, random_state=42)

texts_train, texts_val, labels_train, labels_val = train_test_split(texts_train, labels_train, test_size=0.1, random_state=42)


# 5.Vectorization
max_features = 10000  
sequence_length = 50  

vectorize_layer = TextVectorization(max_tokens=max_features, output_sequence_length=sequence_length)

vectorize_layer.adapt(texts_train)

from tensorflow.keras.layers import Dropout, LSTM

train_texts_tensor = tf.constant(texts_train, dtype=tf.string)
val_texts_tensor = tf.constant(texts_val, dtype=tf.string)
test_texts_tensor = tf.constant(texts_test, dtype=tf.string)

model = Sequential([
    vectorize_layer,
    Embedding(input_dim=max_features, output_dim=128),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),  # Added Dropout layer
    Dense(num_classes, activation='softmax')
])


# 6.Training
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_texts_tensor, np.array(labels_train), epochs=10, validation_data=(val_texts_tensor, np.array(labels_val)))

val_results = model.evaluate(tf.constant(texts_val, dtype=tf.string), np.array(labels_val))
test_results = model.evaluate(tf.constant(texts_test, dtype=tf.string), np.array(labels_test))

print(f"Validation Loss: {val_results[0]}, Validation Accuracy: {val_results[1]}")
print(f"Test Loss: {test_results[0]}, Test Accuracy: {test_results[1]}")


# 7.Evaluation
y_true = np.argmax(labels_test, axis=1)
y_pred = np.argmax(model.predict(tf.constant(texts_test, dtype=tf.string)), axis=1)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)


# 8.Metrics

report = classification_report(y_true, y_pred, target_names=index_to_emotion.values())
print("Classification Report:")
print(report)


# 9.Inference

new_texts = [
    "I'm excited about starting my new job!",  
    "Oh, so sad", 
    "I have a routine day at work ahead."  
]

new_texts_tensor = tf.constant(new_texts, dtype=tf.string)
predictions = model.predict(new_texts_tensor)

predicted_indices = np.argmax(predictions, axis=1)
predicted_emotions = [index_to_emotion[idx] for idx in predicted_indices]

for text, emotion in zip(new_texts, predicted_emotions):
    print(f"Text: '{text}' - Predicted Emotion: {emotion}")


# 10.ROC Curve Visualization

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

labels_binarized = label_binarize(y_true, classes=np.arange(num_classes))
y_scores = model.predict(tf.constant(texts_test, dtype=tf.string))

fpr, tpr, _ = roc_curve(labels_binarized.ravel(), y_scores.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# 11.Class-wise Precision, Recall, and F1-Score

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

x = np.arange(len(precision))  
width = 0.05  

plt.figure(figsize=(8, 4))
plt.bar(x - width, precision, width, label='Precision', color='lightblue')
plt.bar(x, recall, width, label='Recall', color='lightgreen')
plt.bar(x + width, f1_score, width, label='F1-Score', color='salmon')

plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score per Class')
plt.xticks(x, index_to_emotion.values())
plt.legend()
plt.show()


# 12.Misclassification Proportion

misclassifications = y_true != y_pred
misclassifications_count = sum(misclassifications)

plt.figure(figsize=(8, 8))
plt.pie([misclassifications_count, len(y_true) - misclassifications_count], 
        labels=['Misclassifications', 'Correct Predictions'], 
        autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title('Proportion of Misclassifications')
plt.show()



# 13.Training and Validation Metrics

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# 14.Confusion Matrix Heatmap

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=index_to_emotion.values(), yticklabels=index_to_emotion.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# 15.Model AUC Comparison


y_true_bin = np.array(labels_test)

model_predictions = {
    "CNN": cnn_predictions,
    "LSTM": lstm_predictions,
    "Logistic Regression": lr_predictions
}

auc_scores = {}
for model_name, predictions in model_predictions.items():
    auc_scores[model_name] = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        auc_scores[model_name][index_to_emotion[i]] = auc(fpr, tpr)

auc_df = pd.DataFrame(auc_scores)

print(auc_df)

auc_df.plot(kind='bar', figsize=(6, 5))
plt.title('AUC-ROC Scores for Each Category Across Models')
plt.xlabel('Categories')
plt.ylabel('AUC Score')
plt.xticks(rotation=45)
plt.legend(title='Models')
plt.show()

