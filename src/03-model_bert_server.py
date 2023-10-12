import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
root = "/home/hsotelo/modelling"
encoding = "iso-8859-1"
data_path = os.path.join(root,"01-bio_v1-manual.csv")
model_bert_name = 'bert-base-multilingual-cased'
df = pd.read_csv(data_path,encoding=encoding)

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_bert_name)

# Encode the labels for intent detection
label_encoder = LabelEncoder()
df['intent_label'] = label_encoder.fit_transform(df['intent'])

# Prepare data for NER
# In this example, we'll assume the 'bio_tags' column is already encoded with BIO tagging
# Modify this part based on your actual data preparation for NER
# The 'bio_tags' column should be a list of lists of encoded tags
# Each encoded tag corresponds to a token in the text
df['ner_tags'] = df['bio_tags'].apply(eval)  # Convert the string representation of lists to actual lists

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the tag mapping for NER
tag_map = {
    'O': 0,
    'B-place': 1,
    'I-place': 2,
    'B-crop': 3,
    'I-crop': 4,
    'B-measure': 5,
    'I-measure': 6,
    'B-date': 7,
    'I-date': 8
}

# Tokenize and encode the text data for NER
def encode_tags(tags, max_len):
    encoded_tags = []
    for tag in tags:
        encoded_tag = [tag_map.get(t, 0) for t in tag[:max_len]]
        encoded_tag.extend([0] * (max_len - len(encoded_tag)))
        encoded_tags.append(encoded_tag)
    return np.array(encoded_tags)

# Tokenize and encode the text data for intent detection
encoded_texts = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, return_tensors='tf')
intent_labels = tf.keras.utils.to_categorical(train_df['intent_label'], num_classes=len(label_encoder.classes_))
ner_tags = encode_tags(train_df['ner_tags'], max_len=len(encoded_texts['input_ids'][0]))

# Define the BERT-based model with two outputs
def build_model(num_intent_labels, num_ner_labels, max_len):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    bert_model = TFBertModel.from_pretrained(model_bert_name)
    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
    pooled_output = bert_outputs.pooler_output
    intent_output = Dense(num_intent_labels, activation='softmax', name="intent_output")(pooled_output)
    ner_output = Dense(num_ner_labels, activation='softmax', name="ner_output")(bert_outputs.last_hidden_state)
    model = Model(inputs=[input_ids, attention_mask], outputs=[intent_output, ner_output])
    return model


# Number of intent labels and NER labels
num_intent_labels = len(label_encoder.classes_)
num_ner_labels = len(tag_map)

# Max sequence length for padding
max_len = max(encoded_texts['input_ids'].shape[1], ner_tags.shape[1])

print("Model parameters",num_intent_labels,num_ner_labels,max_len)
# Build the model
model = build_model(num_intent_labels, num_ner_labels, max_len)

# Compile the model
model.compile(optimizer=Adam(learning_rate=5e-5),
              loss={'intent_output': 'categorical_crossentropy', 'ner_output': 'sparse_categorical_crossentropy'},
              metrics={'intent_output': 'accuracy', 'ner_output': 'accuracy'})

# Display the model summary
model.summary()

# TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
history = model.fit(
    {'input_ids': encoded_texts['input_ids'], 'attention_mask': encoded_texts['attention_mask']},
    {'intent_output': intent_labels, 'ner_output': ner_tags},
    validation_split=0.3,
    epochs=15,
    batch_size=30,
    callbacks=[tensorboard_callback]
)

# Tokenize and encode the text data for the test set
encoded_texts_test = tokenizer(test_df['text'].tolist(),padding='max_length',truncation=True,max_length=max_len, return_tensors='tf')
intent_labels_test = tf.keras.utils.to_categorical(test_df['intent_label'], num_classes=len(label_encoder.classes_))
ner_tags_test = encode_tags(test_df['ner_tags'], max_len=max_len)

intent_loss, intent_accuracy, _, ner_loss, ner_accuracy = model.evaluate(
    {'input_ids': encoded_texts['input_ids'], 'attention_mask': encoded_texts['attention_mask']},
    {'intent_output': intent_labels, 'ner_output': ner_tags},
)
# Evaluate the model on the test set
test_loss, test_intent_accuracy, _, test_ner_loss, test_ner_accuracy = model.evaluate(
    {'input_ids': encoded_texts_test['input_ids'], 'attention_mask': encoded_texts_test['attention_mask']},
    {'intent_output': intent_labels_test, 'ner_output': ner_tags_test},
)

print("Evalutation Training","Intent Loss:", intent_loss,"Intent Accuracy:", intent_accuracy,"NER Loss:", ner_loss,"NER Accuracy:", ner_accuracy)
print("Evalutation Test","Intent Loss:", test_loss,"Intent Accuracy:", test_intent_accuracy,"NER Loss:", test_ner_loss,"NER Accuracy:", test_ner_accuracy)

# Function to plot model performance
def plot_model_performance(history):
    plt.figure(figsize=(12, 8))
    # Plot training & validation accuracy values - intent
    plt.subplot(2, 2, 1)
    plt.plot(history.history['intent_output_accuracy'])
    plt.plot(history.history['val_intent_output_accuracy'])
    plt.title('Model Intent Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    # Plot training & validation loss values - intent
    plt.subplot(2, 2, 2)
    plt.plot(history.history['intent_output_loss'])
    plt.plot(history.history['val_intent_output_loss'])
    plt.title('Model Intent Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    # Plot training & validation accuracy values - ner
    plt.subplot(2, 2, 3)
    plt.plot(history.history['ner_output_accuracy'])
    plt.plot(history.history['val_ner_output_accuracy'])
    plt.title('Model NER Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    # Plot training & validation loss values - ner
    plt.subplot(2, 2, 4)
    plt.plot(history.history['ner_output_loss'])
    plt.plot(history.history['val_ner_output_loss'])
    plt.title('Model NER Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(root,"performance.png"), format='png', dpi=300)
    #plt.show()

# Plot model performance
plot_model_performance(history)

##########
test_predictions = model.predict({'input_ids': encoded_texts_test['input_ids'], 'attention_mask': encoded_texts_test['attention_mask']})

# Extract predicted intent labels and NER tags
predicted_intent_labels_test = test_predictions[0].argmax(axis=1)
predicted_ner_labels_test = test_predictions[1].argmax(axis=2).flatten()

def plot_confusion_matrix(intent_conf_matrix,ner_conf_matrix):
    plt.figure(figsize=(12, 8))
    # Plot training & validation accuracy values - intent
    plt.subplot(1, 2, 1)
    sns.heatmap(intent_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Intent Confusion Matrix')
    # Plot confusion matrix for NER
    plt.subplot(1, 2, 2)
    sns.heatmap(ner_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=tag_map.keys(), yticklabels=tag_map.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('NER Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(root,"confusion_matrix.png"), format='png', dpi=300)
    
# Plot confusion matrix for intent
intent_conf_matrix = confusion_matrix(test_df['intent_label'], predicted_intent_labels_test)
ner_conf_matrix = confusion_matrix(ner_tags_test.flatten(), predicted_ner_labels_test)
plot_confusion_matrix(intent_conf_matrix,ner_conf_matrix)

demeter_path = os.path.join(root,"demeter_model")
tf.keras.models.save_model(model,demeter_path)
