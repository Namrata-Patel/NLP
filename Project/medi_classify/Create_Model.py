import streamlit as st
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from torch.utils.data import DataLoader, Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits
    
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)



# Streamlit UI
st.title("BERT Text Classification with Streamlit")
data_file = st.file_uploader("Upload your CSV file", type=["csv"])
if data_file is not None:
    data = pd.read_csv(data_file)
    if 'text' in data.columns and 'label' in data.columns:
        st.write("Data successfully loaded!")

        bert_model_name = 'bert-base-uncased'
        num_classes = 2
        max_length = 128
        batch_size = 16
        num_epochs = 4
        learning_rate = 2e-5

        train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42)

        print(len(train_texts), len(train_labels))

        
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BERTClassifier(bert_model_name, num_classes).to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        if st.button("Train Model"):
            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}")
                train(model, train_dataloader, optimizer, scheduler, device)
                accuracy, report = evaluate(model, val_dataloader, device)
                print(f"Validation Accuracy: {accuracy:.4f}")
                print(report)
            st.success("Model trained successfully!")
        
        if st.button("Save Trained Model"):
            torch.save(model.state_dict(), "bert_classifier.pth")
            #save_model(model, label_encoder)
            st.success("Model and label encoder saved successfully!")
    else:
        st.error("CSV must include 'text' and 'label' columns")
else:
    st.warning("Please upload a CSV file to proceed.")