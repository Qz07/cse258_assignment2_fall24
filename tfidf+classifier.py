import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom Dataset
class ReviewDataset(Dataset):
    def __init__(self, embeddings, ratings):
        self.embeddings = embeddings
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.ratings[idx], dtype=torch.long)

# Define Model
class RatingClassifier(nn.Module):
    def __init__(self, input_size, num_classes=6):
        super(RatingClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load Data
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Process Text to Embeddings
def preprocess_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=500)  # Adjust features based on need
    train_embeddings = vectorizer.fit_transform(train_texts).toarray()
    test_embeddings = vectorizer.transform(test_texts).toarray()
    return train_embeddings, test_embeddings

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    return acc

# Main
if __name__ == "__main__":
    # Load data
    train_texts = load_pickle("train_texts.pickle")
    train_ratings = load_pickle("train_ratings.pickle")
    test_texts = load_pickle("test_texts.pickle")
    test_ratings = load_pickle("test_ratings.pickle")

    # Preprocess text to embeddings
    train_embeddings, test_embeddings = preprocess_text(train_texts, test_texts)

    # Create datasets
    train_dataset = ReviewDataset(train_embeddings, train_ratings)
    test_dataset = ReviewDataset(test_embeddings, test_ratings)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model setup
    input_size = train_embeddings.shape[1]  # Dimensionality of embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RatingClassifier(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    evaluate_model(model, test_loader)
