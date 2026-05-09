import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer


# testing if GPU is available and printing CUDA version
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print(torch.version.cuda)



# Preparing the data for training and testing
print("Loading data...")

train_df = pd.read_csv('sentiment_data.tsv', sep='\t')
X_train = train_df['Text'].values
y_train = train_df['Label'].values

test_df = pd.read_csv('sentiment_test_data.tsv', sep='\t')
X_test = test_df['Text'].values
y_test = test_df['Label'].values

# Convert text into numerical data (TF-IDF vectors)
# This turns words into numbers so the neural network can understand them


# helper list for stop_words
custom_stop_words = [
    'a', 'about', 'above', 'after', 'again', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'by', 'can', 
    'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 
    'most', 'my', 'myself', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 
    'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 
    'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 
    'up', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 
    'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
]



vectorizer = TfidfVectorizer(max_features=1000, stop_words=custom_stop_words)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Creating a custom Dataset class to feed data into the model in batches
class SentimentDataset(Dataset):
    def __init__(self, features, labels):
        # Convert data to PyTorch Tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets
train_dataset = SentimentDataset(X_train_vec, y_train)
test_dataset = SentimentDataset(X_test_vec, y_test)

# Create dataloaders (feeds data into the model in batches)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Define the neural network architecture
class SentimentModel(nn.Module):
    def __init__(self, input_dim):
        super(SentimentModel, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Layer 2 (Output Layer)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


input_dim = X_train_vec.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SentimentModel(input_dim)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)





# Train the model
epochs = 15
print("\nStarting Training...")

for epoch in range(epochs):
    model.train() 
    total_loss = 0
    
    for batch_features, batch_labels in train_loader:
        # 1. Clear old gradients
        optimizer.zero_grad()
        
        # 2. Forward pass (Make predictions)
        predictions = model(batch_features).squeeze() 
        
        # 3. Calculate loss (How wrong were the predictions)
        loss = criterion(predictions, batch_labels)
        
        # 4. Backward pass (Calculate gradients)
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# Evaluate the model on the test set
model.eval() # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad(): # Don't track gradients during testing
    for batch_features, batch_labels in test_loader:
        predictions = model(batch_features).squeeze()
        # If prediction > 0.5, guess 1 (Positive), else guess 0 (Negative)
        predicted_labels = (predictions > 0.5).float()
        
        total += batch_labels.size(0)
        correct += (predicted_labels == batch_labels).sum().item()

accuracy = (correct / total) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%\n")

# Make predictions on new reviews
def predict_sentiment(text):
    model.eval()
    # 1. Vectorize the text using the same vectorizer
    vec_text = vectorizer.transform([text]).toarray()
    
    # 2. Convert to PyTorch Tensor
    tensor_text = torch.tensor(vec_text, dtype=torch.float32)
    
    # 3. Get prediction
    with torch.no_grad():
        prediction = model(tensor_text).item()
        
    # 4. Interpret result
    sentiment = "Positive 🟢" if prediction >= 0.5 else "Negative 🔴"
    print(f"Review: '{text}'")
    print(f"Sentiment: {sentiment} (Confidence score: {prediction:.4f})\n")

print("Enter review or type \"Close\" to exit:")
while True:
    user_input = input("Review: ")
    if user_input.lower() == "close":
        break
    predict_sentiment(user_input)