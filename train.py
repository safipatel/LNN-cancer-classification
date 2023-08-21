import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from models import CNN_Net, LNN
import medmnist
import matplotlib.pyplot as plt
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)


# CONSTANTS
HIDDEN_NEURONS = 19     # how many hidden neurons for CFC
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_OUTPUT_CLASSES = 2
# Constants based off CNN architecture
NCP_INPUT_SIZE = 16
SEQUENCE_LENGTH = 32


## FLAGS
MODEL_ARCH = "LNN"       # Models: "LNN" or "CNN" or "DNN"
SAVE_OR_LOAD = "SAVE" # "SAVE" to save, "LOAD" to load, None to disable
MODEL_PATH = "saved_models/LNN_SAVE_full"    # path to load/save

# Load in MedMNIST data
cancer_info = medmnist.INFO["breastmnist"]   # BINARY CLASSIFICATION 1x28x28
labels = cancer_info['label'] #{'0': 'malignant', '1': 'normal, benign'}
data_file = medmnist.BreastMNIST

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.5], std = [0.5])])

train_set = data_file(split = 'train', transform = transform, download = False)    # set these download flags to True for the first run if not already downloaded
train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
test_set = data_file(split = 'test', transform = transform, download = False)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = BATCH_SIZE, shuffle = False)
val_set = data_file(split = 'val', transform = transform, download = False)
val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = BATCH_SIZE, shuffle = False)


if MODEL_ARCH == "LNN":
    model = LNN(NCP_INPUT_SIZE, HIDDEN_NEURONS, NUM_OUTPUT_CLASSES, SEQUENCE_LENGTH).to(device)
elif MODEL_ARCH == "CNN":
    model = CNN_Net(in_channels=1, num_classes=NUM_OUTPUT_CLASSES).to(device)



print("Amount of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


### TRAINING SECTION

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

train_loss_dict = {}
train_acc_dict = {}
val_acc_dict = {}

if SAVE_OR_LOAD == "LOAD":
    print("Loading presaved model at ", MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH))
    with open(MODEL_PATH+"_train_data", 'rb') as handle:
        train_loss_dict, train_acc_dict, val_acc_dict = pickle.load(handle)
else:
    n_total_steps = len(train_loader)
    for epoch in tqdm(range(NUM_EPOCHS)):

        model.train()
        # Keep track of metrics as we go along
        train_loss, train_count = 0, 0
        n_train_correct, n_train_samples = 0, 0
        n_val_correct, n_val_samples = 0, 0

        for i, (images, labels) in tqdm(enumerate(train_loader)):  

            # Put data on GPU before running
            images = images.to(device)
            labels = labels.to(device)

            labels = labels.squeeze().long()
            
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            predicted = torch.argmax(outputs.data, 1)
            
            # Store training loss and accuracy
            n_train_samples += labels.size(0)
            n_train_correct += (predicted == labels).sum().item()
            
            train_loss += loss.item()
            train_count += 1
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 5 == 0:
                print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                
        train_loss_dict[epoch] = train_loss / train_count
        train_acc_dict[epoch] = n_train_correct / n_train_samples

        # Test model on validation every epoch and record accuracy
        model.eval()
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.squeeze().long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            predicted = torch.argmax(outputs.data, 1)
            
            n_val_samples += labels.size(0)
            n_val_correct += (predicted == labels).sum().item()
        
        val_acc_dict[epoch] = n_val_correct / n_val_samples

# Saves model if enabled
    if SAVE_OR_LOAD == "SAVE":
        torch.save(model.state_dict(), MODEL_PATH)
        with open(MODEL_PATH + "_train_data", 'wb') as handle:
            pickle.dump([train_loss_dict, train_acc_dict, val_acc_dict], handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved model at ", MODEL_PATH)

### Graph training loss over epochs
epochs = range(0, NUM_EPOCHS)

plt.plot(epochs, [x for x in train_loss_dict.values()], label='Training Loss')

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.xticks(range(0, NUM_EPOCHS, 10))
plt.legend(loc='best')
plt.show()


### Graph validation/testing accuracy
epochs = range(0, NUM_EPOCHS)

plt.plot(epochs, [x for x in train_acc_dict.values()], label='Training Accuracy')
plt.plot(epochs, val_acc_dict.values(), label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.xticks(range(0, NUM_EPOCHS, 10))
plt.legend(loc='best')
plt.show()

print("Final validation accuracy: ", val_acc_dict.get(NUM_EPOCHS - 1))


# TESTING

model.eval()
y_true = torch.tensor([]).to(device)
y_score = torch.tensor([]).to(device)


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        labels = labels.squeeze().long()
        outputs = outputs.softmax(dim=-1)
        labels = labels.float().resize_(len(labels), 1)

        y_true = torch.cat((y_true, labels), 0)
        y_score = torch.cat((y_score, outputs), 0)

    y_true = y_true.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    
    evaluator = medmnist.Evaluator('breastmnist', 'test')
    metrics = evaluator.evaluate(y_score)

    print('%s  Area under curve (AUC): %.4f  Accuracy (ACC): %.4f' % ("Test metrics", *metrics))
