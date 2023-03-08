
############################# LIBRARIES #############################

import gc
import os
from timeit import default_timer as timer
from typing import Dict

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

#####################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################## Train Functions #############################

def train(model, trainloader, optimizer, criterion, scaler):
    model.train()
    print('Training')

    train_running_loss = 0.0
    train_running_correct = 0

    counter = 0
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        
        # Extract data and load to device
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Lower precision to use less memory
        with torch.cuda.amp.autocast():
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)

        train_running_loss += loss.item()

        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # backpropagation
        scaler.scale(loss).backward()

        # update the optimizer parameters
        scaler.step(optimizer)
        scaler.update()

        # Garbage collection
        torch.cuda.empty_cache()
        _ = gc.collect()


    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc
# validation

def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            # Extract data and load to device
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Lower precision to use less memory
            with torch.cuda.amp.autocast():
                # forward pass
                outputs = model(image)
                # calculate the loss
                loss = criterion(outputs, labels)

            valid_running_loss += loss.item()

            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            # Garbage collection
            torch.cuda.empty_cache()
            _ = gc.collect()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# Test

def test(model, testloader, criterion):
    model.eval()
    print('Testing')
    test_running_loss = 0.0
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            # Extract data and load to device            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Lower precision to use less memory
            with torch.cuda.amp.autocast():
                # forward pass
                outputs = model(image)
                # calculate the loss
                loss = criterion(outputs, labels)
                
            test_running_loss += loss.item()
            
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()

            # Garbage collection
            torch.cuda.empty_cache()
            _ = gc.collect()
        
    # loss and accuracy for the complete epoch
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


#####################################################################

######################### Helper Functions ##########################

def setup_model_save_path(save_folder: str) -> str:
    """Initialize a folder path to store checkpoint informations

    Args:
        save_folder (str): Name of folder to keep checkpoints in 

    Returns:
        str: Path of folder to store checkpoints
    """
    project_base_path = os.getcwd()
    save_path = os.path.join(project_base_path, save_folder)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    return save_path


def save_checkpoint(state: dict, save_path: str, model_name: str) -> None:
    print("Saving checkpoint")

    filename = f'{model_name}_{state["epoch"]}.pth.tar'
    state_save_path = os.path.join(save_path, filename)

    torch.save(state, state_save_path)

def load_checkpoint(save_path:str, epoch:str) -> Dict:
    """Loads checkpoint and returns a state dictionary

    Args:
        save_path (str): Path of folder that contains checkpoints
        epoch (str): Checkpoint epoch to load from

    Returns:
        Dict: State dictionary
    """

    filename = f'{epoch}.pth.tar'
    file_path = os.path.join(save_path, filename)

    state = torch.load(file_path)
    return state

#####################################################################

############################# Configs ###############################

# Model Save Path
checkpoint_folder_name = 'Saves'
save_path = setup_model_save_path(checkpoint_folder_name)


start_epoch = 0 # No need to change

# Number of epochs to train the model for
num_epochs = 10

# Whether to load previous model or not
load_model = False

# Which epoch to load
load_epoch = 10

# Save checkpoints every x epochs
save_checkpoint_every = 5

#####################################################################


####################### Model Initialization ########################

trainloader = None
testloader = None
valloader = None

model: torch.nn.Module = None
optimizer: torch.optim.Optimizer = None
criterion = None

#####################################################################


####################### Load States #################################

if load_model:
    start_epoch = load_epoch
    
    # Load state from checkpoint
    state = load_checkpoint(save_path, 'model', load_epoch)

    model = model.load_state_dict(state['state_dict'])
    optimizer = optimizer.load_state_dict(state['optimizer'])

    train_loss = state['train_loss']
    train_acc = state['train_acc']
    val_loss = state['val_loss']
    val_acc = state['val_acc']
else:    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

#####################################################################


####################### Training Loop ###############################

# If gradient are float16 gradients, lower value of gradients may cause it to underflow to 0
# Gradients are scaled up before backpropagation and scaled down before optimizer step
scaler = torch.cuda.amp.GradScaler()


for epoch in range(load_epoch, load_epoch+num_epochs):
    print(f"Epoch: {epoch+1}/{load_epoch+num_epochs}")

    start_time = timer()

    train_epoch_loss, train_epoch_acc = train(model, trainloader, optimizer, criterion, scaler)
    end_time = timer()
    val_epoch_loss, val_epoch_acc = validate(model, valloader, criterion)

    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    train_acc.append(train_epoch_acc)
    val_acc.append(val_epoch_acc)

    print(f"Train loss: {train_epoch_loss:.3f}, \
          Train acc: {train_epoch_acc:.3f}, \
          Val loss: {val_epoch_loss:.3f}, \
          Val acc: {val_epoch_acc:.3f} \
          Epoch time = {(end_time - start_time):.3f}s")

    
    
    if (epoch+1) % save_checkpoint_every == 0:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        save_checkpoint(checkpoint)

#####################################################################

#################### Plot Training Results ##########################

plt.figure(figsize=(10, 7))
plt.plot(
    train_acc, color='green', linestyle='-', 
    label='Train Accuracy'
)
plt.plot(
    val_acc, color='blue', linestyle='-', 
    label='Validation Accuracy'
)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(
    train_loss, color='orange', linestyle='-', 
    label='Train Loss'
)
plt.plot(
    val_loss, color='red', linestyle='-', 
    label='Validation Loss'
)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


#####################################################################

###################### Test Model ###################################

test_loss, test_acc = test(model, testloader, criterion)
print(f"Test loss: {test_loss:.3f}, test acc: {test_acc:.3f}")

#####################################################################

