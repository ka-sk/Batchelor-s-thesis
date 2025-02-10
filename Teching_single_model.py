from torch.utils.data import DataLoader
from pathlib import Path
from gnlse_dataset import GnlseDataset
from torchvision import transforms
import torch
import nn_model as model
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import functions_train_test as ttfun
from scipy.io import savemat


# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

# Data path in string format
data_path = Path('new_data/')
train_path = data_path / 'train'
test_path = data_path / 'test'

save_path = 'Results'

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# transform function
transform_fun = transforms.Compose([])

# getting dataset
train_data = GnlseDataset(data_dir=train_path,
                          transform=transform_fun)

test_data = GnlseDataset(data_dir=test_path,
                         transform=transform_fun)

# features to be predicted
features = train_data.labels

# Setup the batch size hyperparameter
BATCH_SIZE = 128

NUM_WORKERS = os.cpu_count()
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
                              batch_size=BATCH_SIZE,  # how many samples per batch?
                              shuffle=True  # shuffle data every epoch?
                              )

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False  # don't necessarily have to shuffle the testing data
                             )

img_custom, label_custom = next(iter(train_dataloader))
print(f"Image shape: {img_custom.shape} -> [batch_size, data]")
print(f"Label shape: {label_custom.shape} _> [batch_size, features]")

model_0 = model.CnnModel(input_shape=1,
                         hidden_units=20,
                         output_shape=len(features)).to(device)

# Set number of epochs
NUM_EPOCHS = 3

# Setup loss function and optimizer
loss_fn = loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=1)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model_0
model_0_results = ttfun.train(model=model_0,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS,
                        device=device)

# End the timer and print out how long it took
end_time = timer()

# End the timer and print out how long it took
end_time = timer()

savemat(mdict=model_0_results, file_name=os.path.join(save_path, 'results.mat'))

torch.save(obj=model_0.state_dict(),
           # only saving the state_dict() only saves the models learned parameters
           f = os.path.join(save_path, 'state_dict.pth'))

print(f"Total training time: {end_time-start_time:.3f} seconds")


############################################
# torchinfo, summary


