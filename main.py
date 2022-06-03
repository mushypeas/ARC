import torch
from PIL import Image
from torch import nn, optim
from fuzzy_art import FuzzyArt
from torchvision import transforms, datasets, models


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

# Hyperparameters
LEARNING_RATE = 0.1
CHOICE_PARAM = 1
VIGILENCE_PARAM = 0
EPOCHS     = 10
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./.data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./.data',
                   train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True)
    
def embed(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        
        embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data.squeeze())
        print(embedded.shape)
        
def train(model, train_loader, optimizer):
    assert type(model) is FuzzyArt
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE)[0], target.to(DEVICE)[0]
        print(target)
        data = torch.flatten(data)
        output = model.fuzzy_art(data, training=True)

def evaluate(model, test_loader, optimizer):
    assert type(model) is FuzzyArt
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(DEVICE)[0], target.to(DEVICE)[0]
        data = torch.flatten(data)
        output = model.fuzzy_art(data, training=False)


model = FuzzyArt(cp=CHOICE_PARAM, vp=VIGILENCE_PARAM, lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    # train(model, train_loader, optimizer=None)
    embed(train_loader)