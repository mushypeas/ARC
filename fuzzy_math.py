import torch

'''
Fuzzy Set Theory Math Functions

'''
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

def fuzzy_min(arr1, arr2):
    assert type(arr1) is torch.Tensor
    assert type(arr2) is torch.Tensor
    arr1.to(DEVICE)
    arr2.to(DEVICE)
    
    output = torch.zeros(len(arr1)).to(DEVICE)
    for i in range(len(arr1)):
        val = torch.min(arr1[i], arr2[i])
        output[i] = val
    return output

def fuzzy_max(arr1, arr2):
    assert type(arr1) is torch.Tensor
    assert type(arr2) is torch.Tensor
    arr1.to(DEVICE)
    arr2.to(DEVICE)

    output = torch.zeros(len(arr1)).to(DEVICE)
    for i in range(len(arr1)):
        val = max(arr1[i], arr2[i])
        output[i] = val
    return output

if __name__ == '__main__':
    print(DEVICE)