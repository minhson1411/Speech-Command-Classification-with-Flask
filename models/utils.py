import torch

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    tensor = tensor.cpu().detach().numpy()
    return tensor[0][0].argsort(axis=-1)[-3:][::-1]

def label_to_index(word, labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index, labels):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]
