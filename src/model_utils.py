import os
import torch


def save_model(model, filepath):
    directorio = "./models/"
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    filepath = directorio + filepath
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model
