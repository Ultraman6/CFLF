import torch


def getGradient(model_current, model_previous):
    # Assuming model_current and model_previous are two PyTorch models with the same architecture
    state_dict_current = model_current.state_dict()
    state_dict_previous = model_previous.state_dict()
    # Calculate the gradients by subtracting the parameters
    return state_dict_current - state_dict_previous

def gradient_flatten(gradient):
    # Flatten the gradients into a 1D vector
    return torch.cat([torch.flatten(x) for x in gradient.values()])