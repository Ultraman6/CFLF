import numpy as np
import torch

# def getGradient(model_current, model_previous):
#     # Assuming model_current and model_previous are two PyTorch models with the same architecture
#     # state_dict_current = model_current.state_dict()
#     # state_dict_previous = model_previous.state_dict()
#     # Calculate the gradients by subtracting the parameters
#     return model_current - model_previous

# 根据模型获取梯度
import torch
from collections import OrderedDict


def getGradient(model_current, model_previous):
    gradient = OrderedDict()
    for (name, current_param), (_, previous_param) in zip(model_current.items(), model_previous.items()):
        # Ensure the parameter names from both models match
        assert name == _, "The models have different architectures or parameter orders"
        # Subtract the corresponding parameters of the models to get the gradient
        gradient[name] = current_param - previous_param
    return gradient

def bindGradient(model_previous, gradient_current):
    model_current = OrderedDict()
    print(type(model_previous))
    print(type(gradient_current))
    for (name, previous_param), (_, current_grad) in zip(model_previous.items(), gradient_current.items()):
        # Ensure the parameter names from both models match
        assert name == _, "The models have different architectures or parameter orders"
        # Subtract the corresponding parameters of the models to get the gradient
        model_current[name] = previous_param + current_grad
    return model_current


# 根据梯度获取范数
def calGradientNorm(gradient):
    # Flatten all tensors into a single vector and compute its norm
    flat_gradient = torch.cat([tensor.flatten() for tensor in gradient.values()])
    norm = torch.norm(flat_gradient)
    return norm


def calGradientDot(u, v):
    # Flatten all tensors into a single vector and compute its norm
    u_vector = torch.cat([param.data.flatten() for param in u.values()])
    v_vector = torch.cat([param.data.flatten() for param in v.values()])
    # Compute the dot product of the two vectors
    return torch.dot(u_vector, v_vector)


def gradient_flatten(gradient):
    # Flatten the gradients into a 1D vector
    return torch.cat([torch.flatten(x) for x in gradient.values()])


def gradient_flatten_and_shapes(gradient):
    flat_gradient = torch.cat([torch.flatten(x) for x in gradient.values()])
    original_shapes = [x.shape for x in gradient.values()]
    parameter_names = list(gradient.keys())
    return flat_gradient, original_shapes, parameter_names

def reconstruct_gradients(flat_gradient, original_shapes, parameter_names):
    position = 0
    reconstructed_gradients = OrderedDict()

    for shape, name in zip(original_shapes, parameter_names):
        num_elements = np.prod(shape)
        if num_elements != flat_gradient[position:position + num_elements].numel():
            raise ValueError(f"Shape mismatch for parameter {name}")

        segment = flat_gradient[position:position + num_elements].view(shape)
        reconstructed_gradients[name] = segment
        position += num_elements

    return reconstructed_gradients


# Example usage:
# flat_gradient is your flattened gradient tensor
# original_shapes is the list of original shapes of the gradient tensors
# reconstructed = reconstruct_gradients(flat_gradient, original_shapes)
