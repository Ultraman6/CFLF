settings = {
    "deep_learning_settings": [{
        "name": "dataset",
        "type": "str",
        "default": "mnist",
        "option": ["mnist", "cifar10", "cifar100", "femnist", "fashionmnist", "synthetic", "shakespare"],
        "description": "The name of the dataset to use."
    }, {
        "name": "model",
        "type": "str",
        "default": "cnn",
        "option": {
            "mnist": ["cnn", "logistic"],
            "cifar10": ["cnn", "resnet18"],
            "cifar100": ["cnn", "resnet18"],
            "femnist": ["cnn", "resnet18", "logistic"],
            "fashionmnist": ["cnn", "logistic"],
            "synthetic": ["logistic"],
            "shakesparea": ["lstm"]
        },
        "description": "Model architecture of dataset to use."
    }, {
        "name": "init_mode",
        "type": "str",
        "default": "xaiver_uniform",
        "option": ["xaiver_uniform", "kaiming_uniform", "kaiming_normal", "xavier_normal", "xavier_uniform"],
        "description": "Initialization mode for model weights."
    }, {
        "name": "batch_size",
        "type": "int",
        "default": 10,
        "description": "Batch size when trained and tested"
    }, {
        "name": "learning_rate",
        "type": "float",
        "default": 0.01,
        "description": "Learning rate when training"
    }, {
        "name": "loss_function",
        "type": "str",
        "default": 'ce',
        "option": ['ce', 'bce', 'mse'],
        "description": "Learning rate when training"
    }, {
        "name": "optimizer",
        "type": "str",
        "default": "sgd",
        "option": ["sgd", "adam"],
        "description": "Optimizer to use for training."
    }, {
        "name": "scheduler",
        "type": "str",
        "default": "none",
        "option": ["none", "step", "plateau", "cosine"],
        "description": "Learning rate scheduler to use."
    }, {
        "name": "optimizer_settings",
        "type": "float",
        "default": 0.0,
        "description": "Momentum for the optimizer."
    },
    ]
}

deep_learning_settings = [
    {
        "name": "epochs",
        "type": "int",
        "default": 10,
        "description": "Number of epochs to train the model."
    },
    {
        "name": "batch_size",
        "type": "int",
        "default": 32,
        "description": "How many samples per batch to load."
    },
    {
        "name": "learning_rate",
        "type": "float",
        "default": 0.01,
        "description": "Learning rate for the optimizer."
    },
    {
        "name": "momentum",
        "type": "float",
        "default": 0.5,
        "description": "Momentum for the optimizer."
    },
    {
        "name": "seed",
        "type": "int",
        "default": 1,
        "description": "Random seed for reproducibility."
    },
    {
        "name": "log_interval",
        "type": "int",
        "default": 10,
        "description": "How many batches to wait before logging training status."
    },
    {
        "name": "save_model",
        "type": "bool",
        "default": False,
        "description": "For Saving the current Model."
    },
    {
        "name": "standalone",
        "type": "bool",
        "default": False,
        "description": "Run in standalone mode without federation."
    }
]

dataset_settings = [
    {
        "name": "dataset_root",
        "type": "str",
        "default": "D:\\datasets for CFLF",
        "description": "Root directory for dataset storage."
    },
    {
        "name": "dataset",
        "type": "str",
        "default": "MNIST",
        "description": "The name of the dataset to use."
    },
    {
        "name": "train_split",
        "type": "float",
        "default": 0.8,
        "description": "Fraction of data to be used for training."
    },
    {
        "name": "use_cuda",
        "type": "bool",
        "default": False,
        "description": "Enable CUDA for model training."
    },
    {
        "name": "sample_mapping",
        "type": "str",
        "default": "{}",
        "description": "JSON string for sample mapping in federated setting."
    },
    {
        "name": "class_mapping",
        "type": "str",
        "default": "{}",
        "description": "JSON string for class distribution among clients."
    },
    {
        "name": "noise_mapping",
        "type": "str",
        "default": "{}",
        "description": "JSON string for noise distribution among clients."
    }
]

advanced_settings = [
    {
        "name": "optimizer",
        "type": "str",
        "default": "SGD",
        "description": "Optimizer to use for training."
    },
    {
        "name": "model",
        "type": "str",
        "default": "CNN",
        "description": "Model architecture to use."
    },
    {
        "name": "data_augmentation",
        "type": "bool",
        "default": False,
        "description": "Apply data augmentation techniques."
    },
    {
        "name": "early_stopping",
        "type": "bool",
        "default": False,
        "description": "Enable early stopping."
    },
    {
        "name": "early_stopping_patience",
        "type": "int",
        "default": 10,
        "description": "Patience for early stopping."
    },
    {
        "name": "weight_decay",
        "type": "float",
        "default": 0.0,
        "description": "Weight decay (L2 penalty)."
    },
    {
        "name": "dropout_rate",
        "type": "float",
        "default": 0.5,
        "description": "Dropout rate for dropout layers."
    },
    {
        "name": "activation_function",
        "type": "str",
        "default": "ReLU",
        "description": "Activation function to use."
    }
]
