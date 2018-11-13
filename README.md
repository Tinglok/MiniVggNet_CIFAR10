# MiniVggNet-CIFAR-10
Keras implementation of CIFAR-10, val_acc 90+%



## Requirements

numpy / scipy

tensorflow >= 1.8

tensorboard >= 1.8.0

Keras >= 2.0

matplotlib (optional, for visualization)
## Usage

### Prepare datasets

- CIFRA-10 dataset

#### Online Testing
Run directly the training script.

#### Offline Testing

##### For Windows user:
Follow `C:\Users\Administrator\.keras` for dataset preparation.

##### For Linux user:
Follow `root\\.keras\datasets` for dataset preparation.

** After setting up a dataset, you may want to train the model.

### Train model
There's a `train_network_keras.py` file at the root directory,
you can run it like this. 

```bash
python train_network_keras.py
```

### Evaluate pretrain model

There's a `evaluate_train_model.py` file at the evaluate_model directory, 
you can evaluate the model in the test set like this:

```bash
python evaluate_train_model.py
```

### Visualizing utils

- Visualizing cnn filter

There's a `visualizing_cnn_filter.py` file at the visualizing_utils directory, 
you can see the filters in the cnn layer like this:

```bash
python visualizing_cnn_filter.py
```

- Visualizing cnn graph

There's a `visualizing_cnn_graph.py` file at the visualizing_utils directory, 
you can see how is one of the training set train in the cnn layer like this:

```bash
python visualizing_cnn_filter.py
```

- Visualizing acc-loss graph

There's a `plot_acc_loss.py` file at the visualizing_utils directory, 
I have already stored the acc and loss data in the csv file, 
you can see the acc-loss graph like this:

```bash
python plot_acc_loss.py
```

### Tensorboard implementation

If you want to see more details in the model, 
you can use the tensorboard like this:

```bash
tensorboard --logdir=Graph
```
## Limitations

- Only single GPU training is implemented.

- Has not been tested and run in Windows system.
