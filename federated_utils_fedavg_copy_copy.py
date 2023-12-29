
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# def load(paths, verbose=-1):
#     '''expects images for each class in seperate dir, 
#     e.g all digits in 0 class in the directory named 0 '''
#     data = list()
#     labels = list()
#     # loop over the input images
#     for (i, imgpath) in enumerate(paths):
#         # load the image and extract the class labels
#         im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
#         image = np.array(im_gray).flatten()
#         label = imgpath.split(os.path.sep)[-2]
#         # scale the image to [0, 1] and add to list
#         data.append(image/255)
#         labels.append(label)
#         # show an update every `verbose` images
#         if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
#             print("[INFO] processed {}/{}".format(i + 1, len(paths)))
#     # return a tuple of the data and labels
#     return data, labels


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))}



from torch.utils.data import DataLoader, TensorDataset

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a DataLoader object off it
    args:
        shard: a data, label constituting a client's data shard
        bs: batch size
    return:
        DataLoader object'''
    if not data_shard:
        # Handle empty data shard (return an empty DataLoader or handle it accordingly)
        return DataLoader(TensorDataset(torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)), batch_size=bs, shuffle=False)

    # separate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = TensorDataset(torch.tensor(list(data), dtype=torch.float32), torch.tensor(list(label), dtype=torch.float32))
    return DataLoader(dataset, batch_size=bs, shuffle=True)





# def batch_data(data_shard, bs=32):
#     '''Takes in a clients data shard and create a DataLoader object off it
#     args:
#         shard: a data, label constituting a client's data shard
#         bs: batch size
#     return:
#         DataLoader object'''
#     # separate shard into data and labels lists
#     data, label = zip(*data_shard)
#     dataset = TensorDataset(torch.tensor(list(data), dtype=torch.float32), torch.tensor(list(label), dtype=torch.float32))
#     return DataLoader(dataset, batch_size=bs, shuffle=True)
# Example usage:
# clients_batched = dict()
# for (client_name, data) in clients.items():
#     clients_batched[client_name] = batch_data(data, bs=32)


class SimpleMLP(nn.Module):
    def __init__(self, shape, classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(shape, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 50)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(50, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
    

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the batch size
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clients
    global_count = sum([len(clients_trn_data[client_name]) for client_name in client_names]) * bs
    # get the total number of data points held by a client
    local_count = len(clients_trn_data[client_name]) * bs
    return local_count / global_count


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. This is equivalent to scaled avg of the weights'''
    avg_grad = [torch.stack(layer_grad).sum(dim=0) for layer_grad in zip(*scaled_weight_list)]
    return avg_grad

# def sum_scaled_weights_admm(scaled_weight_list):
#     '''Return the sum of the listed scaled weights. This is equivalent to scaled avg of the weights'''
#     avg_grad = [torch.stack(layer_grad).sum(dim=0) for layer_grad in zip(*scaled_weight_list)]
#     return avg_grad

# def update_avg_with_delta(avg_grad, delta_x_hats,num_of_client):
#     '''Update avg_grad by adding the corresponding elements from delta_x_hats'''
#     for avg_layer, delta_layers in zip(avg_grad, zip(*delta_x_hats)):
#         sum_delta_layer = torch.stack(delta_layers).sum(dim=0)
#         avg_layer += sum_delta_layer/num_of_client
    
#     return avg_grad


def scale_model_weights(weight, scalar):
    weight_final = [scalar * w for w in weight]
    return weight_final





def create_non_iid_clients(image_list, label_list, num_clients=10, initial='clients'):
    """
    Return a dictionary with keys as clients' names and values as data shards,
    represented as tuples of images and label lists. The data distribution among
    clients is non-identically distributed (non-IID).

    Args:
        image_list (list): A list of numpy arrays of training images.
        label_list (list): A list of binarized labels for each image.
        num_clients (int): Number of federated members (clients).
        initial (str): The clients' name prefix, e.g., 'clients_1'.

    Returns:
        dict: A dictionary mapping client names to non-IID data shards.
    """

    # Create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # Randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    shards = []
    for i in range(num_clients):
        # Define a custom distribution pattern for each client (you can customize this based on your needs)
        # For example, distribute 70% of samples from class 0 and 30% from class 1 to the first client,
        # and vice versa for the second client
        if i % 2 == 0:
            class_0_fraction = 0.5
        else:
            class_0_fraction = 0.45

        # Select samples based on the defined distribution
        class_0_samples = int(len(data) * class_0_fraction)
        client_data = data[:class_0_samples] if i % 2 == 0 else data[class_0_samples:]

        # Remove selected samples from the data to avoid overlap
        data = data[class_0_samples:]

        shards.append(client_data)


    # Number of clients must equal the number of shards
    assert len(shards) == len(client_names)

    return {client_names[i]: shards[i] for i in range(len(client_names))}

# Example usage:
# Assuming image_list and label_list are your data
# client_data = create_non_iid_clients(image_list, label_list, num_clients=10, initial='client')
# client_name = 'client_1'  # Change this to the specific client name you want to access
# client_dataset = CustomDataset(*zip(*client_data[client_name]))
# client_dataloader = DataLoader(client_dataset, batch_size=32, shuffle=True)





def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
  '''
  Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
  Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    n_clients : number of clients
    classes_per_client : number of classes per client
    shuffle : True/False => True for shuffling the dataset, False otherwise
    verbose : True/False => True for printing some info, False otherwise
  Output:
    clients_split : client data into desired format
  '''
  #### constants #### 
  n_data = data.shape[0]
  n_labels = np.max(labels) + 1


  ### client distribution ####
  data_per_client = clients_rand(len(data), n_clients)
  data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]
  
  # sort for labels
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]
  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)
    
  # split data among clients
  clients_split = []
  c = 0
  for i in range(n_clients):
    client_idcs = []
        
    budget = data_per_client[i]
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      
      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      
      budget -= take
      c = (c + 1) % n_labels
      
    clients_split += [(data[client_idcs], labels[client_idcs])]

  def print_split(clients_split): 
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()
      
    if verbose:
      print_split(clients_split)
  
  clients_split = np.array(clients_split)
  
  return clients_split


def test_model(X_test, Y_test, model, comm_round):
    bce = nn.BCELoss()
    with torch.no_grad():
        logits = model(X_test)
        loss = bce(logits, Y_test)
        Y_prdt = (logits > 0.5).int()
        acc = accuracy_score(Y_test.cpu().numpy(), Y_prdt.cpu().numpy())
        F1 = f1_score(Y_test.cpu().numpy(), Y_prdt.cpu().numpy(), zero_division=1)
        precision = precision_score(Y_test.cpu().numpy(), Y_prdt.cpu().numpy(), zero_division=1)
        recall = recall_score(Y_test.cpu().numpy(), Y_prdt.cpu().numpy(), zero_division=1)

        cm = confusion_matrix(Y_test.cpu().numpy(), Y_prdt.cpu().numpy())
        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        fpr, tpr, thresholds = metrics.roc_curve(Y_test.cpu().numpy(), logits.cpu().numpy())
        auc_value = roc_auc_score(Y_test.cpu().numpy(), logits.cpu().numpy())

        print('comm_round: {} | global_acc: {:.3%} | global_loss: {} | global_f1: {} | global_precision: {} | global_recall: {} | global_auc: {}| flobal_FPR: {} '.format(
            comm_round, acc, loss.item(), F1, precision, recall, auc_value, FPR))
    return acc, loss.item(), F1, precision, recall, auc_value, FPR