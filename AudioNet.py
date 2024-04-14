from utils.tools import *
from utils.loss import *
from network import *

import torch
import torch.optim as optim
import time
from scipy.spatial.distance import hamming

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {
        "alpha": 0.1,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "step_continuation": 20,
        "batch_size": 64,
        "net":ResNet,
        "dataset": "dcase_task2",
        "margin": 1,
        "epoch": 150,
        "test_map": 50,
        "save_path": "save/",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:2"),
        "bit_list": [16,32,64,128],
    }
    config = config_dataset(config)
    return config

def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **config["optimizer"]["optim_params"])
    similaritylayer = CosineSimilarity()
    closs = ContrastiveLoss(margin=config['margin'], similaritylayer=similaritylayer)
    wloss = WeightedLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):
        net.train()
        train_loss = 0

        for input_feat, labels in train_loader:
            input_feat = input_feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            hcode, embeddings = net(input_feat)

            emb_i = embeddings[:len(embeddings)//2]
            emb_j = embeddings[len(embeddings)//2:]

            hcode_i = hcode[:len(hcode)//2]
            hcode_j = hcode[len(hcode)//2:]

            d_eud = torch.cdist(emb_i, emb_j, p=2)  

            weighted_loss = wloss(hcode_i, hcode_j, labels, config)

            d_tot = d_eud + weighted_loss
            contrastive_loss = closs(d_tot, labels)

            loss = 0.5 * contrastive_loss + 0.5 * weighted_loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print(f"{config['info']}[{epoch + 1}/{config['epoch']}][{current_time}] bit:{bit}, dataset:{config['dataset']}, training.... loss:{train_loss:.3f}")

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            
    # save and load entire model

    FILE = "/model/model.pth"
    torch.save(net, FILE)
    
if __name__ == "__main__":
    config = get_config()
    print(config)
    
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/audionet_{config['dataset']}_{bit}.json"
        train_val(config, bit)
