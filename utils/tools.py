import numpy as np
import pandas as pd
import torch.utils.data as util_data
from torchvision import transforms
import torch
import librosa
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json


def config_dataset(config):
    
    config["dataset"] == "dcase_task2"
    config["topK"] = 100       # 9473
    config["n_class"] = 41     # n_class in train set or database

    config["data_path"] = "/hdd_storage/data/sagard/" + config["dataset"] + "/"
    
    config["data"] = {
        "train_set": {"list_path": "/hdd_storage/data/sagard/" + config["dataset"] + "/train.csv", "batch_size": config["batch_size"]},
        "database": {"list_path": "/hdd_storage/data/sagard/" + config["dataset"] + "/train.csv", "batch_size": config["batch_size"]},
        "test": {"list_path": "/hdd_storage/data/sagard/" + config["dataset"] + "/test.csv", "batch_size": config["batch_size"]}}
    return config
    

class ContrastiveAudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_list):
        self.audio_list = AudioList(audio_list)
        self.indexes_by_label = self._index_by_label()

    def _index_by_label(self):
        indexes_by_label = {}
        for index in range(len(self.audio_list)):
            _, label, _ = self.audio_list[index]
            label = label.argmax().item()  # Get the label as int
            if label not in indexes_by_label:
                indexes_by_label[label] = []
            indexes_by_label[label].append(index)
        return indexes_by_label

    def __getitem__(self, idx):
        anchor_feature, anchor_label, _ = self.audio_list[idx]
        anchor_label_int = anchor_label.argmax().item()

        # Select a positive sample
        positive_idx = idx
        while positive_idx == idx:  # Ensure different sample
            positive_idx = random.choice(self.indexes_by_label[anchor_label_int])
        positive_feature, _, _ = self.audio_list[positive_idx]

        # Select a negative sample
        negative_label_int = random.choice(list(set(range(41)) - {anchor_label_int}))
        negative_idx = random.choice(self.indexes_by_label[negative_label_int])
        negative_feature, _, _ = self.audio_list[negative_idx]

        return anchor_feature, positive_feature, negative_feature, anchor_label

    def __len__(self):
        return len(self.audio_list)

   
class AudioList(object):

    def __init__(self, audio_list):
        self.csv = pd.read_csv(audio_list).iloc[:,1:]

    def __getitem__(self, index):
        path = '/hdd_storage/data/sagard/dcase_task2/'+ self.csv.iloc[index,0]
        label =  int(self.csv.iloc[index,1])
        
        zeros = torch.zeros((41,))      #Change number of classes here too
        zeros[label] = 1
        
        y, sr = librosa.load(path, sr=None)
        audio_padded = librosa.util.fix_length(y, size=220500)   # 10 sec

        hop_length = 512*2
        S = librosa.feature.melspectrogram(y=audio_padded, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=128)
        logS = librosa.power_to_db(abs(S))
        
        feature = torch.Tensor(logS)
        
        # feature, target = torch.load(path),zeros
        feature_3d = torch.stack((feature, feature, feature), 0)
        return feature_3d, zeros, index

    def __len__(self):
        return len(self.csv)


def get_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]


    dsets["train_set"] = ContrastiveAudioDataset(data_config["train_set"]["list_path"])
    dset_loaders["train_set"] = DataLoader(dsets["train_set"],
                                           batch_size=data_config["train_set"]["batch_size"],
                                           shuffle=True,  # Shuffle for training
                                           num_workers=data_config["train_set"].get("num_workers", 4),
                                           pin_memory=True)  # Consider using pin_memory for faster data transfer to CUDA devices


    for data_set in ["test", "database"]:
        dsets[data_set] = AudioList(data_config[data_set]["list_path"])
        dset_loaders[data_set] = DataLoader(dsets[data_set],
                                            batch_size=data_config[data_set]["batch_size"],
                                            shuffle=False,  # No need to shuffle test and database sets
                                            num_workers=data_config[data_set].get("num_workers", 4),
                                            pin_memory=True)





    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])




def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall


def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
    device = config["device"]
    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    if "pr_curve_path" not in  config:
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
    else:
        # need more memory
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                     trn_binary.numpy(), trn_label.numpy(),
                                                     config["topK"])
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]

        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }


    if mAP > Best_mAP:
        Best_mAP = mAP
        # if "save_path" in config:
        #     save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
        #     os.makedirs(save_path, exist_ok=True)
        #     print("save in ", save_path)
        #     np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
        #     np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
        #     np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
        #     np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
        #     torch.save(net.state_dict(), os.path.join(save_path, "model.pt"))
    print("\n")
    print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print("\n")
    print(config)
    return Best_mAP
