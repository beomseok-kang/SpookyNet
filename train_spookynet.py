
from spookynet import SpookyNet, SpookyNetCalculator
from ase import Atoms

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from torch.nn import MSELoss
from tqdm import tqdm
# from ema_to import ExponentialMovingAverage as EMA
from ema_pytorch import EMA
from sklearn.linear_model import LinearRegression
import torch.nn as nn

class Normalizer:
    def __init__(self, x):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = 1e-5

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

def register_bias(model, els, dataset, Zmax=87):
    
    mol_n_els = []
    labels = []
    for feat in dataset:
        Z = feat.Z
        labels.append(feat.label.item())
        n_els = []
        for el in els:
            n_els.append(np.sum((Z == el).numpy()))
        mol_n_els.append(n_els)

    X = np.array(mol_n_els)
    y = np.array(labels)
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(X, y)
    
    element_bias = nn.Parameter(torch.zeros(Zmax, 2), requires_grad=False)
    element_bias.data[els, 0] = torch.tensor(linreg.coef_, dtype=torch.float32)
    model.element_bias = element_bias

model = SpookyNet(
    num_features=128,
    num_modules=6,
    Zmax=87,
).to(torch.float32)
model.reset_parameters()

f = h5py.File("/home/beom/orbnet/data/qmspin_caltech/qmspin_all.hdf5", "r")

bohr_to_angstrom = 0.529177210903
Ha_to_eV = 27.211386024367243

def _get_neighborlists(nats):
    idx = torch.arange(nats, dtype=torch.int64)
    idx_i = idx.view(-1, 1).expand(-1, nats).reshape(-1)
    idx_j = idx.view(1, -1).expand(nats, -1).reshape(-1)
    # exclude self-interactions
    idx_i_res = idx_i[idx_i != idx_j]
    idx_j_res = idx_j[idx_i != idx_j]
    return idx_i_res, idx_j_res

def get_feats(numbers, positions, magmom):
    # positions in angstrom please
    dtype = torch.float32
    nats = len(numbers)
    idx_i, idx_j = _get_neighborlists(nats)
    
    Z = torch.tensor(numbers, dtype=torch.int64)
    Q = torch.tensor([0], dtype=dtype)
    S = torch.tensor([magmom], dtype=dtype)
    R = torch.tensor(positions, dtype=dtype, requires_grad=True)
    
    return Z, Q, S, R, idx_i, idx_j
    
class MolFeature:
    def __init__(self, Z, Q, S, R, idx_i, idx_j, label, batch_seg=None):
        self.Z = Z
        self.Q = Q
        self.S = S
        self.R = R
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.label = label
        self.batch_seg = batch_seg
    
    def to(self, device):
        dev_feat = MolFeature(
            self.Z.to(device),
            self.Q.to(device),
            self.S.to(device),
            self.R.to(device),
            self.idx_i.to(device),
            self.idx_j.to(device),
            self.label.to(device),
            self.batch_seg.to(device) if self.batch_seg is not None else None
        )
        return dev_feat
    
    def get_args(self):
        return [
            self.Z,
            self.Q,
            self.S,
            self.R,
            self.idx_i,
            self.idx_j,
        ], self.batch_seg
    
    def get_label(self):
        return self.label

    @staticmethod
    def batch(features):
        # features : List["MolFeature"]
        nmol = len(features)
        nats = [len(f.Z) for f in features]
        head = 0
        for i in range(nmol):
            nat = nats[i]
            batch_seg_mol = torch.tensor(
                [i for _ in range(nat)],
                dtype=torch.int64
            )
            if i == 0:
                Z = features[i].Z
                Q = features[i].Q
                S = features[i].S
                R = features[i].R
                idx_i = features[i].idx_i
                idx_j = features[i].idx_j
                label = features[i].label
                batch_seg = batch_seg_mol
            else:
                Z = torch.cat([Z, features[i].Z])
                Q = torch.cat([Q, features[i].Q])
                S = torch.cat([S, features[i].S])
                R = torch.cat([R, features[i].R])
                idx_i = torch.cat([idx_i, features[i].idx_i + head])
                idx_j = torch.cat([idx_j, features[i].idx_j + head])
                label = torch.cat([label, features[i].label])
                batch_seg = torch.cat([batch_seg, batch_seg_mol])
            head += nat
        return MolFeature(Z, Q, S, R, idx_i, idx_j, label, batch_seg=batch_seg)
        

class BasicDataset(Dataset):
    def __init__(self, features, name="train"):
        self.features = features
        self.name = name

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def batcher(self, batch):
        features = batch
        batch_feature_sets = MolFeature.batch(features)
        return batch_feature_sets

Ha_to_eV = 27.21139

train_dataset = BasicDataset([], name="train")
test_dataset = BasicDataset([], name="test")
for mode in ["train", "test"]:
    features = []
    if mode == "test":
        dataset = test_dataset
    else:
        dataset = train_dataset
    for mol in tqdm(list(f[mode].keys())):
    # for mol in tqdm(f[mode].keys()):
        for geo in f[mode][mol].keys():
            data_mol = f[mode][mol][geo]
            atomic_numbers = data_mol["atomic_numbers"][()]
            geometry = data_mol["geometry_bohr"][()] * bohr_to_angstrom
            net_spin = data_mol["net_spin"][()]
            label = data_mol["mrci_energy_Ha"][()] * Ha_to_eV
            Z, Q, S, R, idx_i, idx_j = get_feats(atomic_numbers, geometry, net_spin)
            mol_feat = MolFeature(Z, Q, S, R, idx_i, idx_j, torch.tensor([label], dtype=torch.float32))
            features.append(mol_feat)
    dataset.features = features
    
test_feats = test_dataset.features
np.random.shuffle(test_feats)
val_feats = test_feats[:1000]
test_feats = test_feats[1000:]
test_dataset = BasicDataset(test_feats, name="test")
val_dataset = BasicDataset(val_feats, name="val")

register_bias(model, [1, 6, 7, 8, 9], train_dataset, Zmax=87)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ema = EMA(model, beta=0.999)
ema.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold=0)

num_epochs = 10
batch_size = 100
# batch_size = 10
validation_frequency = 1000  # Evaluate every 1k steps
validation_loss_min = 1e99

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_dataset.batcher
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=val_dataset.batcher
)

def myloss(y, pred, N):
    return torch.norm(y - pred) / np.sqrt(N)

loss_fn = MSELoss(reduction="sum")
mae_fn = torch.nn.L1Loss(reduction="sum")

training_step = 0
epoch = 0
model.to(device)
break_flag = False
while True:
    train_loss = 0
    epoch += 1
    model.train()
    for feat in tqdm(train_loader):  # Replace train_loader with your DataLoader
        
        feat = feat.to(device)
        args, batch_seg = feat.get_args()
        label = feat.get_label()
        curr_batch_size = len(label)
        output = model(
            *args,
            use_forces=False,
            use_dipole=False,
            num_batch=curr_batch_size,
            batch_seg=batch_seg
        )
        # loss = loss_fn(output[0], label).sqrt()  # RMSE loss
        loss = myloss(output[0], label, curr_batch_size)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_step += 1
        train_loss += loss.item()
        # Update EMA model
        ema.update()

        # Evaluate on validation set every 1k steps
        if training_step % validation_frequency == 0:
            model.eval()
            total_val_loss = 0
            total_val_mae = 0
            val_size = len(val_loader.dataset)
            with torch.no_grad():
                for feat in val_loader:  # Replace validation_loader with your DataLoader
                    feat = feat.to(device)
                    args, batch_seg = feat.get_args()
                    label = feat.get_label()
                    curr_batch_size = len(label)
                    val_output = model(
                        *args,
                        use_forces=False,
                        use_dipole=False,
                        num_batch=curr_batch_size,
                        batch_seg=batch_seg
                    )
                    val_loss = myloss(val_output[0], label, curr_batch_size)
                    val_mae = mae_fn(val_output[0], label)
                    total_val_loss += val_loss.item()
                    total_val_mae += val_mae.item()

            mean_val_loss = total_val_loss / len(val_loader)
            mean_val_mae = total_val_mae / val_size
            print(f'Epoch {epoch}, Step {training_step}: Validation Loss = {mean_val_loss}, MAE = {mean_val_mae}, LR = {optimizer.param_groups[0]["lr"]}')

            # Decay learning rate if validation loss does not decrease
            if mean_val_loss < validation_loss_min:
                validation_loss_min = mean_val_loss
                model.cpu()
                model.save('best_model.pth')
                model.to(device)
            
            model.train()
            scheduler.step(mean_val_loss)
    
        if optimizer.param_groups[0]['lr'] < 1e-5:
            break_flag = True
            break
    if break_flag:
        break
    
    print(f'Epoch {epoch}, Training Loss = {train_loss / len(train_loader)}')
    
# TODO: Run test set
                
print("All process finished.")








