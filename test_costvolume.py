    import torch

from stp3.datas.NuscenesData import FuturePredictionDataset
from nuscenes.nuscenes import NuScenes
from stp3.trainer import TrainingModule

from stp3.models.stp3 import STP3
from stp3.config import get_parser, get_cfg

nusc = NuScenes(version='v1.0-mini', dataroot="/mnt/e/datasets/nuscenes/mini", verbose=False)

cfg = get_cfg()

traindata = FuturePredictionDataset(nusc, 0, cfg)
valdata = FuturePredictionDataset(nusc, 1, cfg)

trainloader = torch.utils.data.DataLoader(traindata, batch_size=512, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
# model = STP3(cfg)
# model_tm = TrainingModule(cfg.convert_to_dict())

for i, batch in enumerate(trainloader):
    
    break