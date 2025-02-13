import torch
from fairseq.models.av_hubert import AVHubertModel
state = torch.load("/workspace/AVTSR/avtnet_deltalm/utils/base_noise_pt_noise_ft_433h.pt", map_location=torch.device("cpu"))
