# unlearn_kd/utils/seed.py
import os, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn 관련
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 환경 변수
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed fixed to {seed}]")
