"""实验配置：dataclass + JSON 加载 + 命令行覆盖。"""
from dataclasses import dataclass, fields, asdict
import argparse
import json
from pathlib import Path


@dataclass
class Config:
    # ---- 数据 ----
    dataset: str = "cifar10"
    n_way: int = 10            # 类别数
    k_shot: int = 10           # 每类 support 样本数（few-shot 规模）
    noise_rate: float = 0.0    # 对称标签噪声比例 {0.0, 0.1, 0.2, 0.4}
    noise_type: str = "symmetric"
    seed: int = 0
    # ---- 模型 ----
    backbone: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    normalize: bool = True     # 特征是否 L2 归一化
    batch_size: int = 256
    device: str = "auto"       # "auto" -> 有 CUDA 用 CUDA，否则 CPU
    # ---- 方法（Step 3+） ----
    loss: str = "gce"          # ce / gce / sce
    gce_q: float = 0.7         # GCE 参数
    sel_beta: float = 1.0      # 筛选阈值：mean + beta*std
    sel_iters: int = 5         # 筛选迭代轮数
    # ---- 路径 ----
    data_dir: str = "data"
    out_dir: str = "results"

    @classmethod
    def from_json(cls, path):
        cfg = cls()
        if path is not None and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_args(cls, argv=None):
        parser = argparse.ArgumentParser(description="workshop3 实验")
        parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
        for f in fields(cls):
            flag = f"--{f.name.replace('_', '-')}"
            if f.type is bool:
                parser.add_argument(flag, type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None)
            elif f.type is int:
                parser.add_argument(flag, type=int, default=None)
            elif f.type is float:
                parser.add_argument(flag, type=float, default=None)
            else:
                parser.add_argument(flag, type=str, default=None)
        args = parser.parse_args(argv)

        cfg = cls.from_json(args.config)
        for f in fields(cls):
            v = getattr(args, f.name)
            if v is not None:
                setattr(cfg, f.name, v)
        return cfg

    def to_dict(self):
        return asdict(self)
