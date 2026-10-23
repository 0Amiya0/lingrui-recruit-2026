# 模型注册与入口扩展

`models/__init__.py` 的 build_model 注册 baseline/unet/derainnet/mymodel；
main.get_args 中 choices 与之保持一致。新增 DerainNet.py 和 MyModel.py 各自定义网络类，统一输入输出 N×3×H×W。
训练、推理、导出、Flask 均通过同一构建函数恢复模型，checkpoint 记录模型名称，避免手动选错结构。

模型参数检查只约束 MyModel<4,000,000 个参数，4M 在这里解释为参数数量，不是权重文件 4MB。
若以 float32 保存，4M 参数权重本身约 16MB，训练 checkpoint 还会包含优化器状态。
题目有不同口径时应提前与负责人确认，不把模型文件字节数和参数数混用。

状态：注册与代码已提供，未执行。
