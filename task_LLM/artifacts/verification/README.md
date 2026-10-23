# 当前验证产物

2026-10-15 从现有项目实际运行得到，原始训练日志与正式 checkpoint 保留。

| 文件 | 证据 |
|---|---|
| environment.json | 当前环境、上游 revision、正式 checkpoint 配置、参数量、数据规模 |
| generation.json | 正式模型和同架构随机模型固定种子生成 |
| generation_repeat.json | 同环境重复运行，与 generation.json 完全相同 |
| generation_repeat.log | 重复生成的控制台输出 |
| loss.csv / loss_curve.png | 从原始训练日志提取的 9 个评估点及曲线 |
| smoke_train.log | 独立小模型 4 步训练入口验证 |
| smoke-model/ckpt.pt | 短训练生成的新 checkpoint |
| smoke_generation.json | 加载短训练 checkpoint 后的生成验证 |
| tests.log | 行为测试结果 |

短训练不能代替完整训练或证明收敛。训练日志的上游 GradScaler 弃用警告不影响此次运行，
没有通过设置 KMP_DUPLICATE_LIB_OK 绕过 OpenMP 冲突；绘图使用独立进程。

本轮借助 AI 修改代码、补充测试、核对理论和整理实验文档。提交者仍需阅读笔记、核对源码，
并能够自行解释模型流程及结论；这些文档不能作为独立自主完成学习的证明。
