# 上游来源

出题人提供 URL： https://github.com/Aster-1024/Image-Deraining-Recruitment-Task
访问时跳转到： https://github.com/Lingrui-Studio/Image-Deraining-Recruitment-Task

2026-10-15 读取 main 分支公开源码。`models/unet.py` 保留其模型结构与 forward，
仅统一换行符；原始 LICENSE 与 README 保留于本目录。上游采用 MIT License，版权标注见 LICENSE。
其他模块根据题目重新组织实现，保留相同模型名称与命令行学习接口。

主要有意差异：同步配对增强、不缩小测试图、train 内验证划分、结构实验使用 MSE、
CUDA/AMP、恢复状态与独立输出目录。因测试协议和划分不同，不能与上游 resize128/test选模型的数值直接横比。
这里没有克隆或建立 Git 工作树，没有替用户推送或发布上游 Fork。
