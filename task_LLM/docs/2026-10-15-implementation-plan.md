# LLM implementation plan

**Goal:** 完善三份笔记与真实可复现的 nanoGPT 实验链路。
**Architecture:** 保留上游模型和训练入口，在外层提供生成、日志解析、环境采集与验证工具。
**Tech Stack:** Python、PyTorch、NumPy、matplotlib、unittest。

按当前会话顺序执行，沿用已批准设计；不创建新 Git 仓库，不提交已有未跟踪文件。

1. 先写失败测试：日志解析函数 `parse_log(text)`、可复现采样 `generate_text(model, ids, ...)`、非法输入校验、checkpoint 配置恢复。
2. 改造 `generate_demo.py`：main 入口、argparse、按 checkpoint 构建随机对照、固定采样 seed、输出 JSON。
3. 改造 `plot_loss.py`：main 入口、CSV、科学计数法解析、空日志报错。
4. 新增 `scripts/verify_environment.py`，保存环境、数据、checkpoint、Git revision 摘要。
5. 测试 nanoGPT 未来输入不影响此前 logits、训练与推理形状、权重共享、目标偏移。
6. 运行 `python -m unittest discover -s tests -v`；运行既有 checkpoint 生成；日志导出曲线；独立目录短训练并加载生成。
7. 修正三份笔记，更新 README、依赖与当前验证结果。核对 roadmap 后报告已完成项及限制。
