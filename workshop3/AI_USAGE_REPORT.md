# AI 使用报告

> Workshop 题目三：标签噪声下的小样本图像分类（Few-shot Image Classification under Label Noise）

- **项目目录**：`Lingrui/workshop3/`
- **提交人**：________（姓名 / 学号）
- **完成时间**：2026-09
- **报告目的**：按课程 / 工作坊要求，如实说明在本项目中生成式 AI 工具的使用情况、人机分工、验证方式与诚信边界。

---

## 1. 使用的 AI 工具

| 工具 | 厂商 | 使用方式 |
|---|---|---|
| Claude Code（Claude 系列模型） | Anthropic | 交互式命令行助手，用于代码生成、脚本编写、论文起草、文档撰写与排错 |

- 本项目**未使用**其它生成式 AI 工具（如未使用请保留；如有请在此补充并说明用途）。
- 模型 / 版本如有具体要求，请在此补充（例如「Claude Sonnet 4.x」等）。

---

## 2. 使用范围：AI 在各个环节做了什么

| 环节 | AI 的贡献 | 人工的贡献 |
|---|---|---|
| **选题与问题定义** | 帮助梳理 `task.md` 的硬性要求（对称噪声、筛选/重加权/鲁棒损失类方法） | 确定题目、阅读并理解任务要求 |
| **研究路线与方法设计** | 提出「距离驱动样本筛选 + 鲁棒损失（GCE/SCE）」方案，并分析其与「小损失技巧」的对应关系 | 拍板采用「低成本诚实化」路线；决定加入强基线、多种子、AUROC、noisy-test 等升级项 |
| **代码实现**（`src/`、`scripts/`） | 生成/修改 backbone、baseline、method、metrics、config、dataset 等模块及全部实验脚本 | 审阅代码、提出修改意见（如「先实现但不跑」）、确认需求变更（CPU → GPU 支持） |
| **实验编排**（`run_all.ps1`、`run_experiments.py`） | 编写 Windows PowerShell 全流程脚本与跨平台 Python 编排器 | 在本机运行实验，产出 `results/` 全部数据与图表 |
| **结果回填与论文**（`paper/short_paper.md`、`.tex`） | 起草英文 short paper 与 ICLR 2023 LaTeX 版；根据 `results/` 回填表格数值、嵌入图片、更新结论 | 逐项核对数字、决定论文定位与诚实性表述（如「可识别性套利」限制） |
| **文档**（README、本报告） | 撰写/改写 README 与 AI 使用报告 | 提供使用场景（GPU 机器、显存、跨平台）等需求 |

---

## 3. 人机分工总结

- **AI 负责**：代码骨架与实现、实验脚本、论文初稿、图表代码、README 与报告等「生成型」工作。
- **人工负责**：
  1. 研究问题与方法的最终决策；
  2. 实验的实际运行（AI 仅编写脚本，未代跑实验）；
  3. 结果的核对、论文结论的最终把关；
  4. 本报告所述内容的确认与签署。
- 本项目所有论文结论、代码正确性与学术责任，由**作者本人**承担。

---

## 4. 验证与质量保障

AI 生成内容经过以下验证，而非直接采用：

1. **代码语法检查**：对全部 `.py` 文件执行 `python -m py_compile`，通过。
2. **导入冒烟测试**：对全部实验脚本执行 import 自检，无导入错误。
3. **指标单元测试**：`scripts/test_metrics.py` 对 AUROC、risk-coverage（AURC）做 sanity check，通过。
4. **PowerShell 脚本校验**：对 `run_all.ps1` 做 AST 解析检查（并修复了 UTF-8 BOM 导致的 Windows PowerShell 5.1 解析问题），通过。
5. **论文数字核对**：论文中的表格数值逐项对照 `results/summary_meanstd.csv`、`results/ablation_summary.csv`、`results/error_analysis.json` 回填，并与原始逐种子 CSV 交叉验证。
6. **人工复核**：________（请填写你本人最终通读论文、确认结论的说明）。

---

## 5. 参考文献真实性说明

论文所引文献（Zhang & Sabuncu 2018「GCE」、Wang et al. 2019「SCE」、Snell et al. 2017「Prototypical Networks」、Han et al. 2018「Co-teaching」、Li et al. 2020「DivideMix」）均为真实存在的学术论文，由 AI 依常见格式给出、由作者确认。

---

## 6. 诚信声明

- 本项目允许按课程要求在 AI 辅助下完成，且已按要求如实披露。
- AI 仅作为工具参与代码、脚本、论文初稿的生成；研究问题、方法取舍、实验执行与结论把关由作者完成。
- 如发现 AI 生成内容存在事实性错误，作者已/将在提交前修正并对其负责。

**签名**：____________　**日期**：____________
