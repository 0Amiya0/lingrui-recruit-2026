# ============================================================
# run_all.ps1 — workshop3 全流程实验脚本（Windows PowerShell）
#
# 说明：
#   设备由 -Device 决定（默认 auto：有 CUDA 用 GPU，否则 CPU）。
#   首次运行需先安装依赖（见 README.md）。
#   主实验矩阵默认跑多个种子（报告 mean±std）；消融与规模敏感性用首个种子。
#
# 用法（在 PowerShell 中）：
#   .\run_all.ps1                      # 全流程（3 种子）
#   .\run_all.ps1 -K 10 -Seeds "0,1,2" -Loss gce
#   .\run_all.ps1 -Seeds "0"           # 单种子（快速验证）
#   .\run_all.ps1 -Rates "0.0,0.2,0.4" # 只跑部分噪声率
#   .\run_all.ps1 -Device cuda:0        # 显式指定 GPU（多卡机器）
#   .\run_all.ps1 -Batch 64             # 显存小（2GB 级）时降低特征提取 batch
#
# 若提示禁止运行脚本，先执行：
#   Set-ExecutionPolicy -Scope Process Bypass
# ============================================================
param(
    [int]$K = 10,                     # few-shot 每类样本数
    [string]$Seeds = "0,1,2",         # 随机种子（逗号分隔，主矩阵跑多种子）
    [string]$Loss = "gce",            # 鲁棒损失: ce / gce / sce
    [string]$Rates = "0.0,0.1,0.2,0.4",  # 标签噪声率
    [string]$Device = "auto",         # 设备: auto / cuda / cuda:0 / cpu
    [int]$Batch = 256,                # 特征提取 batch_size（显存小降到 64/128）
    [string]$Python = "python"        # Python 解释器
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 修复 anaconda(MKL) 与 torch 的 OpenMP 冲突，并保证中文输出正常
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Step {
    param([string]$Name, [string[]]$CmdArgs)
    Write-Host ""
    Write-Host "===== $Name =====" -ForegroundColor Cyan
    & $Python @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] $Name 退出码 $LASTEXITCODE，终止。" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "===== $Name 完成 =====" -ForegroundColor Green
}

Write-Host "workshop3 全流程实验 | K=$K Seeds=$Seeds Loss=$Loss Rates=$Rates Device=$Device Batch=$Batch" -ForegroundColor Yellow

$seedList = $Seeds.Split(",")
$rateList = $Rates.Split(",")
$firstSeed = $seedList[0]

# ---------- 阶段 1+2：每个种子 × 每个噪声率：数据准备 + baseline + 方法 ----------
foreach ($s in $seedList) {
    foreach ($r in $rateList) {
        Invoke-Step "数据准备 (noise=$r, seed=$s)" @(
            "scripts\prepare_data.py", "--k-shot", "$K", "--noise-rate", "$r", "--seed", "$s",
            "--device", "$Device", "--batch-size", "$Batch"
        )
    }
    foreach ($r in $rateList) {
        Invoke-Step "Baseline (noise=$r, seed=$s)" @(
            "scripts\run_baseline.py", "--k-shot", "$K", "--noise-rate", "$r", "--seed", "$s",
            "--device", "$Device"
        )
        Invoke-Step "方法 (noise=$r, seed=$s)" @(
            "scripts\run_method.py", "--k-shot", "$K", "--noise-rate", "$r",
            "--seed", "$s", "--loss", "$Loss", "--device", "$Device"
        )
    }
}

# ---------- 阶段 3：消融实验（首个种子 @ 最高噪声率，扫超参） ----------
$maxRate = ($rateList | ForEach-Object { [double]$_ } | Measure-Object -Maximum).Maximum
Invoke-Step "消融实验 (noise=$maxRate, seed=$firstSeed)" @(
    "scripts\run_ablation.py", "--k-shot", "$K", "--noise-rate", "$maxRate", "--seed", "$firstSeed",
    "--device", "$Device"
)

# ---------- 阶段 4：few-shot 规模敏感性（K ∈ {1,5,20} @ 最高噪声率，首个种子） ----------
foreach ($kk in @(1, 5, 20)) {
    if ($kk -eq $K) { continue }
    Invoke-Step "few-shot 规模 K=$kk (noise=$maxRate, seed=$firstSeed)" @(
        "scripts\prepare_data.py", "--k-shot", "$kk", "--noise-rate", "$maxRate", "--seed", "$firstSeed",
        "--device", "$Device", "--batch-size", "$Batch"
    )
    Invoke-Step "Baseline K=$kk" @(
        "scripts\run_baseline.py", "--k-shot", "$kk", "--noise-rate", "$maxRate", "--seed", "$firstSeed",
        "--device", "$Device"
    )
    Invoke-Step "方法 K=$kk" @(
        "scripts\run_method.py", "--k-shot", "$kk", "--noise-rate", "$maxRate",
        "--seed", "$firstSeed", "--loss", "$Loss", "--device", "$Device"
    )
}

# ---------- 阶段 5：指标自检 + 多种子聚合 + 画图 ----------
Invoke-Step "指标自检" @("scripts\test_metrics.py")
Invoke-Step "多种子聚合 mean±std" @("scripts\aggregate_seeds.py")
Invoke-Step "画主结果图" @("scripts\plot_results.py")

# ---------- 阶段 6：错误分析（首个种子 @ 最高噪声率） ----------
Invoke-Step "错误分析 (noise=$maxRate, seed=$firstSeed)" @(
    "scripts\error_analysis.py", "--k-shot", "$K", "--noise-rate", "$maxRate", "--seed", "$firstSeed",
    "--device", "$Device"
)

Write-Host ""
Write-Host "全流程完成。结果见 results/ 目录。" -ForegroundColor Green
Write-Host "  - results/baseline_summary.csv / method_summary.csv（逐种子）"
Write-Host "  - results/summary_meanstd.csv（多种子 mean±std）"
Write-Host "  - results/ablation_summary.csv / error_analysis.json"
Write-Host "  - results/figs/*.png（含 risk_coverage.png）"
