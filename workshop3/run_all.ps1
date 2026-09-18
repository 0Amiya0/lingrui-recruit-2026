# ============================================================
# run_all.ps1 — workshop3 全流程实验脚本（Windows PowerShell）
#
# 说明：
#   本机无 GPU，所有实验均用 CPU 版 PyTorch 在本脚本中编排执行。
#   首次运行需先安装依赖（见 README.md）。
#
# 用法（在 PowerShell 中）：
#   .\run_all.ps1                      # 全流程
#   .\run_all.ps1 -K 10 -Seed 0 -Loss gce
#   .\run_all.ps1 -Rates "0.0,0.2,0.4" # 只跑部分噪声率
#
# 若提示禁止运行脚本，先执行：
#   Set-ExecutionPolicy -Scope Process Bypass
# ============================================================
param(
    [int]$K = 10,                  # few-shot 每类样本数
    [int]$Seed = 0,                # 随机种子
    [string]$Loss = "gce",         # 鲁棒损失: ce / gce / sce
    [string]$Rates = "0.0,0.1,0.2,0.4",  # 标签噪声率
    [string]$Python = "python"     # Python 解释器
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

Write-Host "workshop3 全流程实验 | K=$K Seed=$Seed Loss=$Loss Rates=$Rates" -ForegroundColor Yellow

$rateList = $Rates.Split(",")

# ---------- 阶段 1：数据准备（每个噪声率各一份 support 特征 + 噪声 mask） ----------
foreach ($r in $rateList) {
    Invoke-Step "数据准备 (noise=$r)" @(
        "scripts\prepare_data.py", "--k-shot", "$K", "--noise-rate", "$r", "--seed", "$Seed"
    )
}

# ---------- 阶段 2：主实验矩阵（baseline + 方法，每个噪声率） ----------
foreach ($r in $rateList) {
    Invoke-Step "Baseline (noise=$r)" @(
        "scripts\run_baseline.py", "--k-shot", "$K", "--noise-rate", "$r", "--seed", "$Seed"
    )
    Invoke-Step "方法 (noise=$r)" @(
        "scripts\run_method.py", "--k-shot", "$K", "--noise-rate", "$r",
        "--seed", "$Seed", "--loss", "$Loss"
    )
}

# ---------- 阶段 3：消融实验（在最高噪声率下扫超参） ----------
$maxRate = ($rateList | ForEach-Object { [double]$_ } | Measure-Object -Maximum).Maximum
Invoke-Step "消融实验 (noise=$maxRate)" @(
    "scripts\run_ablation.py", "--k-shot", "$K", "--noise-rate", "$maxRate", "--seed", "$Seed"
)

# ---------- 阶段 4：few-shot 规模敏感性（K ∈ {5,20} @ 最高噪声率，K=主值已跑） ----------
foreach ($kk in @(5, 20)) {
    if ($kk -eq $K) { continue }
    Invoke-Step "few-shot 规模 K=$kk (noise=$maxRate)" @(
        "scripts\prepare_data.py", "--k-shot", "$kk", "--noise-rate", "$maxRate", "--seed", "$Seed"
    )
    Invoke-Step "Baseline K=$kk" @(
        "scripts\run_baseline.py", "--k-shot", "$kk", "--noise-rate", "$maxRate", "--seed", "$Seed"
    )
    Invoke-Step "方法 K=$kk" @(
        "scripts\run_method.py", "--k-shot", "$kk", "--noise-rate", "$maxRate",
        "--seed", "$Seed", "--loss", "$Loss"
    )
}

# ---------- 阶段 5：指标自检 + 画图 ----------
Invoke-Step "指标自检" @("scripts\test_metrics.py")
Invoke-Step "画主结果图" @("scripts\plot_results.py")

# ---------- 阶段 6：错误分析 ----------
Invoke-Step "错误分析 (noise=$maxRate)" @(
    "scripts\error_analysis.py", "--k-shot", "$K", "--noise-rate", "$maxRate", "--seed", "$Seed"
)

Write-Host ""
Write-Host "全流程完成。结果见 results/ 目录。" -ForegroundColor Green
Write-Host "  - results/baseline_summary.csv / method_summary.csv / ablation_summary.csv"
Write-Host "  - results/error_analysis.json"
Write-Host "  - results/figs/*.png"
