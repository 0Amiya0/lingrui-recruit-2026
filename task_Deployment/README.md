# task_Deployment：交互、部署与接口调用

用 Flask 搭建一个 API 服务，接收后端同学发来的 POST 请求（表单数据），
完成四则运算后以 JSON 返回结果。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `app.py` | Flask 服务：解析请求 → 校验 → 调用计算程序 → 构造 JSON 响应 |
| `calculator.py` | 本地的另一个 py 程序，只负责四则运算 |

## 运行

```bash
pip install flask      # 若未安装
python app.py          # 在本目录下运行
```

服务默认监听 `http://127.0.0.1:5000`；需要被其他机器访问时显式设置 `--host 0.0.0.0`。

## 接口

- 方法：`POST`
- 路径：`/` 或 `/calculate`（两者等价）
- 请求体：`form-data`，包含三个键：

| 键 | 说明 |
| --- | --- |
| `Operation` | `sum` / `multiply` / `subtract` / `divide` |
| `Data_A` | 操作数 A |
| `Data_B` | 操作数 B |

- 响应：`{"Result": 结果}`，状态码 `200`
- 错误：`{"error": "..."}`，状态码 `400`（缺字段 / 非数字 / 非法操作 / 除零）

## Apifox 测试

1. 新建请求，方法选 `POST`，URL 填 `http://127.0.0.1:5000/calculate`
2. Body 选 `form-data`，添加三行（注意大小写）：
   - `Operation` = `sum`
   - `Data_A` = `3`
   - `Data_B` = `4`
3. 发送，返回 `{"Result": 7}`

## 本轮交付状态

只完成代码和笔记，按用户要求未在本地安装依赖、运行测试、启动服务或推理。
没有生成测试通过记录、Apifox 截图或云部署结果。测试程序位于 `tests/test_api.py`。

## 环境与 CUDA

建议在目标设备使用 Python 3.10–3.12 新建虚拟环境。不要把本机 CPU 版环境直接复制到游戏本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

AutoDL Linux 对应激活命令为 `source .venv/bin/activate`；也可使用 AutoDL 已有的 PyTorch 环境。
按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择与显卡驱动兼容的 torch/torchvision 组合。
先用 `nvidia-smi` 查看 GPU 和驱动，再用 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 自检。
CUDA runtime 随 PyTorch 包提供，通常无需另装完整 Toolkit；不要同时混装 CPU/CUDA 两套 torch。

## 运行推理服务

四则运算不需要 torch。预训练分类模式首次启动会下载 ImageNet ResNet-18 权重，需联网或提前准备 torch 缓存：

```powershell
python app.py --model classifier --device cuda:0
```

去雨模型使用相邻 `task_Deraining` 训练产生的 checkpoint，不接受上游缺少模型元信息的旧权重：

```powershell
python app.py --model derainer --device cuda:0 --checkpoint ../task_Deraining/runs/baseline/best.pt
```

`auto` 自动选 CUDA（可用时），`cpu` 强制 CPU，`cuda:0` 强制第一块 GPU；显式 CUDA 不可用会报错，防止无意退回 CPU。
每个服务进程启动加载一次模型，后续请求复用。请求体上限 8 MiB、图像上限 400 万像素，仅接受 JPEG/PNG/WEBP。
去雨输出为 JSON 中的 `image_base64` PNG，可由客户端解码保存。

需要 API key 时在启动前设置环境变量。Windows PowerShell：`$env:API_KEY="自行设置的值"`；Linux：`export API_KEY="自行设置的值"`。
请求头携带 `X-API-Key`。未配置 key 时接口不要求认证；`/health` 始终可以访问，返回 `model_ready`。

## Apifox / curl 联调

将 `openapi.yaml` 作为 OpenAPI 3 文档导入 Apifox，设置服务地址，必要时设置 `X-API-Key`。
对 `/predict` 的 Body 选择 form-data，字段 `file` 的类型必须选 File，不能选文本；不要手动填写 multipart boundary。

```powershell
curl.exe -X POST http://127.0.0.1:5000/calculate -F Operation=sum -F Data_A=3 -F Data_B=4
curl.exe -X POST http://127.0.0.1:5000/predict -F "file=@example.png"
python -m unittest discover -s tests -v
```

Linux 使用 `curl` 替代 `curl.exe`。可额外测试除零、缺字段、`NaN`、空文件、非图片、超大文件及错误 key。
基础成功响应仍严格为 `{"Result": result}`，错误返回 JSON `error` 字段。400 输入错误；401 key 错误；
413 上传过大；415 类型错误；503 未配置模型；500 推理内部异常，详细堆栈只写服务日志。

## AutoDL 持续服务（选做，未实际部署）

Waitress 支持 Windows 和 Linux。Linux 在 `task_Deployment/` 下：

```bash
export MODEL_KIND=derainer
export MODEL_DEVICE=cuda:0
export DERAIN_CHECKPOINT=../task_Deraining/runs/baseline/best.pt
waitress-serve --listen=127.0.0.1:5000 --threads=4 --call app:application
```

本地用 AutoDL 实例页面给出的 SSH 主机/端口建立隧道：
`ssh -N -L 5000:127.0.0.1:5000 -p 实例SSH端口 root@实例SSH主机`，再让本机 Apifox 请求 localhost:5000。
这样不必假定 AutoDL 任意公网端口均可访问。若使用平台端口映射，按平台当前说明配置，不把 API key 写进代码。
单进程内推理串行化；多个服务进程会复制模型并增加显存，先用单进程。按实际显存缩小图片或分批请求。

更完整的知识点见 [notes.md](notes.md)。
