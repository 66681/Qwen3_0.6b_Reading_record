## Qwen3 架构学习记录

## 创建环境

### 1. 安装 uv（如未安装）

PowerShell 中执行：

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 创建虚拟环境

```bash
uv venv
```

> 执行后会在项目根目录生成 `.venv/` 文件夹。`pyproject.toml` 已存在，**不需要** `uv init`（那会覆盖现有配置）。

### 3. 激活环境

**Windows (PowerShell / CMD):**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 4. 安装依赖

先装 PyTorch（需要 CUDA 专用索引），再装其余依赖：

```bash
# PyTorch (CUDA 12.8)
uv pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 其余依赖
uv pip install -r requirements.txt
```

> **为什么用 `uv pip install` 而不是 `uv add`？**
> - `uv add` 会把依赖写入 `pyproject.toml`，适合从零管理项目
> - 本项目已有 `requirements.txt` 维护依赖列表，用 `uv pip install` 直接从文件安装即可，不修改 `pyproject.toml`
> - PyTorch 用了自定义 CUDA 索引，`uv pip install` 可以直接传 `--index-url`

###  Qwen3与传统transformer有什么区别：

GQA pre Normalization QK 做了Normalization 并且Normalization采用RMSnormalization 
