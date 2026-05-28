---
name: ssh-remote-connect
description: 当用户要求 Codex SSH 登录已配置的远程主机、查看或更新本项目保存的 SSH 连接信息、在本地设置远程密码、同步本地改动到远端 vLLM Ascend/NPU 环境、运行远端验证命令，或排查项目内 SSH 远程连接和调试流程时使用。
---

# SSH 远程连接

## 概览

这个技能管理项目内的 SSH 连接、远端同步、远端验证和日志收集流程。连接信息保存在 `scripts/connection.local.env`，脚本会读取这些值，但不会打印密码。

默认本地仓库是事实源，远端机器是安装和验证目标。远端 vLLM Ascend 环境安装按 `vllm-ascend-remote-env` skill 执行，但 SSH 认证和连接变量统一使用本 skill。

## 文件

- `scripts/connection.local.env`：本地凭据文件。不要打印其中的密码值。
- `scripts/connection.env.example`：支持字段的模板文件。
- `scripts/connect.sh`：连接到已配置的远程主机。
- `scripts/set-password.sh`：提示输入密码，并写入本地凭据文件。

## 连接流程

1. 查看非敏感配置：
   ```bash
   sed -n '/SSH_PASSWORD/!p' .agents/skills/ssh-remote-connect/scripts/connection.local.env
   ```
2. 设置或更新密码：
   ```bash
   .agents/skills/ssh-remote-connect/scripts/set-password.sh
   ```
3. 连接远程主机：
   ```bash
   .agents/skills/ssh-remote-connect/scripts/connect.sh
   ```

## 远端变量

需要 `rsync` 或安装远端环境时，从 `connection.local.env` 读取连接变量，再构造远端目标。不要把密码写进命令参数。

```bash
CONNECT_SKILL_DIR=".agents/skills/ssh-remote-connect"
CONNECT_CONFIG="${SSH_REMOTE_CONFIG:-${CONNECT_SKILL_DIR}/scripts/connection.local.env}"

set -a
. "${CONNECT_CONFIG}"
set +a

: "${SSH_HOST:?Missing SSH_HOST}"
: "${SSH_PORT:?Missing SSH_PORT}"
: "${SSH_USER:?Missing SSH_USER}"

SSH_TARGET="${SSH_USER}@${SSH_HOST}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/root/vllm-ascend}"
REMOTE_VENV="${REMOTE_REPO_DIR}/.venv"
REMOTE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/python3.11.14/bin:\$PATH"
REMOTE_RSYNC_SSH="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new"
```

如果设置了 `SSH_PASSWORD` 且本机存在 `sshpass`，远端命令使用：

```bash
SSHPASS="${SSH_PASSWORD}" sshpass -e ssh -p "${SSH_PORT}" \
  -o StrictHostKeyChecking=accept-new "${SSH_TARGET}" "uname -a"
```

如果没有密码或没有 `sshpass`，使用普通 `ssh`。直接连接优先使用 `scripts/connect.sh`：

```bash
.agents/skills/ssh-remote-connect/scripts/connect.sh "cd ${REMOTE_REPO_DIR} && source ${REMOTE_VENV}/bin/activate && python --version"
```

## 镜像和输出策略

远端安装默认直接把命令输出打印到当前终端，不保存本地日志。只有命令输出过长、失败后需要复盘，或用户明确要求保留证据时，才把输出保存到本地文件。

默认 PyPI 镜像使用华为云源；实测远端下载 `vllm==0.18.0` 明显快于清华源。

```bash
PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-repo.huaweicloud.com}"

# 备选：
# PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
# PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
# PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
# PIP_TRUSTED_HOST="mirrors.aliyun.com"
```

安装普通 PyPI 包时使用：

```bash
python -m pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
  --timeout 180 --retries 10 <packages>
```

`torch==2.9.0+cpu` / `torchvision==0.24.0+cpu` / `torchaudio==2.9.0+cpu` 仍使用 PyTorch CPU index；如果远端下载慢，先在本机下载 Linux x86_64 wheelhouse，再用 `rsync` 上传到远端安装。

## 远端调试流程

1. 检查本地改动范围：
   ```bash
   git status --short
   git diff --stat
   ```
2. 只读检查远端状态：
   ```bash
   .agents/skills/ssh-remote-connect/scripts/connect.sh \
     "export PATH=${REMOTE_PATH}; cd ${REMOTE_REPO_DIR} && test -d ${REMOTE_VENV} && source ${REMOTE_VENV}/bin/activate && python --version"
   ```
3. 同步本地仓库时，先 dry-run，再正式同步。远端验证优先只同步相关源码、测试和必要配置，避免全仓 `--delete` 触碰远端已编译或已安装的第三方目录。排除 `.git`、虚拟环境、日志、缓存、构建产物和大模型权重，并显式保护远端第三方目录：
   ```bash
   RSYNC_PATHS=(
     pyproject.toml
     setup.py
     vllm_ascend/
     tests/
   )

   RSYNC_EXCLUDES=(
     --exclude='.git/'
     --exclude='.venv/'
     --exclude='.remote-logs/'
     --exclude='.mypy_cache/'
     --exclude='.pytest_cache/'
     --exclude='.ruff_cache/'
     --exclude='__pycache__/'
     --exclude='*.pyc'
     --exclude='build/'
     --exclude='dist/'
     --exclude='*.egg-info/'
     --exclude='*.safetensors'
     --exclude='*.bin'
     --exclude='*.pt'
   )

   RSYNC_PROTECTS=(
     --filter='P csrc/third_party/***'
     --filter='P third_party/***'
     --filter='P vllm-workspace/***'
     --filter='P .venv/***'
     --filter='P build/***'
     --filter='P dist/***'
   )

   SSHPASS="${SSH_PASSWORD}" sshpass -e rsync -az --dry-run --itemize-changes --delete \
     "${RSYNC_EXCLUDES[@]}" "${RSYNC_PROTECTS[@]}" -e "${REMOTE_RSYNC_SSH}" \
     "${RSYNC_PATHS[@]}" "${SSH_TARGET}:${REMOTE_REPO_DIR}/"

   SSHPASS="${SSH_PASSWORD}" sshpass -e rsync -az --delete \
     "${RSYNC_EXCLUDES[@]}" "${RSYNC_PROTECTS[@]}" -e "${REMOTE_RSYNC_SSH}" \
     "${RSYNC_PATHS[@]}" "${SSH_TARGET}:${REMOTE_REPO_DIR}/"
   ```

   只有需要完整重建远端仓库时，才使用 `./` 作为同步源；使用前必须确认远端没有需要保留的编译产物、第三方源码、wheelhouse 或本地验证资产。
4. 运行远端验证命令时，始终进入远端仓库并激活远端 venv：
   ```bash
   .agents/skills/ssh-remote-connect/scripts/connect.sh \
     "cd ${REMOTE_REPO_DIR} && source ${REMOTE_VENV}/bin/activate && python -m pytest -q tests/ut/test_envs.py"
   ```
5. NPU 验证先采集设备信息：
   ```bash
   .agents/skills/ssh-remote-connect/scripts/connect.sh \
     "export PATH=${REMOTE_PATH}; npu-smi info"
   ```

## 远端 vLLM Ascend 环境安装

安装远端环境时执行 `vllm-ascend-remote-env` skill 的步骤，但把其中的 `SSH_REMOTE_TARGET` / `SSH_REMOTE_PORT` / `SSH_REMOTE_PASSWORD` 替换为本 skill 的 `SSH_TARGET` / `SSH_PORT` / `SSH_PASSWORD`。

安装过程默认直接打印输出。出现下载过慢时，先切换 `PIP_INDEX_URL`；大型 wheel 仍然过慢时，在本机下载 Linux wheelhouse 后通过 `rsync` 上传远端安装。

## 规则

- 永远不要打印 `SSH_PASSWORD`，也不要在总结中包含它。
- 如果设置了 `SSH_PASSWORD` 且本机存在 `sshpass`，`connect.sh` 会使用 `sshpass -e`。
- 如果没有 `sshpass`，或密码为空，`connect.sh` 会退回使用普通 `ssh`。
- 传给 `connect.sh` 的额外参数会继续传给 `ssh`，因此可以这样执行远程命令：
  ```bash
  .agents/skills/ssh-remote-connect/scripts/connect.sh uname -a
  ```
- 不要把密码、私钥或 token 写入 skill、PRD、日志、命令摘要或最终汇报。
- 大范围同步前必须先 dry-run，并检查同步范围。
- 远端验证同步默认只同步相关源码、测试和必要配置；不要用全仓 `--delete` 触碰远端已编译或已安装的第三方目录。
- 同步包含 `csrc/`、`third_party/`、`vllm-workspace/`、`.venv/`、`build/`、`dist/` 等路径时，必须先加 `RSYNC_PROTECTS` 保护规则。
- 如果远端存在用户声明的重要改动，停止同步并报告冲突点。
- 最终汇报只写本地改动文件、远端命令退出码和关键结果摘要；只有实际保存过日志时才写日志路径。
