---
name: vllm-ascend-remote-env
description: "在远端 Ascend/NPU 机器上安装 vLLM Ascend 开发环境：补齐 rg/rsync/git/编译工具，安全同步本地仓库，创建 .venv，安装 torch-npu/vLLM/vllm-ascend 依赖，编译自定义算子，并运行通用环境 smoke 与既有 UT 验证。"
---

# vLLM Ascend 远端环境安装

用这个 skill 处理“把远端机器装成能编译、运行、验证当前 vllm-ascend 改动”的任务。默认本地仓库是事实源，远端只作为安装和验证目标。

## 安全约束

- 需要 SSH 时，优先搭配 `ssh-remote-connect` skill 的连接变量和认证规则。
- 密码只能从当前进程环境变量 `SSH_REMOTE_PASSWORD` 读取；不要把密码、私钥、token 写入命令参数、skill、日志、PRD、shell history 或仓库文件。
- 使用 `sshpass` 时只允许 `SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e ...`，不要使用会把密码暴露在进程列表里的参数模式。
- 远端环境可以安装系统包、创建 venv、同步仓库、编译产物；不要覆盖远端用户声明的重要改动。
- 安装和验证输出默认直接打印到当前终端；只有输出过长、失败后需要复盘，或用户明确要求保留证据时，才写入本地 `.remote-logs/<date>-<topic>/`。

## PyPI 镜像策略

默认普通 PyPI 包使用华为云源；远端下载大包明显慢时，再切换到其他镜像或本地 wheelhouse。

```bash
PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-repo.huaweicloud.com}"

# 备选：
# PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
# PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
# PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
# PIP_TRUSTED_HOST="mirrors.aliyun.com"
```

普通包安装统一追加：

```bash
-i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" --timeout 180 --retries 10
```

`torch` 三件套仍使用 PyTorch CPU index；不要从普通 PyPI 镜像安装 `torch==2.10.0`，华为云普通 PyPI 上可能命中 CUDA wheel 并拉取 NVIDIA 依赖，不适合 NPU 环境。大型 wheel 下载过慢时，在本机下载 Linux x86_64 wheelhouse，再用 `rsync` 上传远端安装。

## main 分支版本依赖

默认使用 vLLM Ascend 官方版本管理策略页面里的 `main` 兼容性矩阵。每次安装前先查看该页面；页面会定期更新，本地 `docs/source/conf.py`、`requirements.txt`、`pyproject.toml` 只作为仓库内校验来源。

```bash
VLLM_ASCEND_VERSION_POLICY_URL="${VLLM_ASCEND_VERSION_POLICY_URL:-https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/community/versioning_policy.html}"
VLLM_MAIN_COMMIT="${VLLM_MAIN_COMMIT:-1ac10f159a09897baada01b14b6a0dd6442aefd6}"
VLLM_MAIN_TAG="${VLLM_MAIN_TAG:-v0.20.2}"
CANN_VERSION="${CANN_VERSION:-9.0.0}"
CANN_OBS_BASE_URL="${CANN_OBS_BASE_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
TRITON_ASCEND_VERSION="${TRITON_ASCEND_VERSION:-3.2.1}"
RIPGREP_VERSION="${RIPGREP_VERSION:-14.1.1}"
```

## 默认变量

如果用户没有另行指定，沿用这些变量；不要在 skill 中填入密码。

```bash
SSH_REMOTE_TARGET="${SSH_REMOTE_TARGET:-root+vm-cWQ6VxEWjhnvCdN0@106.75.235.239}"
SSH_REMOTE_PORT="${SSH_REMOTE_PORT:-32222}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/root/vllm-ascend}"
REMOTE_VLLM_DIR="${REMOTE_VLLM_DIR:-/root/vllm}"
REMOTE_VENV="${REMOTE_REPO_DIR}/.venv"
REMOTE_SSH=(ssh -p "${SSH_REMOTE_PORT}" -o StrictHostKeyChecking=accept-new "${SSH_REMOTE_TARGET}")
REMOTE_RSYNC_SSH="ssh -p ${SSH_REMOTE_PORT} -o StrictHostKeyChecking=accept-new"
REMOTE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/python3.11.14/bin:\$PATH"
```

如果当前项目同时使用 `ssh-remote-connect` skill，优先读取其本地连接文件并映射变量。该文件只能本机读取，禁止同步到远端。

```bash
if [ -f .agents/skills/ssh-remote-connect/scripts/connection.local.env ]; then
  set -a
  source .agents/skills/ssh-remote-connect/scripts/connection.local.env
  set +a
  SSH_REMOTE_TARGET="${SSH_USER}@${SSH_HOST}"
  SSH_REMOTE_PORT="${SSH_PORT}"
  SSH_REMOTE_PASSWORD="${SSH_PASSWORD}"
  REMOTE_SSH=(ssh -p "${SSH_REMOTE_PORT}" -o StrictHostKeyChecking=accept-new "${SSH_REMOTE_TARGET}")
  REMOTE_RSYNC_SSH="ssh -p ${SSH_REMOTE_PORT} -o StrictHostKeyChecking=accept-new"
fi
```

## 安装流程

### 1. 建立日志和认证前置检查

```bash
test -n "${SSH_REMOTE_PASSWORD:-}" || {
  printf '%s\n' "SSH_REMOTE_PASSWORD is not set" >&2
  exit 1
}
command -v sshpass >/dev/null || {
  printf '%s\n' "sshpass is required for password auth" >&2
  exit 1
}
```

### 2. 选择当前 main 对应的版本组

安装前在本地刷新 `upstream/main`，从仓库记录读取 vLLM commit/tag。`pip_vllm_version` 只作为发布轮子信息，不作为本流程的 vLLM 安装来源。

```bash
git fetch upstream main
CONF_FILE="$(mktemp)"
git show upstream/main:docs/source/conf.py > "${CONF_FILE}"
VLLM_MAIN_COMMIT="$(
  python3 - "${CONF_FILE}" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
print(re.search(r'main_vllm_commit\s*=\s*"([^"]+)"', text).group(1))
PY
)"
VLLM_MAIN_TAG="$(
  python3 - "${CONF_FILE}" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
print(re.search(r'main_vllm_tag\s*=\s*"([^"]+)"', text).group(1))
PY
)"
rm -f "${CONF_FILE}"
printf 'VLLM_MAIN_COMMIT=%s\nVLLM_MAIN_TAG=%s\n' "${VLLM_MAIN_COMMIT}" "${VLLM_MAIN_TAG}"
```

### 3. 远端 smoke 和系统工具安装

远端非登录 shell 的 `PATH` 可能很瘦，所有命令都先显式设置 `REMOTE_PATH`。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" \
  "export PATH=${REMOTE_PATH}; set -e; uname -a; npu-smi info || true"
```

安装常用工具和编译工具。同步工具是 `rsync`；`rg` 优先安装系统包 `ripgrep`，若包源不提供则安装官方 Linux x86_64 预编译二进制。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -e
  if command -v apt-get >/dev/null; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      rsync git ripgrep curl wget tar unzip zip xz-utils gzip bzip2 patch \
      build-essential pkg-config cmake ninja-build python3-dev libnuma-dev jq \
      ca-certificates openssh-client tmux less vim procps
  elif command -v dnf >/dev/null; then
    dnf install -y \
      rsync git ripgrep curl wget tar unzip zip xz gzip bzip2 patch \
      gcc gcc-c++ make pkgconf cmake ninja-build python3-devel numactl-devel jq \
      ca-certificates openssh-clients tmux less vim procps-ng || \
    dnf install -y \
      rsync git curl wget tar unzip zip xz gzip bzip2 patch \
      gcc gcc-c++ make pkgconf cmake ninja-build python3-devel numactl-devel jq \
      ca-certificates openssh-clients tmux less vim procps-ng
  elif command -v yum >/dev/null; then
    yum install -y \
      rsync git ripgrep curl wget tar unzip zip xz gzip bzip2 patch \
      gcc gcc-c++ make pkgconfig cmake ninja-build python3-devel numactl-devel jq \
      ca-certificates openssh-clients tmux less vim procps-ng || \
    yum install -y \
      rsync git curl wget tar unzip zip xz gzip bzip2 patch \
      gcc gcc-c++ make pkgconfig cmake ninja-build python3-devel numactl-devel jq \
      ca-certificates openssh-clients tmux less vim procps-ng
  else
    echo 'no supported package manager' >&2
    exit 1
  fi

  if ! command -v rg >/dev/null 2>&1; then
    tmpdir=\$(mktemp -d)
    archive=\"ripgrep-${RIPGREP_VERSION}-x86_64-unknown-linux-musl\"
    curl -L --retry 5 --connect-timeout 30 \
      -o \"\${tmpdir}/ripgrep.tar.gz\" \
      \"https://github.com/BurntSushi/ripgrep/releases/download/${RIPGREP_VERSION}/\${archive}.tar.gz\"
    tar -xzf \"\${tmpdir}/ripgrep.tar.gz\" -C \"\${tmpdir}\"
    install -m 0755 \"\${tmpdir}/\${archive}/rg\" /usr/local/bin/rg
    rm -rf \"\${tmpdir}\"
  fi

  command -v rsync
  command -v rg
  command -v git
  command -v gcc
  command -v g++
  command -v cmake
  command -v patch
"
```

### 4. 同步本地仓库

先 dry-run，再正式同步。不要同步 `.git`、虚拟环境、远端日志、缓存、构建产物或大模型权重。

```bash
RSYNC_EXCLUDES=(
  --exclude='.git/'
  --exclude='.venv/'
  --exclude='.remote-logs/'
  --exclude='.mypy_cache/'
  --exclude='.pytest_cache/'
  --exclude='.ruff_cache/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='._*'
  --exclude='.DS_Store'
  --exclude='build/'
  --exclude='dist/'
  --exclude='*.egg-info/'
  --exclude='*.safetensors'
  --exclude='*.bin'
  --exclude='*.pt'
  --exclude='.agents/skills/ssh-remote-connect/scripts/connection.local.env'
)

SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e rsync -az --dry-run --itemize-changes --delete \
  "${RSYNC_EXCLUDES[@]}" -e "${REMOTE_RSYNC_SSH}" \
  ./ "${SSH_REMOTE_TARGET}:${REMOTE_REPO_DIR}/"

SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e rsync -az --delete \
  "${RSYNC_EXCLUDES[@]}" -e "${REMOTE_RSYNC_SSH}" \
  ./ "${SSH_REMOTE_TARGET}:${REMOTE_REPO_DIR}/"
```

如果远端暂时装不上 `rsync`，可以临时用 tar over SSH 同步，但必须排除 macOS AppleDouble 文件。同步后再清理一次 `._*`，否则 CANN 会把 `._opapi_stub.cpp` 当 C++ 源文件编译。

```bash
tar --exclude='.git' --exclude='.venv' --exclude='._*' --exclude='.DS_Store' \
  --exclude='build' --exclude='dist' --exclude='*.egg-info' \
  --exclude='.agents/skills/ssh-remote-connect/scripts/connection.local.env' \
  -czf /tmp/vllm-ascend-sync.tgz .
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" \
  "mkdir -p ${REMOTE_REPO_DIR}"
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e scp -P "${SSH_REMOTE_PORT}" \
  /tmp/vllm-ascend-sync.tgz "${SSH_REMOTE_TARGET}:/tmp/vllm-ascend-sync.tgz"
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  set -e
  mkdir -p ${REMOTE_REPO_DIR}
  tar -xzf /tmp/vllm-ascend-sync.tgz -C ${REMOTE_REPO_DIR}
  cd ${REMOTE_REPO_DIR}
  find . -name '._*' -delete
  test -f csrc/cmake/third_party/build/modules/patch/protobuf_25.1_change_version.patch
  test ! -f .agents/skills/ssh-remote-connect/scripts/connection.local.env
"
```

### 5. 创建 venv 并安装 Python/NPU 依赖

当前 vLLM Ascend main 依赖以 `VLLM_ASCEND_VERSION_POLICY_URL` 的 `main` 行为准。当前记录为 `torch==2.10.0` / `torch-npu==2.10.0` / `triton-ascend==3.2.1`，CANN 版本为 `9.0.0`，vLLM 提交为 `1ac10f159a09897baada01b14b6a0dd6442aefd6`。必须先安装 torch 三件套，再安装 `torch-npu` 和 `triton-ascend`。不要直接安装 PyPI `vllm==${VLLM_MAIN_TAG#v}`，该 wheel 可能要求不同的 torch 版本。

```bash
PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-repo.huaweicloud.com}"

SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  PYTHON_BIN=/usr/local/python3.11.14/bin/python3
  test -x \${PYTHON_BIN} || PYTHON_BIN=python3
  rm -rf ${REMOTE_VENV}
  \${PYTHON_BIN} -m venv ${REMOTE_VENV}
  source ${REMOTE_VENV}/bin/activate

  python -m pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --timeout 180 --retries 10 -U 'pip<27' 'setuptools<81' wheel packaging
  python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    'torch==${TORCH_VERSION}' 'torchvision==${TORCHVISION_VERSION}' 'torchaudio==${TORCHAUDIO_VERSION}'
  python -m pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --timeout 180 --retries 10 'torch-npu==${TORCH_NPU_VERSION}'
  python -m pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --timeout 180 --retries 10 'triton-ascend==${TRITON_ASCEND_VERSION}'

  grep -Ev '^(torch==|torchvision==|torchaudio==|torch-npu==|triton-ascend==)' requirements.txt \
    > /tmp/vllm_ascend_requirements_no_torch.txt
  python -m pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --timeout 180 --retries 10 -r /tmp/vllm_ascend_requirements_no_torch.txt

  rm -rf ${REMOTE_VLLM_DIR}
  git clone https://github.com/vllm-project/vllm.git ${REMOTE_VLLM_DIR}
  cd ${REMOTE_VLLM_DIR}
  git checkout ${VLLM_MAIN_COMMIT}
  VLLM_TARGET_DEVICE=empty python -m pip install -i "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" -e . --no-build-isolation --no-deps
"
```

如果远端无法稳定访问 GitHub，在本机打包 vLLM 指定 commit 后上传远端。

```bash
LOCAL_VLLM_BUNDLE="${LOCAL_VLLM_BUNDLE:-/tmp/vllm-${VLLM_MAIN_COMMIT}.bundle}"
git -C "${LOCAL_VLLM_SOURCE_DIR:-../vllm}" fetch origin "${VLLM_MAIN_COMMIT}"
git -C "${LOCAL_VLLM_SOURCE_DIR:-../vllm}" bundle create "${LOCAL_VLLM_BUNDLE}" "${VLLM_MAIN_COMMIT}"
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e rsync -az \
  -e "${REMOTE_RSYNC_SSH}" \
  "${LOCAL_VLLM_BUNDLE}" "${SSH_REMOTE_TARGET}:/tmp/vllm.bundle"
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  set -euo pipefail
  rm -rf ${REMOTE_VLLM_DIR}
  git clone /tmp/vllm.bundle ${REMOTE_VLLM_DIR}
  cd ${REMOTE_VLLM_DIR}
  git checkout ${VLLM_MAIN_COMMIT}
  source ${REMOTE_VENV}/bin/activate
  VLLM_TARGET_DEVICE=empty python -m pip install -i "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" -e . --no-build-isolation --no-deps
"
```

### 6. 准备 Catlass 和 CANN 环境

如果同步时排除了 `.git`，远端不能执行 `git submodule update`。需要显式拉取 `catlass`。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  rm -rf csrc/third_party/catlass
  git clone https://gitcode.com/cann/catlass.git csrc/third_party/catlass
  CATLASS_COMMIT=\$(git config -f .gitmodules --get submodule.csrc/third_party/catlass.commit || true)
  if [ -n "\${CATLASS_COMMIT}" ]; then
    git -C csrc/third_party/catlass fetch origin "\${CATLASS_COMMIT}" || \
      git -C csrc/third_party/catlass fetch origin
    git -C csrc/third_party/catlass checkout "\${CATLASS_COMMIT}"
  fi
  test -d csrc/third_party/catlass/include
  test -f csrc/third_party/catlass/include/catlass/catlass.hpp
"
```

### 7. 910B2C CANN/OPP 完整性检查

910B2C 机器不能只看 `npu-smi` 和 `torch.npu.is_available()`；还要确认 OPP 里有 910B 算子包。缺完整 910B OPP 时，模型初始化可能在普通 NPU 张量操作上失败。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  source ${REMOTE_VENV}/bin/activate
  if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
  elif [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    set -u
  fi
  ls -d /usr/local/Ascend/cann-* 2>/dev/null || true
  find /usr/local/Ascend -maxdepth 8 -path '*opp*ascend910b*' -type d 2>/dev/null | head
  python - <<'PY'
import torch
import torch_npu

torch.npu.set_device(0)
x = torch.empty((2, 2), device='npu')
x.zero_()
torch.npu.synchronize()
print('npu_zero_smoke', x.cpu().tolist())
PY
"
```

如果 `zero_()` 失败并出现 `aclnnInplaceZero failed`、`error code is 561103`、`Parse dynamic kernel config fail`，按当前 CANN 版本安装官方 910B ops 包，再重跑上面的 smoke。

优先在本机缓存官方 `.run` 安装器，再用 `rsync` 传到远端，便于后续复用。

```bash
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-${HOME}/.cache/vllm-ascend}"
OPS_RUN_NAME="Ascend-cann-910b-ops_${CANN_VERSION}_linux-x86_64.run"
mkdir -p "${LOCAL_CACHE_DIR}"
curl -L -C - --retry 20 --retry-delay 3 --connect-timeout 30 \
  --speed-time 60 --speed-limit 10240 \
  -o "${LOCAL_CACHE_DIR}/${OPS_RUN_NAME}" \
  "${CANN_OBS_BASE_URL}/${OPS_RUN_NAME}"
test -s "${LOCAL_CACHE_DIR}/${OPS_RUN_NAME}"
stat -f '%z %N' "${LOCAL_CACHE_DIR}/${OPS_RUN_NAME}" 2>/dev/null || \
  stat -c '%s %n' "${LOCAL_CACHE_DIR}/${OPS_RUN_NAME}"

SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e rsync -az --progress \
  -e "${REMOTE_RSYNC_SSH}" \
  "${LOCAL_CACHE_DIR}/${OPS_RUN_NAME}" \
  "${SSH_REMOTE_TARGET}:/usr/local/Ascend/${OPS_RUN_NAME}"

SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  stat -c '%s %n' /usr/local/Ascend/${OPS_RUN_NAME}
  chmod +x /usr/local/Ascend/${OPS_RUN_NAME}
  /usr/local/Ascend/${OPS_RUN_NAME} \
    --install --quiet --install-path=/usr/local/Ascend --install-for-all --force
"
```

当前 9.0.0 x86_64 官方 OBS 地址是 `https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-x86_64.run`。官方包下载地址可能变化；地址失效时，到昇腾社区 CANN 离线安装文档里按相同命名规则查 `Ascend-cann-910b-ops_${CANN_VERSION}_linux-x86_64.run`。不要默认下载或手工安装裸 `.rpm` 包；`.run` 安装器会处理内部组件包和安装路径。不要用 910A、310P 或只有 toolkit 的包替代 910B ops 包。

安装 910B ops 后必须重跑 `npu_zero_smoke`。只有 `zero_()` 通过，才继续编译 vLLM Ascend。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  source ${REMOTE_VENV}/bin/activate
  if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    set -u
  elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
  fi
  python - <<'PY'
import torch
import torch_npu

torch.npu.set_device(0)
x = torch.empty((2, 2), device='npu')
x.zero_()
torch.npu.synchronize()
print('npu_zero_smoke', x.cpu().tolist())
PY
"
```

### 8. 编译安装 vllm-ascend

Ascend 910B2C 使用 `SOC_VERSION=ascend910b2c`。如果硬件不同，先用 `npu-smi info` 判断并调整。

```bash
PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-repo.huaweicloud.com}"

SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  source ${REMOTE_VENV}/bin/activate
  if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
  elif [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    set -u
  fi
  if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
  fi
  export SOC_VERSION=ascend910b2c
  export CPATH=${REMOTE_REPO_DIR}/csrc/third_party/catlass/include:\${CPATH:-}
  test -f csrc/cmake/third_party/build/modules/patch/protobuf_25.1_change_version.patch
  test -f csrc/third_party/catlass/include/catlass/catlass.hpp
  python -m pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    -e . --no-build-isolation
  python - <<'PY'
import torch, torch_npu, vllm, vllm_ascend
print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('vllm', vllm.__version__)
print('vllm_ascend', getattr(vllm_ascend, '__version__', 'unknown'))
PY
"
```

如果 pip 输出被截断，只看到 `gmake: *** [Makefile:156: all] Error 2`，先查真实失败点。修复后保留 `csrc/build` 继续构建。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  source ${REMOTE_VENV}/bin/activate
  if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    set -u
  elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
  fi
  export SOC_VERSION=ascend910b2c
  export CPATH=${REMOTE_REPO_DIR}/csrc/third_party/catlass/include:\${CPATH:-}
  cd csrc/build
  gmake -j8 > /tmp/csrc_gmake_j8.log 2>&1 || {
    grep -nEi 'error:|fatal|No such file|Killed|undefined reference|gmake.*\\*\\*\\*' /tmp/csrc_gmake_j8.log | head -n 80
    exit 1
  }
"
```

## 通用验证

### 1. NPU 状态

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" \
  "export PATH=${REMOTE_PATH}; source ${REMOTE_VENV}/bin/activate; npu-smi info"
```

### 2. 基础 Python/NPU smoke

用于确认远端 venv、torch-npu、vLLM、vllm-ascend、NPU 可见性和自定义 op 动态库导入正常。这个 smoke 不依赖新增特性代码。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  source ${REMOTE_VENV}/bin/activate
  if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    set -u
  elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
  fi
  python - <<'PY'
import torch
import torch_npu
import vllm
import vllm_ascend.vllm_ascend_C  # noqa: F401

print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('vllm', vllm.__version__)
print('npu_available', torch.npu.is_available())
torch.npu.set_device(0)
anchor = torch.zeros(1, dtype=torch.float32, device='npu')
torch.npu.synchronize()
print('anchor', tuple(anchor.shape), anchor.dtype, anchor.device)
print('custom_namespace_available', hasattr(torch.ops, '_C_ascend'))
PY
"
```

### 3. 既有 UT 子集

默认使用仓库原本就有的通用测试，避免依赖当前分支新增的专项代码。若某个测试在目标分支不存在，先用 `test -f` 过滤，再记录实际执行列表。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  export PATH=${REMOTE_PATH}
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  source ${REMOTE_VENV}/bin/activate
  if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    set -u
  elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
  fi
  TESTS=(
    tests/ut/test_envs.py
    tests/ut/test_ascend_config.py
    tests/ut/quantization/test_quant_parser.py
    tests/ut/quantization/methods/test_registry.py
    tests/ut/ops/test_prepare_finalize.py
  )
  EXISTING_TESTS=()
  for test_path in \"\${TESTS[@]}\"; do
    if [ -f \"\${test_path}\" ]; then
      EXISTING_TESTS+=(\"\${test_path}\")
    else
      echo \"skip missing test: \${test_path}\"
    fi
  done
  printf 'running tests:%s\n' \" \${EXISTING_TESTS[*]}\"
  python -m pytest -q \"\${EXISTING_TESTS[@]}\"
"
```

期望结果：上述既有 UT 子集通过；如果失败，记录失败测试、退出码和日志路径。

### 4. 依赖一致性检查

最后执行 `pip check`。CANN 自带 profiler/te 包可能报告额外依赖或 pandas/opentelemetry 版本冲突；只要核心 smoke 和自定义扩展导入通过，这类 CANN 工具包声明冲突不阻断本流程。

```bash
SSHPASS="${SSH_REMOTE_PASSWORD}" sshpass -e "${REMOTE_SSH[@]}" "
  cd ${REMOTE_REPO_DIR}
  source ${REMOTE_VENV}/bin/activate
  python -m pip check || true
"
```

## 常见故障和处理

- `rsync: command not found`：远端缺 `rsync`，先执行系统工具安装步骤。
- `rg: command not found`：远端缺 `ripgrep`，先执行系统工具安装步骤；若包源不提供 `ripgrep`，安装步骤会改用官方 Linux x86_64 预编译二进制。
- `torch==2.10.0` 从华为云普通 PyPI 拉取 CUDA/NVIDIA 依赖：停止该安装，改用 `https://download.pytorch.org/whl/cpu` 安装 `torch`、`torchvision`、`torchaudio`，再从华为云源安装 `torch-npu`。
- `torch-npu 2.10.0` 与 torch 版本不匹配：重新从 `https://download.pytorch.org/whl/cpu` 安装 `torch==2.10.0`、`torchvision==0.25.0`、`torchaudio==2.10.0`，再安装 `torch-npu==2.10.0`。
- 需要强制恢复当前 main 版本组：执行 `python -m pip install --force-reinstall --no-deps --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0`，再用华为云源 `--force-reinstall --no-deps --no-cache-dir` 安装 `torch-npu==2.10.0` 和 `triton-ascend==3.2.1`。
- vLLM 版本不匹配：删除 `${REMOTE_VLLM_DIR}` 后，重新 checkout `VLLM_MAIN_COMMIT` 并执行 `VLLM_TARGET_DEVICE=empty python -m pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" -e . --no-build-isolation --no-deps`。
- `triton-ascend` 版本不匹配：执行 `python -m pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" --force-reinstall --no-deps triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi`。
- `torch.empty(..., device="npu").zero_()` 或模型初始化时报 `aclnnInplaceZero failed`、`error code is 561103`、`Parse dynamic kernel config fail`：这是 910B2C 的 CANN/OPP 不完整或版本不匹配；安装 `Ascend-cann-910b-ops_${CANN_VERSION}_linux-x86_64.run` 到 `/usr/local/Ascend`，再重跑 `npu_zero_smoke`。
- 910B ops 安装器提示 driver 未安装，但 `npu-smi info` 正常且 `npu_zero_smoke` 通过：按 smoke 结果判断；该提示不一定阻断 vLLM 验证。
- CANN/OPP 更新后仍出现基础 NPU 算子异常：同时重装匹配版本组 `torch==2.10.0+cpu`、`torch-npu==2.10.0`、`triton-ascend==3.2.1`，确认 `torch.npu.is_available()` 和 `npu_zero_smoke` 都通过后再跑 vLLM。
- `build_aclnn.sh` 在旧 CANN 8.5.1 下报 generated op soc version 不支持：不要继续在旧 CANN 上排 C++ 代码，先升级到与当前分支匹配的 CANN/OPP 9.0.0 和 910B ops 包。
- `arctic-inference` 或其他 CMake Python 扩展报 `Could NOT find Python_INCLUDE_DIRS Development.Module`：安装 `python3-devel`；同时安装 `ninja-build`，再重新执行 requirements 安装。
- `dependency catlass is missing` 且远端没有 `.git`：显式 `git clone https://gitcode.com/cann/catlass.git csrc/third_party/catlass`，再 checkout `.gitmodules` 里的 `submodule.csrc/third_party/catlass.commit`。
- `fatal error: 'catlass/catlass.hpp' file not found`：确认 `csrc/third_party/catlass/include/catlass/catlass.hpp` 存在，并在编译前执行 `export CPATH=${REMOTE_REPO_DIR}/csrc/third_party/catlass/include:${CPATH:-}`。
- `build_aclnn.sh` 或 `gmake` 报 `patch: command not found`：安装系统包 `patch` 后从现有 `csrc/build` 继续构建，不要删除已经生成的 CANN binary 产物。
- CANN 把 `._opapi_stub.cpp`、`._*.patch` 等文件当作源码或 patch：这是 macOS AppleDouble 文件进入远端仓库，执行 `find ${REMOTE_REPO_DIR} -name '._*' -delete`，后续同步排除 `._*`。
- protobuf patch 文件缺失，例如 `protobuf_25.1_change_version.patch: No such file or directory`：同步 `csrc/cmake/third_party/build/modules/patch/` 到远端同路径，再从现有 `csrc/build` 继续执行 `gmake -j8`；通过后重新执行 `python -m pip install -e . --no-build-isolation`。
- pip editable 安装内部固定 `gmake -j2 package` 时耗时较长：先在 `csrc/build` 手动执行 `gmake -j8` 验证并生成大部分产物，再回到仓库根目录执行 pip editable 安装。
- `ModuleNotFoundError: No module named 'attr'`：补 `attrs`。
- `ModuleNotFoundError: No module named 'pkg_resources'`：安装 `setuptools<81`。
- vLLM/vllm-ascend import 缺 `cbor2`、`gguf` 等：执行 vLLM 运行时依赖补齐步骤。
- 用 `python - <<'PY'` 直接启动 vLLM 时，multiprocessing `spawn` 报无法打开 `<stdin>`：把脚本写到 `/tmp/*.py`，加 `if __name__ == "__main__":` 和 `multiprocessing.freeze_support()`，再用 `python /tmp/*.py` 运行。
- 国内源下载模型：安装 `modelscope`，设置 `VLLM_USE_MODELSCOPE=True`、`MODELSCOPE_CACHE=/data/modelscope_cache`、`HF_HOME=/data/huggingface_home`、`TRANSFORMERS_CACHE=/data/huggingface_home/hub`；先用 `Qwen/Qwen2.5-0.5B-Instruct` 做最小模型验证，再用 `Qwen/Qwen2.5-1.5B-Instruct` 扩大验证。
- CANN 自定义算子构建时间很长：完整 Ascend910B custom ops 会调用 `opc/bisheng` 生成多组 tiling key，内部并行负载可能很高。正常安装优先用 `python -m pip install -e . --no-build-isolation`；只有日志交错看不出首个错误时，才进入 `csrc/build` 用 `gmake -j1 > /tmp/csrc_gmake_j1.log 2>&1` 定位失败点。修复后保留 `csrc/build` 继续跑。

## 汇报格式

完成后用中文简短汇报：

- 远端目录、venv 路径、同步范围；
- 系统工具安装结果，特别是 `rg`、`rsync`、`git`、`gcc/g++`；
- Python 栈版本：`torch`、`torch_npu`、`vllm`、`vllm_ascend`；
- 编译安装命令和退出码；
- NPU 状态摘要；
- 基础 Python/NPU smoke、既有 UT 子集的退出码和关键结果；
- 日志路径：默认无；只有实际保存过日志时提供 `.remote-logs/` 路径；
- 未解决的依赖冲突或服务部署限制。
