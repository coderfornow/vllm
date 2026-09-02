#!/usr/bin/env bash
# Regenerate the Python locks consumed by ROCm runtime and CI images.

set -euo pipefail

UV_VERSION="0.12.1"
UV_BIN="${UV_BIN:-uv}"
uv_command=("${UV_BIN}")
if [[ "$("${uv_command[@]}" --version | awk '{print $2}')" != "${UV_VERSION}" ]]; then
    [[ "${UV_BIN}" == "uv" ]] || { echo "${UV_BIN} must be uv ${UV_VERSION}" >&2; exit 1; }
    uv_command=(uvx --from "uv==${UV_VERSION}" uv)
fi
[[ "$("${uv_command[@]}" --version | awk '{print $2}')" == "${UV_VERSION}" ]]
excluded_packages=(
    torch torchvision torchaudio triton
    cuda-bindings cuda-pathfinder cuda-toolkit cupy-cuda12x
)
for suffix in "" -cu12 -cu13; do
    for package in \
        nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \
        nvidia-cuda-runtime nvidia-cudnn nvidia-cufft nvidia-cufile \
        nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt \
        nvidia-nccl nvidia-nvjitlink nvidia-nvshmem nvidia-nvtx; do
        excluded_packages+=("${package}${suffix}")
    done
done

exclude_args=()
for package in "${excluded_packages[@]}"; do
    exclude_args+=(--no-emit-package "${package}")
done

common_args=(
    --quiet
    --index-strategy unsafe-best-match
    --python-platform x86_64-manylinux_2_28
    --python-version 3.12
    --custom-compile-command "bash tools/lock_rocm_requirements.sh"
    "${exclude_args[@]}"
)

"${uv_command[@]}" pip compile requirements/rocm.txt \
    --output-file requirements/rocm-ci.txt "${common_args[@]}"
"${uv_command[@]}" pip compile requirements/rocm-lmcache.in \
    --output-file requirements/rocm-lmcache.txt "${common_args[@]}"
"${uv_command[@]}" pip compile requirements/test/rocm.in \
    --constraint requirements/rocm.txt \
    --output-file requirements/test/rocm.txt "${common_args[@]}"
