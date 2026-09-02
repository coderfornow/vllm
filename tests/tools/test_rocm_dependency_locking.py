# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import regex as re

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "lockfile",
    [
        "requirements/rocm-ci.txt",
        "requirements/rocm-lmcache.txt",
        "requirements/test/rocm.txt",
    ],
)
def test_rocm_lockfiles_contain_only_exact_requirements(lockfile: str) -> None:
    entries = [
        line
        for line in (REPO_ROOT / lockfile).read_text().splitlines()
        if line and not line[0].isspace() and not line.startswith("#")
    ]
    exact_requirement = re.compile(
        r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
        r"(?:\[[A-Za-z0-9,._-]+\])?==[^=\s][^\s]*(?:\s*;\s*.+)?$"
    )
    assert entries
    names = []
    invalid_entries = []
    for entry in entries:
        match = exact_requirement.fullmatch(entry)
        if match is None:
            invalid_entries.append(entry)
            continue
        names.append(match.group("name").lower().replace("_", "-"))
    assert not invalid_entries, invalid_entries
    assert len(names) == len(set(names))


def _docker_stage(path: str, name: str) -> str:
    dockerfile = (REPO_ROOT / path).read_text()
    match = re.search(
        rf"(?ms)^FROM\s+[^\n]+\s+AS\s+{re.escape(name)}\s*$"
        r"(?P<body>.*?)(?=^FROM\s|\Z)",
        dockerfile,
    )
    assert match, f"missing {name} stage in {path}"
    return re.sub(r"\\\s*\n|\s+", " ", match.group("body"))


@pytest.mark.parametrize(
    ("dockerfile", "stage", "source_lock", "installed_lock", "must_check"),
    [
        (
            "docker/Dockerfile.rocm_base",
            "final",
            "requirements/rocm-ci.txt",
            "rocm-ci.txt",
            True,
        ),
        (
            "docker/Dockerfile.rocm",
            "build_vllm_dependencies",
            "requirements/rocm-ci.txt",
            "rocm-ci.txt",
            False,
        ),
        (
            "docker/Dockerfile.rocm",
            "ci_base",
            "requirements/test/rocm.txt",
            "rocm-test-reqs.txt",
            True,
        ),
    ],
)
def test_rocm_stages_install_locks_without_resolving(
    dockerfile: str,
    stage: str,
    source_lock: str,
    installed_lock: str,
    must_check: bool,
) -> None:
    body = _docker_stage(dockerfile, stage)

    assert source_lock in body
    assert re.search(
        rf"uv pip install --system --no-deps -r \S*{re.escape(installed_lock)}", body
    )
    if must_check:
        assert "uv pip check --system" in body


def test_rocm_release_stages_do_not_resolve_loose_runtime_requirements() -> None:
    for stage in ("build_vllm_wheel_release", "final_common"):
        body = _docker_stage("docker/Dockerfile.rocm", stage)
        assert "uv pip install --system --no-deps -r requirements/rocm-ci.txt" in body
        assert not re.search(r"uv pip install[^\n]*-r requirements/rocm\.txt", body)

    runtime_lock = (REPO_ROOT / "requirements/rocm-ci.txt").read_text()
    for dependency in ("flydsl==0.2.4", "pybind11==", "soundfile==", "quart=="):
        assert re.search(rf"(?m)^{re.escape(dependency)}", runtime_lock)

    lmcache = _docker_stage("docker/Dockerfile.rocm", "final_lmcache_true")
    assert "uv pip install --system --no-deps -r /tmp/rocm-lmcache.txt" in lmcache
    assert "uv pip check --system" in lmcache


def test_rocm_native_build_retains_setup_requirement_metadata() -> None:
    body = _docker_stage("docker/Dockerfile.rocm", "build_vllm_dependencies")
    assert "requirements/common.txt" in body
    assert "requirements/rocm.txt" in body


def test_torchcodec_install_does_not_accept_an_unpinned_parent_copy() -> None:
    script = (REPO_ROOT / "tools/install_torchcodec_rocm.sh").read_text()
    assert "0b261b98080925f2b709712a5491a1e8dd817065" in script
    assert "already installed and working. Skipping." not in script


def test_pinned_flash_attention_source_is_always_built() -> None:
    body = _docker_stage("docker/Dockerfile.rocm_base", "build_fa")
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in body
