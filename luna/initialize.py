"""Provision Luna's local signing keys and report production preflight blockers."""
from __future__ import annotations

import base64
import os
from pathlib import Path
import stat
import subprocess

from aegis_core.enforce import create_keypair, inspect_keys, key_directory

from .combat import combat_check


_TEMPLATE = """# 复制为 .env.luna，填写所用网关。进程环境变量优先。
# 远程端点使用 HTTPS，路径通常以 /v1 结尾。
LUNA_REASONING_BASE_URL=
LUNA_REASONING_API_KEY=
LUNA_REASONING_MODEL=
LUNA_CODING_BASE_URL=
LUNA_CODING_API_KEY=
LUNA_CODING_MODEL=
# Ollama 本地备选端点默认为 http://127.0.0.1:11434/v1
LUNA_LOCAL_MODEL=
"""
_KEY_FILES = ("ed25519-private.pem", "ed25519-public.pem", "key-metadata.json")
_FIXES = {
    "environment": "安装项目依赖和 Python 3.11/3.12，并检查可用磁盘。",
    "ed25519_key": "检查本地密钥权限、配对及过期时间，或为新密钥指定独立目录。",
    "git_permissions": "运行 gh auth login，并确认对 origin 有推送权限。",
    "sandbox_isolation": "检查 Git worktree 支持和临时目录写权限。",
    "ci": "检查默认分支最近一次 GitHub Actions 运行。",
    "model_gateway": "复制 .env.luna.example 为 .env.luna，填写模型及凭据并启动对应网关。",
}


def _windows_acl(path: Path, *, protect=False):
    script = r"""
$ErrorActionPreference = 'Stop'
$path = $env:LUNA_KEY_ACL_TARGET
$user = [Security.Principal.WindowsIdentity]::GetCurrent().User
$isDirectory = [IO.Directory]::Exists($path)
if ($env:LUNA_KEY_ACL_PROTECT -eq '1') {
    if ($isDirectory) {
        $acl = [Security.AccessControl.DirectorySecurity]::new()
        $inherit = [Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
    } else {
        $acl = [Security.AccessControl.FileSecurity]::new()
        $inherit = [Security.AccessControl.InheritanceFlags]::None
    }
    $acl.SetOwner($user)
    $acl.SetAccessRuleProtection($true, $false)
    $rule = [Security.AccessControl.FileSystemAccessRule]::new(
        $user, [Security.AccessControl.FileSystemRights]::FullControl,
        $inherit, [Security.AccessControl.PropagationFlags]::None,
        [Security.AccessControl.AccessControlType]::Allow)
    $acl.SetAccessRule($rule)
    if ($isDirectory) {
        [IO.Directory]::SetAccessControl($path, $acl)
    } else {
        [IO.File]::SetAccessControl($path, $acl)
    }
}
$sections = [Security.AccessControl.AccessControlSections]'Owner, Access'
if ($isDirectory) {
    $acl = [IO.Directory]::GetAccessControl($path, $sections)
} else {
    $acl = [IO.File]::GetAccessControl($path, $sections)
}
$sidType = [Security.Principal.SecurityIdentifier]
if ($acl.GetOwner($sidType).Value -ne $user.Value) { throw 'Unexpected key owner' }
$rules = @($acl.GetAccessRules($true, $true, $sidType))
if ($rules.Count -ne 1) { throw 'Expected one key access rule' }
$rule = $rules[0]
if ($rule.IdentityReference.Value -ne $user.Value) { throw 'Unexpected key trustee' }
if ($rule.AccessControlType -ne 'Allow') { throw 'Expected an Allow rule' }
$full = [Security.AccessControl.FileSystemRights]::FullControl
if (($rule.FileSystemRights -band $full) -ne $full) { throw 'Expected FullControl' }
if ($isDirectory) {
    $inherit = [Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
    if (!$acl.AreAccessRulesProtected) { throw 'Directory accepts inherited rules' }
    if ($rule.InheritanceFlags -ne $inherit) { throw 'Key files would not inherit protection' }
    if ($rule.PropagationFlags -ne 'None') { throw 'Unexpected access rule propagation' }
}
"""
    script = "try {\n" + script + "\n} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }"
    command = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", command],
                            env=os.environ | {"LUNA_KEY_ACL_TARGET": str(path),
                                              "LUNA_KEY_ACL_PROTECT": str(int(protect))},
                            capture_output=True, timeout=20, check=False)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()[:500]
        raise ValueError(f"无法保护或验证 Windows 密钥 ACL：{path} ({result.returncode}) {detail}")


def _reject_link(path: Path):
    if path.is_symlink() or (path.exists() and
        getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError(f"密钥路径不能是符号链接或重解析点：{path}")


def _key_location(root: Path, requested) -> Path:
    path = Path(requested or key_directory()).expanduser().absolute()
    _reject_link(path)
    path = path.resolve()
    if path.is_relative_to(root):
        raise ValueError("密钥目录必须位于 Git 仓库之外")
    for parent in (path, *path.parents):
        git_marker = parent / ".git"
        if git_marker.is_file() or (git_marker / "HEAD").is_file() or (
            (parent / "HEAD").is_file() and (parent / "objects").is_dir()
        ):
            raise ValueError("密钥目录必须位于 Git 仓库之外")
    return path


def provision_keys(root: Path, requested=None) -> dict:
    if any(os.environ.get(name) for name in ("AEGIS_SIGNING_KEY_PATH", "AEGIS_VERIFYING_KEY_PATH")):
        raise ValueError("运行 init 前请取消单文件密钥路径覆盖，并使用 --keys-dir")
    base = _key_location(root, requested)
    try:
        base.mkdir(parents=True, mode=0o700)
        new = True
    except FileExistsError:
        new = False
    _reject_link(base)
    if not base.is_dir():
        raise ValueError("密钥路径必须为目录")
    paths = [base / name for name in _KEY_FILES]
    for path in paths:
        _reject_link(path)
    if os.name == "nt":
        _windows_acl(base, protect=new)
    elif base.stat().st_uid != os.getuid() or stat.S_IMODE(base.stat().st_mode) != 0o700:
        raise ValueError(f"密钥目录必须由当前用户拥有且权限为 0700：{base}")
    created = not any(path.exists() for path in paths)
    if created:
        lock = base / ".initialize.lock"
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ValueError("密钥初始化正在进行") from error
        try:
            created = not any(path.exists() for path in paths)
            if created:
                create_keypair(base)
        finally:
            lock.rmdir()
    for path in paths:
        _reject_link(path)
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f"密钥材料缺失或不是独立文件：{path}")
        if os.name == "nt":
            _windows_acl(path, protect=created)
        elif path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022:
            raise ValueError(f"密钥文件所有者或权限不安全：{path}")
    state = inspect_keys(base)
    if not state["valid"]:
        raise ValueError("密钥验证失败：" + ", ".join(state["errors"]))
    return {"directory": str(base), "created": created, "fingerprint": state["fingerprint"]}


def initialize(repo_root=".", *, models=None, keys_dir=None):
    root = Path(repo_root).expanduser().resolve(strict=True)
    keys = provision_keys(root, keys_dir)
    template = root / ".env.luna.example"
    try:
        with template.open("x", encoding="utf-8") as stream:
            stream.write(_TEMPLATE)
    except FileExistsError:
        if not template.is_file() or template.is_symlink():
            raise ValueError("模型网关配置模板必须是普通文件")
    result = combat_check(root, models=models, keys_dir=keys["directory"])
    return result | {"keys": keys, "template": str(template),
                     "remediation": {name: _FIXES[name] for name in result["blockers"]}}
