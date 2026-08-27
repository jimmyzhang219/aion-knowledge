#!/usr/bin/env python3
"""
依赖一致性检查 — 扫描 src/ 下所有 import，与 pyproject.toml 声明的依赖交叉验证。

用法:
  python scripts/check-deps.py                     # 全量检查
  python scripts/check-deps.py --files f1.py f2.py # 只检查指定文件

退出码:
  0  — 所有 import 的包均已声明（或属于 stdlib / 项目自身）
  1  — 存在缺失声明
"""

import ast
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 已知的 import 名 → pip 包名 映射
IMPORT_TO_PIP = {
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "cv2": "opencv-contrib-python",  # 仓库内唯一使用方 paddle.py 走 paddle-ocr 组（paddlex 硬钉 contrib）
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "docx": "python-docx",
    "pypdfium2": "pypdfium2",
    "markitdown": "markitdown",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyopenssl",
    "zmq": "pyzmq",
    "umap": "umap-learn",}

# 已知 pip 包名 → import 名（用于归一化）
PIP_TO_IMPORT = {}
for k, v in IMPORT_TO_PIP.items():
    PIP_TO_IMPORT.setdefault(v, []).append(k)

# Python 标准库模块（Python 3.10+ 支持 sys.stdlib_module_names）
_STDLIB_MODULES: set[str] | None = None


def _get_stdlib() -> set[str]:
    global _STDLIB_MODULES
    if _STDLIB_MODULES is not None:
        return _STDLIB_MODULES
    try:
        _STDLIB_MODULES = set(sys.stdlib_module_names)
    except AttributeError:
        # Python < 3.10 fallback
        _STDLIB_MODULES = {
            "abc", "argparse", "ast", "asyncio", "base64", "binascii",
            "collections", "concurrent", "configparser", "contextlib", "copy",
            "csv", "dataclasses", "datetime", "decimal", "email", "enum",
            "functools", "glob", "gzip", "hashlib", "html", "http",
            "importlib", "inspect", "io", "ipaddress", "itertools", "json",
            "logging", "math", "mimetypes", "multiprocessing", "operator",
            "os", "pathlib", "pickle", "pkgutil", "platform", "posixpath",
            "pprint", "queue", "random", "re", "secrets", "shutil", "signal",
            "socket", "sqlite3", "ssl", "statistics", "string", "struct",
            "subprocess", "sys", "tempfile", "textwrap", "threading", "time",
            "tomllib", "traceback", "typing", "unicodedata", "unittest",
            "urllib", "uuid", "warnings", "weakref", "xml", "zipfile",
        }
    _STDLIB_MODULES.add("__future__")
    return _STDLIB_MODULES


def parse_pyproject() -> dict[str, set[str]]:
    """解析 pyproject.toml，返回 {类别: {pip包名}}."""
    result: dict[str, set[str]] = {"main": set(), "optional": set()}

    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        print("❌ pyproject.toml 不存在", file=sys.stderr)
        sys.exit(1)

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    for dep in data.get("project", {}).get("dependencies", []):
        name = _extract_pip_name(dep)
        if name:
            result["main"].add(name)

    for _, deps in data.get("project", {}).get("optional-dependencies", {}).items():
        for dep in deps:
            name = _extract_pip_name(dep)
            if name:
                result["optional"].add(name)

    return result


def _extract_pip_name(dep_spec: str) -> str:
    """从 'pkg>=1.0,<2.0' 中提取纯包名."""
    name = dep_spec.strip()
    # 去掉 extras: pkg[extra] >= 1.0
    for sep in ("[", ">=", "<=", "!=", "~=", "==", ">", "<", "@"):
        idx = name.find(sep)
        if idx != -1:
            name = name[:idx]
    return name.strip().lower()


def pip_to_import_name(pip_name: str) -> str:
    """将 pip 包名转为可能的 import 名（去连字符）。"""
    return pip_name.lower().replace("-", "_").replace(".", "_")


def extract_imports(file_path: Path) -> set[str]:
    """从 Python 文件中提取所有第三方 import 的顶级包名。"""
    imports: set[str] = set()
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                imports.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level in (None, 0):
                top = node.module.split(".")[0]
                imports.add(top)
    return imports


def check(files: list[Path] | None = None) -> int:
    """执行检查，返回退出码。"""
    declared = parse_pyproject()
    declared_pip = declared["main"] | declared["optional"]

    # 将声明过的 pip 包名转成 import 名（可能有多个映射）
    declared_imports: set[str] = set()
    for pip_name in declared_pip:
        # 直接转换
        declared_imports.add(pip_to_import_name(pip_name))
        # 已知别名
        for alias in PIP_TO_IMPORT.get(pip_name, []):
            declared_imports.add(alias.lower())

    # 收集所有 import
    imported: dict[str, set[Path]] = {}
    if files:
        py_files = [PROJECT_ROOT / f for f in files if f.suffix == ".py"]
    else:
        src_dir = PROJECT_ROOT / "src"
        py_files = list(src_dir.rglob("*.py")) if src_dir.exists() else []

    for py_file in py_files:
        for mod in extract_imports(py_file):
            mod_lower = mod.lower()
            # 跳过 stdlib
            if mod_lower in _get_stdlib():
                continue
            # 跳过项目自身
            if mod_lower.startswith("aion_knowledge") or mod_lower.startswith("tests"):
                continue
            imported.setdefault(mod_lower, set()).add(py_file)

    # 检查缺失
    missing: dict[str, list[Path]] = {}
    for mod_name, sources in imported.items():
        if mod_name not in declared_imports:
            # 尝试反向映射到 pip 名
            pip_name = IMPORT_TO_PIP.get(mod_name, mod_name.replace("_", "-"))
            missing[pip_name] = sorted(sources)

    # 输出
    if not missing:
        print("✅ 所有 import 的包均已声明")
        return 0

    print(f"⚠️  以下 {len(missing)} 个包被 import 但未在 pyproject.toml 声明：\n")
    for pkg_name in sorted(missing):
        sources = missing[pkg_name]
        if len(sources) <= 3:
            locs = ", ".join(str(s.relative_to(PROJECT_ROOT)) for s in sources)
        else:
            locs = f"{str(sources[0].relative_to(PROJECT_ROOT))} 等 {len(sources)} 处"
        print(f"  - {pkg_name}  ←  {locs}")

    print()
    print("建议: 将以上包添加到 pyproject.toml 的 dependencies 列表中。")
    print("       注意：部分包名可能需要映射（如 PIL → pillow）。")
    return 1


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="检查依赖声明一致性")
    parser.add_argument("--files", nargs="*", type=Path, help="只检查指定文件")
    args = parser.parse_args()

    sys.exit(check(args.files))


if __name__ == "__main__":
    main()
