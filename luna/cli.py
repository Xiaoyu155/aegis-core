"""Chinese product command line for Luna's Aegis-backed workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .combat import combat_check
from .dashboard import create_dashboard_server
from .runner import run_requirement
from .status import format_status, status_snapshot


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="luna", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="用自然语言需求启动隔离交付流程")
    run.add_argument("text", help="产品需求")
    run.add_argument("--repo", default=".")
    run.add_argument("--models", help="模型路由配置；默认读取仓库 config/models.yaml")
    run.add_argument("--keys-dir", help="Ed25519 密钥目录")
    run.add_argument("--db", help="Aegis 证据账本路径")
    run.add_argument("--run-id", help="可复现的运行 ID；默认自动生成")
    run.add_argument("--timeout", type=float, default=20)
    run.add_argument("--max-rounds", type=int, default=3)
    run.add_argument("--demo", action="store_true",
                     help="运行内嵌库存沙盒：本地确定性候选 + 临时 Ed25519 密钥，不调用外部模型")
    status = commands.add_parser("status", help="查看项目健康度、待验收项和风险拦截")
    status.add_argument("--repo", default=".")
    status.add_argument("--json", action="store_true")
    dashboard = commands.add_parser("dashboard", help="启动回环地址上的 Luna 中文总控台")
    dashboard.add_argument("--repo", default=".")
    dashboard.add_argument("--port", type=int, default=8766)
    combat = commands.add_parser("combat-check", help="实战前检查密钥、Git、隔离、CI、模型网关和磁盘")
    combat.add_argument("--repo", default=".")
    combat.add_argument("--models")
    combat.add_argument("--keys-dir")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if not 1 <= args.max_rounds <= 8 or args.timeout <= 0:
                parser.error("--max-rounds must be 1..8 and --timeout must be positive")
            result = run_requirement(args.text, args.repo, models=args.models, keys_dir=args.keys_dir,
                                     db=args.db, run_id=args.run_id, timeout=args.timeout,
                                     max_rounds=args.max_rounds, demo=args.demo)
            print(f"Luna 运行：{result['status']} · {result['run_id']}")
            print(f"机器审计：{result['machine_log']}")
            print(f"SHA-256：{result['machine_log_sha256']}")
            print(f"产品战报：{result['report']}")
            if result.get("delivery"):
                print(f"交付分支：{result['delivery']['branch']} @ {result['delivery']['commit'][:12]}")
            if result.get("signing"):
                print(f"签名清单：{result['signing']['manifest_sha256']} · Ed25519 验签通过")
            if result.get("reason"):
                print("阻断原因：" + result["reason"])
            return 0 if result["status"] == "SUCCEEDED" else 2
        if args.command == "status":
            snapshot = status_snapshot(args.repo)
            print(_json(snapshot) if args.json else format_status(snapshot))
            return 0
        if args.command == "dashboard":
            server = create_dashboard_server(args.repo, port=args.port)
            print(f"Luna 总控台：http://{server.server_address[0]}:{server.server_address[1]}/", flush=True)
            try:
                server.serve_forever(poll_interval=0.25)
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0
        result = combat_check(args.repo, models=args.models, keys_dir=args.keys_dir)
        print(result["status"])
        print(_json(result))
        return 0 if result["status"] == "READY" else 2
    except (ValueError, OSError, RuntimeError) as error:
        print(_json({"status": "BLOCKED", "error_type": type(error).__name__,
                     "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
