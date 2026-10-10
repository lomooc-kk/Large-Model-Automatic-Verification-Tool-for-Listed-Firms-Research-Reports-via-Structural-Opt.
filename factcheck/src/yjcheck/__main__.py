from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile
from xml.etree.ElementTree import ParseError
from .pipeline import run_check, verify_artifacts


def main(argv=None):
    parser=argparse.ArgumentParser(description="C：带证据的研报数值、单位、期间、口径与引用核查")
    commands=parser.add_subparsers(dest="command",required=True)
    check=commands.add_parser("check")
    check.add_argument("--report",required=True)
    check.add_argument("--source",required=True,action="append")
    check.add_argument("--out",default="data/check")
    check.add_argument("--company")
    check.add_argument("--engine",choices=["pdfplumber","pymupdf"],default="pdfplumber")
    check.add_argument("--model",action="store_true",help="使用 YJCHECK_* 环境变量配置的模型辅助抽取")
    check.add_argument("--review-text", action="store_true", help="同时运行模型文本检测，费用计入共享预算")
    verify=commands.add_parser("verify")
    verify.add_argument("directory")
    for name in ("agent-check", "agent-text"):
        agent = commands.add_parser(name, help="Pi 调度检测与 OpenViking 补证")
        agent.add_argument("--report" if name == "agent-check" else "--input", required=True)
        if name == "agent-check":
            agent.add_argument("--source", action="append", default=[])
            agent.add_argument("--engine", choices=["pdfplumber", "pymupdf"], default="pdfplumber")
            agent.add_argument("--model", action="store_true", help="配对流程也使用模型抽取候选")
        agent.add_argument("--out", default="data/agent-runs")
        agent.add_argument("--kb", help="B export-kb 的产物目录")
        agent.add_argument("--company")
        agent.add_argument("--allow-fallback", action="store_true", help="显式允许 BM25/本地原文降级并留痕")
        agent.add_argument("--max-rounds", type=int, default=6)
        agent.add_argument("--agent-output-tokens", type=int, default=4096, help="Pi 调度单轮输出上限，独立于正文检测配置")
        agent.add_argument("--max-agent-cost", default="2", help="Pi 调度调用费用上限，仍受累计账本约束")
    kb = commands.add_parser("kb", help="OpenViking 服务、入库和检索")
    kb.add_argument("action", choices=["health", "ingest", "search"])
    kb.add_argument("--directory")
    kb.add_argument("--query")
    kb.add_argument("--allow-fallback", action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.command == "kb":
            from yjparse.openviking import health, ingest_kb, search_kb
            if args.action == "health":
                response = health()
            elif not args.directory:
                raise ValueError("ingest/search require --directory")
            elif args.action == "ingest":
                response = ingest_kb(Path(args.directory))
            else:
                if not args.query:
                    raise ValueError("search requires --query")
                response = search_kb(Path(args.directory), args.query, allow_fallback=args.allow_fallback)
            print(json.dumps(response, ensure_ascii=False))
            return 0 if response.get("status") in {"ok", "ready", "completed"} else 2
        if args.command in {"agent-check", "agent-text"}:
            from .agent_workflow import run_agent_check, run_agent_text
            from .model import ModelConfig
            from .pi_bridge import AgentLimits
            config = ModelConfig.from_env()
            config.review_text = getattr(args, "model", False)
            kwargs = dict(model_config=config, out=args.out, kb_dir=args.kb, company=args.company,
                          allow_fallback=args.allow_fallback,
                          limits=AgentLimits(max_rounds=args.max_rounds, max_cost_cny=args.max_agent_cost,
                                             max_output_tokens=args.agent_output_tokens))
            if args.command == "agent-check":
                result, path = run_agent_check(args.report, args.source, engine=args.engine,
                                               detect_with_model=args.model, **kwargs)
            else:
                content = Path(args.input).read_text(encoding="utf-8-sig")
                result, path = run_agent_text(content, document_id="document", **kwargs)
            runtime = result["agent_runtime"]
            print(json.dumps({"output": str(path.resolve()), "agent_runtime": runtime,
                              "summary": result.get("summary", result.get("coverage"))}, ensure_ascii=False))
            return 0 if runtime.get("status") in {"completed", "needs_review"} else 2
        if args.command=="verify":
            passed=verify_artifacts(args.directory)
            print("verified" if passed else "verification_failed")
            return 0 if passed else 1
        config=None
        if args.model or args.review_text:
            from .model import ModelConfig
            config=ModelConfig.from_env()
            config.review_text = args.review_text
        result,path=run_check(args.report,args.source,args.out,args.company,args.engine,config)
        print(json.dumps({"output":str(path.resolve()),"summary":result["summary"]},ensure_ascii=False))
        return 0  # 业务发现错误也是成功执行；调用方看 summary，不以退出码推断研报正确。
    except (ValueError,OSError,KeyError,ImportError,TypeError,AttributeError,RuntimeError,BadZipFile,ParseError) as exc:
        print(f"核查未完成：{exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
