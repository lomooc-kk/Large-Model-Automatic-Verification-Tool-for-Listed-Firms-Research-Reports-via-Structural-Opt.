"""Single-document text review; raw text locations are not PDF page numbers."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import time
import uuid
from pathlib import Path
import streamlit as st
from yjcheck.model import ModelConfig
from yjcheck.model_runtime import BudgetedChatClient
from yjcheck.text_review import detect_text
from yjcheck.review_hints import all_review_hints
from review_store import ReviewStore, REVIEW_STATUSES, REVIEW_LABELS


def _review_controls(store, document_id, identity, key_suffix=""):
    if store is None:
        return
    current = store.get(identity)
    key = "text_review_" + document_id + identity + key_suffix
    selected = st.selectbox("复核结论", REVIEW_STATUSES, index=REVIEW_STATUSES.index(current.get("status", "unreviewed")), format_func=lambda s: REVIEW_LABELS[s], key=key)
    reviewer = st.text_input("复核人", current.get("reviewer", ""), key=key+"_reviewer")
    note = st.text_input("复核依据", current.get("note", ""), key=key+"_note")
    if st.button("开始计时", key=key+"_start"):
        st.session_state[key+"_time"] = time.monotonic()
    if st.button("保存文本复核", key=key+"_save"):
        started = st.session_state.pop(key+"_time", None)
        store.set(identity, selected, note, reviewer, duration_seconds=time.monotonic()-started if started is not None else None)
        st.success("复核意见已单独保存，原始检测结果保持不变。")


PRIORITY_LABELS = {"confirmed": "已确认", "high": "高优先", "low": "低优先"}
PRIORITY_ORDER = {"confirmed": 0, "high": 1, "low": 2}


def show_text_report(report, check_dir=None):
    errors = report.get("errors", [])
    hints = all_review_hints(report)["errors"]
    rejected_hints = [e for e in hints if e.get("invalid_anchor")]
    coverage = report.get("coverage", {})
    agent = report.get("agent_runtime", {})
    if agent.get("status") in {"failed", "stopped"} or coverage.get("agent_complete") is False:
        st.warning("Pi 补证流程未完整结束；以下是已完成的部分结果。原因：" + str(agent.get("stop_reason", "unknown")))
    confirmed = sum(e.get("status") == "confirmed_error" for e in errors)
    store = ReviewStore(check_dir) if check_dir else None
    cols = st.columns(4)
    cols[0].metric("检测候选", len(errors))
    cols[1].metric("经规则确认", confirmed)
    cols[2].metric("待复核提示（含未定位）", sum(e.get("status", "needs_review") == "needs_review" for e in hints))
    cols[3].metric("未通过定位或格式校验", len(rejected_hints))
    if rejected_hints:
        st.caption("复核总量包含无法定位或格式无效的原始提示，同一问题的重复提示仍计入工作量；直接检测中已列出的无效候选不重复计数。")
    if not coverage.get("complete"):
        st.warning("本次全文请求已完成，但部分候选仍需定位或格式复核。" if coverage.get("execution_complete") else "本次检测未完成全部处理，请查看处理范围与原因。")
    if not coverage.get("model_ran"):
        st.caption("本次未运行模型，仅展示离线规则能力。")
    if not errors:
        st.info("本次未输出已定位的错误候选；这不代表全文内容已经证实正确。")
    ordered = sorted(errors, key=lambda e: (PRIORITY_ORDER.get(e.get("review_priority"), 3),
                                            -(e.get("spans") or [{}])[0].get("start") if e.get("spans") else 0))
    for error in ordered:
        if error.get("invalid_anchor"):
            continue
        label = "已确认错误" if error.get("status") == "confirmed_error" else "待人工确认"
        priority = PRIORITY_LABELS.get(error.get("review_priority"))
        caption = f"{label} · {error.get('error_type', '')}"
        if priority and error.get("status") != "confirmed_error":
            caption += f" · 复核优先级：{priority}"
        with st.expander(caption):
            for span in error.get("spans", []):
                st.text(span.get("text", ""))
                st.caption(f"原文字符区间 [{span.get('start')}, {span.get('end')})")
            st.write(error.get("reason", ""))
            if error.get("evidence"):
                st.json(error["evidence"], expanded=False)
            _review_controls(store, report["document_id"], "text:" + error["id"])
    if rejected_hints:
        with st.expander("未定位或格式无效的提示：原始内容与拒绝原因"):
            st.caption("以下引用是模型原始返回，尚未确认能对应原文；不作为已定位证据或已确认错误。")
            for index, hint in enumerate(rejected_hints):
                st.text(f"提示 {index + 1} · {hint.get('error_type', '未知类型')}")
                original = hint.get("original_spans")
                for span in original if isinstance(original, list) else [original]:
                    st.text(str(span.get("text", "")) if isinstance(span, dict) else str(span))
                st.text("模型理由：" + str(hint.get("reason", "")))
                st.text("拒绝原因：" + str(hint.get("rejection_reason", "未能验证原文位置")))
                identity = "text:" + (hint.get("id") or "rejected:" + hashlib.sha256(
                    json.dumps([index, hint], ensure_ascii=False, sort_keys=True).encode()).hexdigest())
                _review_controls(store, report["document_id"], identity, "_rejected")
    redundancy_review = report.get("redundancy_review")
    if isinstance(redundancy_review, dict):
        with st.expander("冗余复核：保留、调整与撤回依据"):
            st.caption("这是模型结合原文提出的复核意见；撤回的原提示及其依据保留在这里，可继续人工复核。")
            if not redundancy_review.get("complete"):
                st.warning("冗余复核未完成，已保留该阶段之前的全部提示。")
            actions = {"retain": "保留", "revise": "调整引用", "withdraw": "撤回提示"}
            for record in redundancy_review.get("decisions", []):
                decision = record.get("decision", {})
                st.write(actions.get(record.get("action"), "待复核") + "：" + str(decision.get("reason", "")))
                for span in record.get("evidence", []):
                    st.text(span.get("text", ""))
                    st.caption(f"原文字符区间 [{span.get('start')}, {span.get('end')})")
                st.json({"原提示": record.get("before"), "复核后": record.get("after")}, expanded=False)
                _review_controls(store, report["document_id"], "text:" + record["error_id"], "_redundancy")
    if store is not None:
        with st.expander("人工复核统计（按复核人计时）"):
            st.json(store.timing_report())
            buffer = io.StringIO()
            fields = ["finding_id", "status", "reviewer", "note", "duration_seconds",
                      "total_duration_seconds", "timing_method", "updated_at"]
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{key: row.get(key, "") for key in fields} for row in store.export_rows()])
            st.download_button("下载复核记录 CSV", "\ufeff" + buffer.getvalue(),
                               file_name="review_records.csv", mime="text/csv", key="download_review_csv")
    with st.expander("处理范围与运行记录"):
        st.json(coverage)
        st.json(report.get("traces", []), expanded=False)
    exported = {**report, "human_reviews": {key: value for key, value in store.load().items() if key.startswith("text:")}} if store else report
    st.download_button("下载文本检测结果", json.dumps(exported, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="text_review.json", mime="application/json", key="download_text_review_" + report.get("document_id", ""))


def text_review_page():
    st.caption("检查单份文本的数值、时间及前后矛盾等问题；原文位置按字符记录。")
    upload = st.file_uploader("上传文本或 Markdown", type=["txt", "md"], key="review_text_file")
    pasted = st.text_area("或粘贴待检查正文", height=240, key="review_text_input")
    content = upload.getvalue().decode("utf-8-sig") if upload else pasted
    use_model = st.checkbox("启用模型检测", key="text_review_use_model", help="使用已有模型接口，费用计入共享预算账本。")
    use_pi = st.checkbox("Pi 自动补证", key="text_review_use_pi",
                         help="由 Pi 调度模型检测及证据检索；外部事实核验另列，不改变文本检测评分口径。")
    kb_dir = st.text_input("证据库目录", key="text_agent_kb_dir") if use_pi else ""
    if st.button("检查文本", disabled=not content.strip(), type="primary"):
        try:
            client = BudgetedChatClient(ModelConfig.from_env()) if use_model and not use_pi else None
            # Historical FinED samples are diagnostic-only. New CLFEC correction
            # examples have a different contract and must not be silently mapped
            # into the fifteen-label detector used by this page.
            examples = []
            if use_pi:
                from yjcheck.agent_workflow import run_agent_text
                report, _ = run_agent_text(content, document_id="text-" + hashlib.sha256(content.encode()).hexdigest(),
                                           model_config=ModelConfig.from_env(), kb_dir=kb_dir.strip() or None,
                                           examples=examples)
            else:
                report = detect_text(content, document_id="text-" + hashlib.sha256(content.encode()).hexdigest(),
                                     detector="hybrid", chat=client, examples=examples,
                                     max_input_tokens=client.settings.context_tokens - client.settings.max_output_tokens if client else 16000)
            st.session_state["text_review_result"] = report
            directory = Path(__file__).resolve().parent / "data/text_review" / uuid.uuid4().hex
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "text_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state["text_review_dir"] = str(directory)
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f"文本检测未完成：{exc}")
    if st.session_state.get("text_review_result"):
        current = st.session_state["text_review_result"]
        if current.get("agent_runtime"):
            with st.expander("Pi 补证过程"):
                st.json(current["agent_runtime"])
        if current.get("external_fact_check"):
            with st.expander("外部证据事实核验（独立结果）"):
                st.json(current["external_fact_check"])
        show_text_report(st.session_state["text_review_result"], st.session_state.get("text_review_dir"))
