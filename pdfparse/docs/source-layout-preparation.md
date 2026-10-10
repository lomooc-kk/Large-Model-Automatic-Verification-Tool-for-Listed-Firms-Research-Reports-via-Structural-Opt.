# 原始研报的可见文字与财务表准备

这两个模块是本地、显式调用的数据准备工具，默认解析引擎和核查入口没有切换到它们。它们输出待审候选及可追溯原文，不生成比赛金标、不证明研报财务数据正确，也不自动把来源池晋升为测试集。

## 调用

```python
import json
import pymupdf
from yjparse.pdf_visibility import audit_page_visibility
from yjparse.financial_table_layout import extract_financial_tables

with pymupdf.open("original-report.pdf") as document:
    page = document[1]
    visibility = audit_page_visibility(page)
    candidate = extract_financial_tables(page, visibility_audit=visibility)
    with open("page2.layout.json", "w", encoding="utf-8") as handle:
        json.dump(candidate, handle, ensure_ascii=False, indent=2)
```

页码元数据使用一基页码。原始坐标为未旋转的 PDF point；`display_bbox` 单独映射到显示坐标。页级可见性副本通过原生全文 hash、页码、旋转角度及实际渲染 hash 绑定，不能把另一个同文字、不同颜色的页面副本用于过滤。

## 可见性

`pdf_visibility.py` 结合文字绘制方式、透明度、字形区域的实际像素和后续覆盖操作，输出 `visible`、`invisible`、`uncertain`。它保留完整原文、trace 和字形记录，不删除源 PDF 或原始文本。

- 深底白字可以可见；白色字体本身不是删除依据。
- 确认白底同色、非绘制文字、零透明度或可靠完整遮挡时，才可判不可见。
- 低对比、复杂背景、局部遮挡及不支持的绘制方式保留不确定。
- 图片或轮廓形式的文字未出现在原生字形里时，这个工具无法恢复内容。`page_visual_completeness_certified` 始终为 false。

## 表格候选

`financial_table_layout.py` 使用独立财务表标题、原生文字坐标及规则年份列，分离并排和上下排列的表，再给行内数值绑定候选年份。年份组跨入另一个标题区域时切开；不满足支持条件的内容仍在 `raw_spans`、`raw_tokens` 和 `unassigned_token_ids` 中。

每个表保留 `table_id`、标题、标题单位、年份 token、行名、单元格坐标和原文。标题单位不自动作为每个比例或每股指标的单位。

- `blank`：该位置没有提取到 token；不能据此证明原件可见为空，更不能当零。
- `extracted`：保留字面数值，包括 `0`、负数及百分号；不验证业务真值。
- `unreadable_source_placeholder`：原文为 `#######` 等占位符，不通过计算恢复。
- `source_dash`：保留横线或斜线，不转为零。
- `ambiguous`：同列多项或超宽等冲突，原始内容保留，`display_text` 不作为确定单元格展示。

不可见文字保留 `raw_text`，`display_text` 为空。不可见或不确定表头/标题不授权可见年份绑定：`raw_year` 仍在，但 `display_year` 为空，`header_binding_status` 为 pending。`year` 是兼容的候选原始年份字段，消费方必须检查绑定状态。行级可见性状态仅说明字形，不认证布局；所有表级 `review_required` 均为 true。

## R9 验证范围

- 24 份来源、174 页只进行了本地候选结构扫描，不代表逐页目视质量通过。
- P03/P07 第2页的8张财务表分开保存；以开发前的视觉收据检查7行34个数值的表身份和年份对应。
- 另外检查基年空格不左移、3个 `#######` 不补值，以及不可见 ROIC 行保留原始记录但排除可见展示。
- P03可见标题漏抽与P07图表内容尚待视觉/OCR核对。当前两份也是开发样本，不是独立新测试成绩。

复现本次候选准备：在仓库根目录设置 `PYTHONPATH=pdfparse/src;factcheck/src;.`，运行 `output/new-competition-pilot-20261010/prepare_layout_candidates.py`。它检查已冻结的两份来源收据和原件 hash，另存结果到 `layout-candidates-v1`，不覆盖原始文本、旧金标或 active_suite。
