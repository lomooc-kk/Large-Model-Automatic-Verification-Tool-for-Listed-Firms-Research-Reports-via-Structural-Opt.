"""Source-bound arithmetic aids for the detector, never standalone verdicts.

For an explicitly stated current price and absolute period change, the base of
the percentage is the previous price (current minus change). Only a narrow,
fully parsed, same-unit construction is handled. Display-rounding intervals
prevent point estimates from manufacturing contradictions. No benchmark labels,
external prices, document identifiers or learned corrections are read here.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import re

_N = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT = r"(?:美元|港元|人民币元|元)\s*[/／]\s*(?:吨|桶|公斤|千克|股)"
_CHANGE = re.compile(
    rf"(?:价格|均价|报价|收盘价|结算价)\s*(?:为|报|达到|录得)\s*"
    rf"(?P<current>{_N})\s*(?P<current_unit>{_UNIT})\s*[，,]\s*"
    rf"(?:环比|较)\s*(?:\d{{1,2}}月\d{{1,2}}日|上期|前期|上周|上月|昨日)?\s*"
    rf"(?P<direction>上涨|下跌|上升|下降|增加|减少|[+＋\-−－])\s*"
    rf"(?P<change>{_N})\s*(?P<change_unit>{_UNIT})\s*[，,]\s*"
    rf"(?:环比)?(?P<rate_label>涨跌幅|涨幅|跌幅|降幅|变动幅度|变化幅度)\s*(?:为|达)?\s*"
    rf"(?P<rate>[+＋\-−－]?{_N})\s*[%％](?![\d.%％])"
)
_NEGATIVE = {"下跌", "下降", "减少", "-", "−", "－"}
_QUANTITY_UNIT = r"(?:亿方|万方|万吨|吨|万千瓦|千瓦|亿元|万元|元|万辆|辆|万套|套)"
_INVENTORY_CHANGE = re.compile(
    rf"库存\s*(?:为|达|达到)?\s*(?P<current>{_N})\s*(?P<unit>{_QUANTITY_UNIT})\s*[，,]\s*"
    rf"(?P<period>周|月|日)?环比\s*(?P<direction>增加|减少|上升|下降)\s*"
    rf"(?P<change>{_N})\s*(?P<change_unit>{_QUANTITY_UNIT})\s*[（(]\s*"
    rf"(?P<rate>[+＋\-−－]?{_N})\s*[%％]\s*[）)]")
_TOTAL_PARTS = re.compile(
    rf"(?P<label>总新增装机|总装机|总供应量|总产量|总销量|总额|总计|合计)\s*(?:为|达|达到)?\s*"
    rf"(?P<total>{_N})\s*(?P<unit>{_QUANTITY_UNIT})\s*[，,]\s*其中"
    rf"(?P<parts>[^。；;\n]{{1,500}})(?=[。；;\n]|$)")
_PART = re.compile(rf"(?P<label>[\u4e00-\u9fff]{{2,24}}?)\s*(?:为|达|达到)?\s*(?P<value>{_N})\s*(?P<unit>{_QUANTITY_UNIT})")
_EPS = rf"{_N}\s*元"
_PE = rf"{_N}\s*倍"
_SEPARATOR = r"\s*(?:、|，|,|和|及|/|／)\s*"
_VALUATION = re.compile(
    rf"(?:EPS|每股收益)\s*(?:分别)?(?:为|是)?\s*(?P<eps>{_EPS}(?:{_SEPARATOR}{_EPS}){{1,7}})\s*[，,]\s*"
    rf"(?:参照|按|以)(?P<price_context>[^。；;\n]{{0,35}}?(?:收盘价|股价))\s*(?P<price>{_N})\s*元\s*(?:计算)?[，,]\s*"
    rf"对应(?:的)?\s*(?:PE|市盈率)\s*(?:分别)?(?:为|是)?\s*(?P<pe>{_PE}(?:{_SEPARATOR}{_PE}){{1,7}})(?=[，,。；;\n]|$)", re.I)

# Deliberately narrower than generic number extraction: only money amounts,
# an explicit total, and an explicit part/share relation. Displayed counts,
# growth rates and forecasts must never become operands by proximity.
_SHARE_MONEY = r"(?:亿|万)?(?:美元|港元|人民币元|元)"
_SHARE_METRIC = r"(?:营业收入|销售收入|销售额|营收|收入|金额)"
_SHARE_TOTAL = re.compile(
    rf"(?P<metric>(?:整体|总){_SHARE_METRIC}|总额|合计金额)\s*"
    rf"(?:达到|录得|为|达)?\s*(?P<value>{_N})\s*(?P<unit>{_SHARE_MONEY})"
    r"(?![\w/／])")
_SHARE_FORECAST = re.compile(
    rf"(?:市场)?(?:预期|预计|预测)\s*(?:为)?\s*(?P<value>{_N})\s*"
    rf"(?P<unit>{_SHARE_MONEY})")
_SHARE_PART = re.compile(
    rf"其中(?P<label>[\w\u4e00-\u9fff \t]{{1,40}}?)(?P<metric>{_SHARE_METRIC})\s*"
    rf"(?:(?P<growth>(?:同比|环比)(?:增长|增加|上升|下降|减少|下滑|增|降)?\s*"
    rf"[+＋\-−－]?{_N}\s*[%％])\s*[，,]\s*)?"
    rf"(?:达到|录得|为|达)?\s*(?P<value>{_N})\s*(?P<unit>{_SHARE_MONEY})\s*"
    r"(?:[（(](?P<forecast>[^（）()\r\n]{1,60})[）)]\s*)?[，,]\s*"
    rf"占\s*(?P<share_metric>(?:整体|总)?{_SHARE_METRIC}|总额)\s*(?:的|比例为|比例|比为)?\s*"
    rf"(?P<rate>{_N})\s*[%％](?![\d.%％\-~～至到])")
_SHARE_UNCERTAIN = re.compile(
    r"预期|预计|预测|预估|指引|目标|约|近|超|逾|至少|至多|不足|不低于|不高于|"
    r"精确|未舍入|未四舍五入|截尾|截断|四舍五入后|含税|不含税|税后|净额|净收入|扣非|调整后")
_SHARE_PERIOD = re.compile(
    r"(?:FY\s*\d{2,4}|20\d{2}年?|Q[1-4]|第?[一二三四1-4]季度|[上下]半年|"
    r"去年|上年|前年|今年|本年|本期|上期|同期|\d{1,2}月)", re.I)
_SHARE_ENTITY_SWITCH = re.compile(
    r"另一|其他公司|竞争对手|(?:[A-Za-z\u4e00-\u9fff]{1,24}(?:公司|集团|银行))")
_SHARE_MAX_DISTANCE = 800


def _share_prefix(paragraph: str, start: int) -> str:
    """Keep qualifiers attached to a total in its own source clause."""
    prefix = paragraph[:start]
    boundary = max((prefix.rfind(mark) for mark in '。；;，,\n：:'), default=-1)
    local = prefix[boundary + 1:]
    # Also reject a forecast introduced immediately before a comma/colon.
    if re.search(r'(?:预计|预期|预测|指引|目标)[，,:：][^。；;，,:：]{0,40}$', prefix):
        return prefix[-80:]
    return local


def _share_metric_family(metric: str) -> str:
    return 'amount' if metric in {'金额', '总金额', '整体金额', '总额', '合计金额'} else 'revenue'


def _number(raw: str) -> Decimal:
    return Decimal(raw.replace(",", "").replace("＋", "+").replace("−", "-").replace("－", "-"))


def _interval(raw: str) -> tuple[Decimal, Decimal]:
    value = _number(raw)
    half_step = Decimal(10) ** value.as_tuple().exponent / 2
    return value - half_step, value + half_step


def price_change_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    """Return checkable intermediates, with global exact-source locations.

    Closed interval endpoints deliberately admit the rounding boundary; this
    aid may abstain on a true discrepancy but must not invent one from a tie.
    ``rounding_compatible`` refers only to the arithmetic relation, not the
    correctness, provenance, or completeness of the source statement.
    """
    if not isinstance(text, str) or type(offset) is not int or offset < 0:
        raise ValueError("text and nonnegative integer source offset required")
    if type(limit) is not int or limit < 0:
        raise ValueError("nonnegative integer limit required")
    checks = []
    for match in _CHANGE.finditer(text):
        if len(checks) >= limit:
            break
        clean_unit = lambda raw: re.sub(r"\s+", "", raw).replace("／", "/")
        unit = clean_unit(match["current_unit"])
        if unit != clean_unit(match["change_unit"]):
            continue
        # Excessively long numeric strings are neither necessary nor safe for
        # this bounded helper; leave them to the regular review path.
        if any(len(match[name]) > 32 for name in ("current", "change", "rate")):
            continue
        with localcontext() as ctx:
            ctx.prec = 40
            current = _number(match["current"])
            delta = _number(match["change"])
            clo, chi = _interval(match["current"])
            dlo, dhi = _interval(match["change"])
            if match["direction"] in _NEGATIVE:
                delta = -delta
                dlo, dhi = -dhi, -dlo
            previous = current - delta
            if clo <= 0 or clo - dhi <= 0:
                continue
            rate = _number(match["rate"])
            rlo, rhi = _interval(match["rate"])
            if match["rate_label"] in {"跌幅", "降幅"} and rate >= 0:
                rate = -rate
                rlo, rhi = -rhi, -rlo
            corners = [d / (c - d) * 100 for c in (clo, chi) for d in (dlo, dhi)]
            low, high = min(corners), max(corners)
            compatible = max(low, rlo) <= min(high, rhi)
            checks.append({
                "kind": "price_period_change",
                "scope": "calculation_only_not_business_verdict",
                "source": {"start": offset + match.start(), "end": offset + match.end(), "text": match.group()},
                "unit": unit,
                "current_price": str(current),
                "signed_absolute_change": str(delta),
                "derived_previous_price": str(previous),
                "formula": "previous=current-change; percent=change/previous*100",
                "point_change_percent": str(delta / previous * 100),
                "possible_change_percent": [str(low), str(high)],
                "reported_signed_change_percent": str(rate),
                "reported_rounding_interval": [str(rlo), str(rhi)],
                "rounding_compatible": compatible,
                "assumptions": ["same_source_clause_and_unit", "round_to_last_displayed_digit"],
            })
    return checks


def inventory_change_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    """Explicit inventory and its own absolute/relative period change only."""
    result = []
    for match in _INVENTORY_CHANGE.finditer(text):
        if len(result) >= limit:
            break
        if match['unit'] != match['change_unit'] or any(len(match[n]) > 32 for n in ('current','change','rate')):
            continue
        with localcontext() as ctx:
            ctx.prec = 40
            current, delta = _number(match['current']), _number(match['change'])
            clo, chi = _interval(match['current'])
            dlo, dhi = _interval(match['change'])
            rate = _number(match['rate'])
            rlo, rhi = _interval(match['rate'])
            if match['direction'] in _NEGATIVE:
                delta, dlo, dhi = -delta, -dhi, -dlo
                if rate >= 0:
                    rate, rlo, rhi = -rate, -rhi, -rlo
            if clo <= 0 or clo-dhi <= 0:
                continue
            values = [d/(c-d)*100 for c in (clo,chi) for d in (dlo,dhi)]
            low, high = min(values), max(values)
            result.append({'kind':'inventory_period_change','scope':'calculation_only_not_business_verdict',
                'source':{'start':offset+match.start(),'end':offset+match.end(),'text':match.group()},
                'unit':match['unit'],'current_quantity':str(current),'signed_absolute_change':str(delta),
                'derived_previous_quantity':str(current-delta),'formula':'previous=current-change; percent=change/previous*100',
                'point_change_percent':str(delta/(current-delta)*100),'possible_change_percent':[str(low),str(high)],
                'reported_signed_change_percent':str(rate),'reported_rounding_interval':[str(rlo),str(rhi)],
                'rounding_compatible':max(low,rlo)<=min(high,rhi),
                'assumptions':['same_source_clause_and_unit','parenthetic_percent_is_this_change','round_to_last_displayed_digit']})
    return result


def total_component_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    """Conditional arithmetic; '其中' never proves an exhaustive partition."""
    result = []
    for match in _TOTAL_PARTS.finditer(text):
        if len(result) >= limit:
            break
        parts = list(_PART.finditer(match['parts']))
        cursor, valid = 0, True
        for part in parts:
            between = match['parts'][cursor:part.start()]
            if (cursor == 0 and between.strip()) or (cursor and not re.fullmatch(_SEPARATOR,between)):
                valid = False
            cursor = part.end()
        if (not valid or match['parts'][cursor:].strip() or not 2 <= len(parts) <= 8
                or any(p['unit'] != match['unit'] for p in parts)
                or len({p['label'] for p in parts}) != len(parts)
                or any(re.search(r'(?:约|约合|近|超(?:过)?|逾|至少|至多|不足|不少于|不超过)(?:为|达|达到)?$',p['label'])
                       or re.search(r'亏损|损失|扣减|抵扣|冲减|减去|负',p['label']) for p in parts)
                or any(len(value)>32 for value in [match['total'],*[p['value'] for p in parts]])):
            continue
        with localcontext() as ctx:
            ctx.prec = 40
            intervals = [_interval(p['value']) for p in parts]
            low, high = sum(i[0] for i in intervals), sum(i[1] for i in intervals)
            tlo, thi = _interval(match['total'])
            result.append({'kind':'conditional_component_sum','scope':'calculation_only_not_business_verdict',
                'source':{'start':offset+match.start(),'end':offset+match.end(),'text':match.group()},
                'unit':match['unit'],'reported_total':match['total'],
                'components':[{'label':p['label'],'value':p['value']} for p in parts],
                'component_sum':str(sum(_number(p['value']) for p in parts)),
                'possible_component_sum':[str(low),str(high)],'reported_total_rounding_interval':[str(tlo),str(thi)],
                'rounding_compatible':max(low,tlo)<=min(high,thi),
                'relation_proven':False,'exhaustive_partition_proven':False,
                'assumptions':['same_unit','round_to_last_displayed_digit'],
                'required_business_checks':['same_object_and_period','mutually_exclusive_components','exhaustiveness_not_implied_by_其中'],
                'interpretation':'Overlapping intervals cannot prove a sum error. Non-overlap alone is not a business verdict; partition/object/period still require proof.'})
    return result


def valuation_rounding_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    """Compare one explicit share price with paired displayed EPS and PE."""
    result = []
    for match in _VALUATION.finditer(text):
        if len(result) >= limit:
            break
        eps, pe = re.findall(_N,match['eps']), re.findall(_N,match['pe'])
        if len(eps) != len(pe) or any(len(v)>32 for v in [match['price'],*eps,*pe]):
            continue
        with localcontext() as ctx:
            ctx.prec = 40
            plo, phi = _interval(match['price'])
            pairs = []
            for e, p in zip(eps,pe):
                elo,ehi = _interval(e)
                vlo,vhi = _interval(p)
                if min(elo,vlo,plo)<=0:
                    break
                low,high = elo*vlo,ehi*vhi
                pairs.append({'eps':e,'pe':p,'point_implied_price':str(_number(e)*_number(p)),
                              'possible_implied_price':[str(low),str(high)],'rounding_compatible':max(low,plo)<=min(high,phi)})
            if len(pairs) != len(eps):
                continue
            result.append({'kind':'explicit_price_eps_pe','scope':'calculation_only_not_business_verdict',
                'source':{'start':offset+match.start(),'end':offset+match.end(),'text':match.group()},
                'share_price':match['price'],'price_rounding_interval':[str(plo),str(phi)],'pairs':pairs,
                'formula':'price=EPS*PE','rounding_compatible':all(p['rounding_compatible'] for p in pairs),
                'assumptions':['explicit_same_price_and_corresponding_sequences','same_currency_yuan','round_to_last_displayed_digit'],
                'required_business_checks':['EPS_definition_matches_PE_denominator','forward_period_alignment'],
                'interpretation':'Displayed EPS/PE may use unrounded internal values; do not flag their point products alone.'})
    return result


def rounded_share_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    """Conditional displayed-money part/total ratios, never business verdicts.

    A part must explicitly say what it is a share of. Exactly one actual total
    must precede it within 800 characters in the same paragraph. Sentence
    boundaries are allowed, but an intervening explicit period/entity switch
    is not. Revenue/sales aliases remain a stated business assumption.
    """
    if not isinstance(text, str) or type(offset) is not int or offset < 0:
        raise ValueError('text and nonnegative integer source offset required')
    if type(limit) is not int or limit < 0:
        raise ValueError('nonnegative integer limit required')
    result = []
    for block in re.finditer(r'\S[\s\S]*?(?=\r?\n[ \t]*\r?\n|\Z)', text):
        paragraph = block.group()
        for part in _SHARE_PART.finditer(paragraph):
            if len(result) >= limit:
                return result
            if (_SHARE_UNCERTAIN.search(part['label'] + part['metric'])
                    or _SHARE_PERIOD.search(part['label'])
                    or _SHARE_ENTITY_SWITCH.search(part['label'])
                    or re.match(r'\s*[-~～至到]', paragraph[part.end():])):
                continue
            forecast = _SHARE_FORECAST.fullmatch(part['forecast']) if part['forecast'] else None
            if part['forecast'] and not forecast:
                continue
            # Do not choose the nearest of several totals, even if one unit
            # happens to agree. Ambiguity is not arithmetic evidence.
            totals = [candidate for candidate in _SHARE_TOTAL.finditer(
                paragraph, max(0, part.end() - _SHARE_MAX_DISTANCE), part.start())
                if not _SHARE_UNCERTAIN.search(_share_prefix(paragraph, candidate.start()))]
            if len(totals) != 1:
                continue
            total = totals[0]
            bridge = paragraph[total.end():part.start()]
            if (_SHARE_PERIOD.search(bridge) or _SHARE_ENTITY_SWITCH.search(bridge)
                    or re.search(r'精确|未舍入|未四舍五入|截尾|截断', paragraph[:part.end()])
                    or total['unit'] != part['unit']
                    or len({_share_metric_family(m) for m in
                            (total['metric'], part['metric'], part['share_metric'])}) != 1
                    or any(len(raw) > 32 for raw in (total['value'], part['value'], part['rate']))):
                continue
            with localcontext() as ctx:
                ctx.prec = 40
                nlo, nhi = _interval(part['value'])
                dlo, dhi = _interval(total['value'])
                rlo, rhi = _interval(part['rate'])
                if nlo <= 0 or dlo <= 0:
                    continue
                low, high = nlo / dhi * 100, nhi / dlo * 100
                overlap_lo, overlap_hi = max(low, rlo), min(high, rhi)
                base = offset + block.start()

                def anchor(start: int, end: int) -> dict:
                    return {'start': base + start, 'end': base + end,
                            'text': paragraph[start:end]}

                excluded = []
                if part['growth']:
                    excluded.append({'role': 'period_growth_not_share',
                                     'source': anchor(*part.span('growth'))})
                if forecast:
                    excluded.append({'role': 'forecast_not_actual',
                                     'source': anchor(*part.span('forecast'))})
                result.append({
                    'kind': 'conditional_part_total_share',
                    'scope': 'calculation_only_not_business_verdict',
                    'source': anchor(total.start(), part.end()),
                    'operand_sources': {
                        'numerator': anchor(part.start('value'), part.end('unit')),
                        'denominator': anchor(total.start('value'), total.end('unit')),
                        'reported_percent': anchor(part.start('rate'), part.end()),
                    },
                    'numerator_label': part['label'],
                    'numerator_metric': part['metric'],
                    'denominator_metric': total['metric'],
                    'reported_share_metric': part['share_metric'],
                    'unit': total['unit'],
                    'numerator_raw': part['value'], 'denominator_raw': total['value'],
                    'reported_percent_raw': part['rate'],
                    'formula': 'percent=part/total*100',
                    'point_percent': str(_number(part['value']) / _number(total['value']) * 100),
                    'numerator_interval': [str(nlo), str(nhi)],
                    'denominator_interval': [str(dlo), str(dhi)],
                    'possible_percent': [str(low), str(high)],
                    'reported_rounding_interval': [str(rlo), str(rhi)],
                    'rounding_compatible': overlap_lo <= overlap_hi,
                    'boundary_only_overlap': overlap_lo == overlap_hi,
                    'relation_proven': False,
                    'excluded_roles': excluded,
                    'assumptions': ['round_to_last_displayed_digit', 'same_currency_and_scale',
                                    'unique_explicit_part_total_link_in_bounded_paragraph'],
                    'required_business_checks': ['same_object_and_period',
                                                 'revenue_sales_metric_equivalence',
                                                 'same_reporting_basis'],
                    'interpretation': 'Interval overlap only blocks a point-division contradiction. '
                                      'Neither overlap nor non-overlap proves source correctness; '
                                      'retain other independent issues in the source.',
                })
    return result


def source_arithmetic_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    if not isinstance(text,str) or type(offset) is not int or offset<0 or type(limit) is not int or limit<0:
        raise ValueError('text, nonnegative integer offset and limit required')
    checks = []
    for parser in (price_change_checks,inventory_change_checks,total_component_checks,valuation_rounding_checks,
                   rounded_share_checks):
        checks.extend(parser(text,offset=offset,limit=limit))
    return sorted(checks,key=lambda item:(item['source']['start'],item['kind']))[:limit]
