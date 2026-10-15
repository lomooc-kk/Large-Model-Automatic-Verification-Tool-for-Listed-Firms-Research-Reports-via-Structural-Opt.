"""Recheck the packaged evidence without network calls or model credentials.

Usage: python tools/reproduce_checks.py [--repo PATH_TO_CURRENT_REPO]
Without --repo, verifies the portable source snapshots, offsets and calculations.
With --repo, also reruns the existing offline detector and parser probes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    base=Path(__file__).resolve().parents[1]
    def read(rel):return json.loads((base/rel).read_text(encoding='utf-8'))
    cases=read('evidence/cases.json')
    inputs={x['doc_id']:x for x in read('evidence/selected_source_records.json')}
    meta=read('evidence/provenance.json')
    checks=[]
    assert len(cases)==12 and len(inputs)==10
    assert sorted(Counter(x['gold']['type'] for x in cases).values())==[3,3,3,3]
    for c in cases:
        row=inputs[c['document_id']]
        assert hashlib.sha256(row['content'].encode()).hexdigest()==c['source_content_sha256']
        assert c['gold']['scorable'] is True
        for s in c['gold']['spans']:
            assert row['content'][s['start']:s['end']]==s['text']
        checks.append({'case_id':c['case_id'],'content_sha_and_offsets':'passed'})
    D=Decimal
    assert D('35')*D('100000000')/D('10000')/D('10000')==D(35)
    assert D('58.76')-D('54.9')==D('3.86')
    assert abs(D('1988.9')/D('2503.8')*100-D('79.4352584072'))<D('.000000001')
    assert abs(D('17.94')/(D('1088.33')-D('17.94'))*100-D('1.6760246265'))<D('.000000001')
    result={'portable_source_and_offset_checks':checks,'calculations':'passed',
            'expected_code_commit':meta['head'],'model_rerun':False}
    if args.repo:
        sys.path.insert(0,str(args.repo/'factcheck/src'))
        from yjcheck.text_review import detect_text
        outputs={}
        for did,row in inputs.items():
            outputs[did]=detect_text(row['content'],document_id=did,detector='hybrid',chat=None,examples=())
            assert outputs[did]['coverage']['model_ran'] is False
        matched=[]
        for c in cases:
            for e in outputs[c['document_id']]['errors']:
                if e['error_type']==c['gold']['type'] and all(any(p['start']<=s['start'] and p['end']>=s['end'] for p in e['spans']) for s in c['gold']['spans']):
                    matched.append(c['case_id'])
        probes=[]
        for pr in read('evidence/parser_probes.json'):
            outcomes={}
            for label in ['before','after']:
                output=detect_text(pr[label]['text'],document_id='offline-diagnostic-control',detector='hybrid',chat=None,examples=())
                actual=[(e['error_type'],e['detector_id']) for e in output['errors']]
                expected=[(e['error_type'],e['detector_id']) for e in pr[label]['errors']]
                outcomes[label]={'actual':actual,'expected':expected,'same_as_pack':actual==expected}
            probes.append({'probe_id':pr['probe_id'],'outcomes':outcomes})
        result.update(offline_documents=len(outputs),offline_target_matches=matched,parser_probes=probes)
    if args.out:
        args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
