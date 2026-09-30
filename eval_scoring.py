import re

def score_answer(answer,expected,has_context):
    text=answer.lower().replace(',','')
    abstained=any(t in text for t in ['unavailable','not provided','not specified','does not','no information','not mentioned','no evidence','not available',"don't have",'cannot determine'])
    facts=(expected in text) if expected else abstained
    refs=re.findall(r'\[S\d+\]',answer)
    invalid=any(ref not in ({'[S1]'} if has_context else set()) for ref in refs)
    repetitive=any(refs.count(ref)>3 for ref in set(refs))
    return {'fact_check':bool(facts),'abstained':abstained,'unsupported_citation':invalid,'repetitive_citations':repetitive,'supported_response_check':bool((facts if has_context else abstained) and not invalid and not repetitive)}
