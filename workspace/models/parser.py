import re

ANS_RE = re.compile(r"^ANSWER:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
CONF_RE = re.compile(r"^CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)

def parse_answer_and_conf(text: str) -> tuple[str, float]:
    ans_m = ANS_RE.search(text or "")
    conf_m = CONF_RE.search(text or "")
    ans = ans_m.group(1).strip() if ans_m else "CANNOT_ANSWER"
    conf = 0.0
    if conf_m:
        try:
            conf = float(conf_m.group(1))
        except ValueError:
            conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return ans, conf
