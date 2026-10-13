"""Explicit model serialization adapters; never extract answers from prose."""

import re

ADAPTERS = {'glm_box_v1', 'robobrain_answer_v1', 'mimo_think_v1', 'hy_answer_v1', 'internvl_think_v1'}


def adapt_answer(answer, adapter=None):
    if adapter is None:
        return answer
    if adapter not in ADAPTERS:
        raise ValueError('unknown answer adapter')
    if adapter == 'internvl_think_v1':
        # The final closing delimiter ends reasoning even if the server's
        # parser has already consumed an earlier delimiter.
        text = answer.rsplit('</think>', 1)[-1]
        return '' if '<think>' in text else text.strip()
    if adapter == 'mimo_think_v1':
        text = answer.strip()
        if not text.startswith('<think>') or text.count('<think>') != 1:
            return answer
        if '</think>' not in text:
            # A truncated reasoning-only response has no final answer.
            return ''
        if text.count('</think>') != 1:
            return answer
        return text.split('</think>', 1)[1].strip()
    if adapter in {'robobrain_answer_v1', 'hy_answer_v1'}:
        text = answer.strip()
        # Accept only one complete, anchored serialization envelope. Never
        # extract a choice from prose, nested/multiple answers or truncation.
        if text.count('<answer>') != 1 or text.count('</answer>') != 1:
            return answer
        if (text.count('<think>'), text.count('</think>')) not in {(0, 0), (1, 1)}:
            return answer
        match = re.fullmatch(r'(?:<think>.*?</think>\s*)?<answer>(.*?)</answer>', text, re.DOTALL)
        if match and '<think>' not in match[1] and '</think>' not in match[1]:
            return match[1].strip()
        return answer
    begin, end = '<|begin_of_box|>', '<|end_of_box|>'
    text = answer.strip()
    if (text.startswith(begin) and text.endswith(end)
            and text.count(begin) == 1 and text.count(end) == 1):
        return text[len(begin):-len(end)].strip()
    return answer
