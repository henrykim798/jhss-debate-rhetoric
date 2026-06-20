#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM rhetoric classifier (replicable, no human coding required).

Classifies each candidate debate sentence into three INDEPENDENT binary labels
(aggressive, inflammatory, discriminatory) using a strict rubric (RUBRIC.md),
then (a) validates against the 60-sentence expert gold set and (b) aggregates to
candidate-debate counts for the regressions.

Default backend: OpenAI (model gpt-4o-mini). To use Anthropic, set BACKEND="anthropic".
Nothing here requires manual labeling; the rubric is fixed and the run is deterministic
at temperature=0, so results are replicable.

USAGE
-----
  pip install openai            # or: pip install anthropic
  export OPENAI_API_KEY=sk-...  # or ANTHROPIC_API_KEY=...
  python llm_classify.py --corpus ../../R1/Data_Files/corpus_sentences.csv \
                         --gold   ../../R1/Data_Files/gold_sample.json \
                         --out    ../llm_labels.csv

Cost note: ~24,546 sentences at batch size 20 = ~1,230 calls. With gpt-4o-mini this
is a few US dollars. Results are cached to llm_cache.jsonl so reruns are free.
"""
import os, json, time, argparse, hashlib, re, sys

BACKEND = os.environ.get("LLM_BACKEND", "openai")   # "openai" or "anthropic"
MODEL   = os.environ.get("LLM_MODEL", "gpt-4o-mini" if BACKEND=="openai" else "claude-3-5-sonnet-latest")
BATCH   = 20

RUBRIC = """You are a careful political-communication coder. For each numbered sentence from a
US presidential debate, assign three INDEPENDENT binary labels (0 or 1). A sentence may
receive more than one 1, or all 0s. Judge the MEANING, not keywords.

aggressive=1: confrontational/assertive/dominance speech toward an opponent or institution
  (accusations of failure, lying, incompetence; blunt "you" challenges) WITHOUT emotional
  escalation or group derogation.
inflammatory=1: emotionally provocative, morally loaded, hyperbolic, fear/anger/outrage
  arousing (catastrophizing, moral condemnation, threat framing, charged name-calling).
discriminatory=1: the SPEAKER demeans/dehumanizes/derogates a PROTECTED GROUP (race,
  ethnicity, religion, national origin, gender, sexual orientation) AS A GROUP.
  Set discriminatory=0 when the sentence: condemns/denounces/describes discrimination by
  others; quotes or attributes discriminatory speech to someone else; uses "those people"
  neutrally/positively; discusses immigration/abortion policy WITHOUT group derogation; or
  attacks an individual or a foreign leader/government (not a protected group).
  When unsure, discriminatory=0 unless the group derogation is explicit.

Return ONLY a JSON array, one object per input sentence, in order:
[{"i":0,"aggressive":0,"inflammatory":1,"discriminatory":0}, ...]"""

FEWSHOT = [
    ("He failed the American people and he knows it.", dict(aggressive=1, inflammatory=0, discriminatory=0)),
    ("This is an absolute disaster that will destroy our country.", dict(aggressive=0, inflammatory=1, discriminatory=0)),
    ("Those immigrants are animals who don't belong in this country.", dict(aggressive=0, inflammatory=1, discriminatory=1)),
    ("He launched his campaign by calling Mexicans rapists.", dict(aggressive=1, inflammatory=0, discriminatory=0)),
    ("He praises thugs like the leader of North Korea.", dict(aggressive=1, inflammatory=1, discriminatory=0)),
    ("Those people on the front lines saved our lives.", dict(aggressive=0, inflammatory=0, discriminatory=0)),
    ("We should expand access to community college.", dict(aggressive=0, inflammatory=0, discriminatory=0)),
]

def _client():
    if BACKEND == "openai":
        from openai import OpenAI
        return OpenAI()
    else:
        import anthropic
        return anthropic.Anthropic()

def _call(client, sentences):
    fs = "\n".join(f'Example: "{t}" -> {json.dumps(d)}' for t, d in FEWSHOT)
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    user = f"{fs}\n\nClassify these sentences:\n{numbered}"
    if BACKEND == "openai":
        r = client.chat.completions.create(
            model=MODEL, temperature=0,
            messages=[{"role": "system", "content": RUBRIC},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"})
        txt = r.choices[0].message.content
    else:
        r = client.messages.create(
            model=MODEL, max_tokens=2000, temperature=0,
            system=RUBRIC, messages=[{"role": "user", "content": user}])
        txt = r.content[0].text
    m = re.search(r"\[.*\]", txt, re.S)
    arr = json.loads(m.group(0) if m else txt)
    return arr

def classify_all(sentences, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            o = json.loads(line); cache[o["h"]] = o["lab"]
    client = _client(); out = [None]*len(sentences); todo = []
    for i, s in enumerate(sentences):
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        if h in cache: out[i] = cache[h]
        else: todo.append(i)
    cf = open(cache_path, "a", encoding="utf-8")
    for b in range(0, len(todo), BATCH):
        idx = todo[b:b+BATCH]; batch = [sentences[j] for j in idx]
        for attempt in range(4):
            try:
                arr = _call(client, batch); break
            except Exception as e:
                if attempt == 3: raise
                time.sleep(2*(attempt+1))
        for k, j in enumerate(idx):
            lab = next((a for a in arr if a.get("i") == k), {})
            rec = {c: int(bool(lab.get(c, 0))) for c in ("aggressive","inflammatory","discriminatory")}
            out[j] = rec
            h = hashlib.md5(sentences[j].encode("utf-8")).hexdigest()
            cf.write(json.dumps({"h": h, "lab": rec})+"\n"); cf.flush()
        print(f"  classified {min(b+BATCH,len(todo))}/{len(todo)} new", end="\r")
    cf.close(); print()
    return out

def validate_gold(gold_path, classify_fn):
    sents = [r["sentence"] for r in json.load(open(gold_path, encoding="utf-8"))]
    # expert single-labels (A/I/D/N), order matches gold_sample.json
    gold = list("NNNIAINNNI"+"NNNINNINNN"+"NNNNNNANNN"+"ANANNNNINI"+"NNNIAINNNN"+"NINNANNNNN")
    preds = classify_fn(sents)
    print("\n=== GOLD VALIDATION (per-category, binary) ===")
    cmap = {"aggressive":"A","inflammatory":"I","discriminatory":"D"}
    for cat, letter in cmap.items():
        tp = sum(p[cat]==1 and g==letter for p,g in zip(preds,gold))
        fp = sum(p[cat]==1 and g!=letter for p,g in zip(preds,gold))
        fn = sum(p[cat]==0 and g==letter for p,g in zip(preds,gold))
        ng = sum(g==letter for g in gold)
        prec = tp/(tp+fp) if tp+fp else float('nan')
        rec  = tp/(tp+fn) if tp+fn else float('nan')
        f1 = 2*prec*rec/(prec+rec) if (prec and rec and prec==prec and rec==rec and prec+rec>0) else float('nan')
        print(f"  {cat:15s} n_gold={ng:2d}  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}")

def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="llm_labels.csv")
    ap.add_argument("--cache", default="llm_cache.jsonl")
    ap.add_argument("--counts", default="llm_counts.csv")
    a = ap.parse_args()

    fn = lambda S: classify_all(S, a.cache)
    # 1) validate on gold first (cheap, sanity check before full run)
    validate_gold(a.gold, fn)
    # 2) classify full corpus
    df = pd.read_csv(a.corpus)
    labels = classify_all(df["sentence"].astype(str).tolist(), a.cache)
    for c in ("aggressive","inflammatory","discriminatory"):
        df[c] = [l[c] for l in labels]
    df.to_csv(a.out, index=False)
    # 3) aggregate to candidate-debate counts
    g = df.groupby(["date","party"])[["aggressive","inflammatory","discriminatory"]].sum().reset_index()
    g.to_csv(a.counts, index=False)
    print(f"\nWrote per-sentence labels -> {a.out}")
    print(f"Wrote candidate-debate counts -> {a.counts}")
    print("Next: merge llm_counts.csv into the poll panel and re-run reanalysis_master.py.")

if __name__ == "__main__":
    main()
