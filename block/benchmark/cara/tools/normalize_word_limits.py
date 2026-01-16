import argparse
import json
import re
from collections import Counter

# Use tokenizer from vLLM compat if available; falls back to HF AutoTokenizer
from block.benchmark.cara.vllm_compat import get_tokenizer


# Soft guidance template used when no explicit counts are present
SOFT_GUIDANCE_TEMPLATE = "Try to answer within {n} words instead."


def _calc_budget_words(prompt: str, tokenizer, max_total_len: int, output_cap: int, tokens_per_word: float, round_bucket: int) -> int:
    tokens = None
    if tokenizer is not None:
        try:
            tokens = len(tokenizer(prompt).input_ids)
        except Exception:
            tokens = None
    if tokens is None:
        # Fallback estimate: words ~ len(prompt.split()) and tokens_per_word ~ 1.3
        approx_words = max(1, len(prompt.split()))
        tokens = int(approx_words * tokens_per_word)
    available = max(0, min(output_cap, max_total_len - tokens))
    # Convert to words and round DOWN to avoid overspecifying 'at least N words'.
    words = int(available / max(tokens_per_word, 0.1))
    rounded = (words // round_bucket) * round_bucket if words > 0 else 0
    return max(rounded, round_bucket) if rounded > 0 else round_bucket


def normalize_prompt(
    prompt: str,
    *,
    tokenizer,
    max_total_len: int = 1024,
    output_cap: int = 1024,
    tokens_per_word: float = 1.3,
    round_bucket: int = 100,
    soft_clamp: bool = False,
    soft_round_bucket: int | None = 50,
) -> tuple[str, list[tuple[int, int]] | None]:
    """Clamp explicit word-count instructions to fit within token budget.

    - Preserves phrasing (at least / no more than / between / bare 'N words').
    - Clamps numeric values above the dynamic budget to the budget (rounded up).
    - Optionally applies soft clamping (append guidance) when no explicit counts.

    Returns: (new_prompt, list of (old_value, new_value)) when changes occur.
    """
    budget_words = _calc_budget_words(prompt, tokenizer, max_total_len, output_cap, tokens_per_word, round_bucket)

    # Patterns
    num = r"((?:\d{1,3}(?:,\d{3})+)|\d{2,5})"
    word = r"\s*[- ]?words?\b"
    between_pat = re.compile(rf"\bbetween\s+{num}\s+(?:and|to)\s+{num}{word}", re.I)
    at_least_pat = re.compile(rf"\b(?:at\s+least|minimum\s+of|no\s+less\s+than)\s+{num}{word}", re.I)
    at_most_pat = re.compile(rf"\b(?:no\s+more\s+than|max(?:imum)?\s+of|at\s+most|up\s+to)\s+{num}{word}", re.I)
    bare_pat = re.compile(rf"\b{num}{word}", re.I)

    replacements: list[tuple[int, int]] = []
    text = prompt
    # Track if any explicit word-count instruction exists (even if no clamping occurs)
    had_counts = any(p.search(prompt) for p in (between_pat, at_least_pat, at_most_pat, bare_pat))

    def _to_int(s: str) -> int:
        return int(s.replace(',', ''))

    # between X and Y words
    def r_between(m: re.Match) -> str:
        x, y = _to_int(m.group(1)), _to_int(m.group(2))
        new_x = min(x, budget_words)
        new_y = min(y, budget_words)
        if new_x != x:
            replacements.append((x, new_x))
        if new_y != y:
            replacements.append((y, new_y))
        # Ensure order
        a, b = sorted((new_x, new_y))
        # Preserve hyphen vs space as in original
        s = m.group(0)
        hyphen = "-" if "-word" in s.lower() else " "
        return re.sub(rf"between\s+{num}\s+(?:and|to)\s+{num}{word}", f"between {a} and {b}{hyphen}words", s, flags=re.I)

    text = between_pat.sub(r_between, text)

    # at least / minimum of / no less than
    def r_atleast(m: re.Match) -> str:
        n = _to_int(m.group(1))
        new_n = min(n, budget_words)
        if new_n != n:
            replacements.append((n, new_n))
        s = m.group(0)
        hyphen = "-" if "-word" in s.lower() else " "
        return re.sub(num + word, f"{new_n}{hyphen}words", s, count=1)

    text = at_least_pat.sub(r_atleast, text)

    # at most / maximum of / no more than / up to
    def r_atmost(m: re.Match) -> str:
        n = _to_int(m.group(1))
        new_n = min(n, budget_words)
        if new_n != n:
            replacements.append((n, new_n))
        s = m.group(0)
        hyphen = "-" if "-word" in s.lower() else " "
        return re.sub(num + word, f"{new_n}{hyphen}words", s, count=1)

    text = at_most_pat.sub(r_atmost, text)

    # bare 'N words'
    def r_bare(m: re.Match) -> str:
        n = _to_int(m.group(1))
        new_n = min(n, budget_words)
        if new_n != n:
            replacements.append((n, new_n))
        s = m.group(0)
        hyphen = "-" if "-word" in s.lower() else " "
        return f"{new_n}{hyphen}words"

    # Apply bare replacements last to avoid stomping over above phrases
    text = bare_pat.sub(r_bare, text)

    # Optional soft clamp: append a guidance when no explicit counts matched.
    if soft_clamp and not had_counts:
        # Compute available tokens for the response given current prompt length
        def _count_tokens(s: str) -> int:
            if tokenizer is not None:
                try:
                    return len(tokenizer(s).input_ids)
                except Exception:
                    pass
            # Fallback estimate
            approx_words = max(1, len(s.split()))
            return int(approx_words * tokens_per_word)

        prompt_tokens = _count_tokens(text)
        available_tokens = max(0, min(output_cap, max_total_len - prompt_tokens))

        # Effective tokens-per-word (fallback if invalid)
        eff_tpw = tokens_per_word if tokens_per_word > 0 else 1.3

        # Iteratively infer overhead from the actual appended guidance; 3 passes max
        def _round_words(n: int) -> int:
            if soft_round_bucket and n > 0:
                return max(1, (n // soft_round_bucket) * soft_round_bucket)
            return max(1, n)

        n_words = max(1, int(available_tokens / eff_tpw))
        n_words = _round_words(n_words)
        for _ in range(3):
            guidance = SOFT_GUIDANCE_TEMPLATE.format(n=n_words)
            append_probe = ("\n\n" + guidance)
            append_tokens = _count_tokens(append_probe)
            refined_available = max(0, available_tokens - append_tokens)
            refined_words = _round_words(int(refined_available / eff_tpw))
            if refined_words == n_words:
                break
            n_words = refined_words

        # Append with separation to avoid colliding with last sentence
        text = text.rstrip() + "\n\n" + SOFT_GUIDANCE_TEMPLATE.format(n=n_words)

    return text, (replacements or None)


def main():
    ap = argparse.ArgumentParser(description="Clamp explicit word-count instructions to fit per-prompt token budgets.")
    ap.add_argument("--infile", required=True, help="Path to input JSONL (with {id, prompt})")
    ap.add_argument("--outfile", required=True, help="Path to output JSONL")
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-3B", help="HF tokenizer name/path (default: Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--trust-remote-code", action="store_true", help="HF trust_remote_code for tokenizer")
    ap.add_argument("--max-total-len", type=int, default=1024, help="Max total length (prompt + output) in tokens")
    ap.add_argument("--output-cap", type=int, default=1024, help="Default output token cap used to compute budgets")
    ap.add_argument("--tokens-per-word", type=float, default=1.3, help="Approx token/word used when tokenizer unavailable (backup only)")
    ap.add_argument("--infer-tokens-per-word", action="store_true", default=True, help="Infer tokens-per-word per prompt from tokenizer (default: enabled)")
    ap.add_argument("--no-infer-tokens-per-word", action="store_false", dest="infer_tokens_per_word", help="Disable per-prompt inference of tokens-per-word")
    ap.add_argument("--calibrate-tpw-samples", type=int, default=0, help="Calibrate a global tokens-per-word using first N prompts (requires --tokenizer). Overrides --tokens-per-word and --infer-tokens-per-word when >0")
    ap.add_argument("--round", type=int, default=100, help="Round budget words up to nearest N (default: 100)")
    ap.add_argument("--dry-run", action="store_true", help="Print stats and examples, do not write file")
    # Soft clamp (disabled by default): append brief guidance when no explicit counts are present
    ap.add_argument("--soft-clamp", action="store_true", help="Append a short length guidance when no explicit word counts exist")
    ap.add_argument("--soft-round", type=int, default=50, help="Down-round soft budgets to nearest N words (default: 50)")
    args = ap.parse_args()

    stats = Counter()
    changed = 0
    examples = []

    with open(args.infile, "r") as fin:
        lines = fin.readlines()

    # Initialize tokenizer if provided
    tok = None
    if args.tokenizer:
        try:
            tok = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
        except Exception as e:
            print(f"Warning: failed to load tokenizer {args.tokenizer}: {e}. Falling back to heuristic.")
            tok = None

    # Calibrate tokens-per-word globally if requested
    tpw_global = None
    if args.calibrate_tpw_samples > 0 and tok is not None:
        import re
        word_re = re.compile(r"\b\w+\b")
        total_tokens = 0
        total_words = 0
        for idx, line in enumerate(lines):
            if idx >= args.calibrate_tpw_samples:
                break
            try:
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
            except Exception:
                continue
            try:
                t = len(tok(prompt).input_ids)
            except Exception:
                continue
            w = len(word_re.findall(prompt)) or len(prompt.split()) or 1
            total_tokens += t
            total_words += w
        if total_words > 0:
            tpw_global = total_tokens / total_words
            # Clamp to reasonable range
            tpw_global = max(1.1, min(2.0, tpw_global))
            print(f"Calibrated tokens-per-word (global): {tpw_global:.3f} using {min(len(lines), args.calibrate_tpw_samples)} samples")

    out_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        prompt = obj.get("prompt", "")
        # Determine tokens-per-word to use for this prompt
        tpw = args.tokens_per_word
        if tpw_global is not None:
            tpw = tpw_global
        elif args.infer_tokens_per_word and tok is not None:
            # Infer per-prompt TPW from tokenizer
            import re as _re
            _word_re = _re.compile(r"\b\w+\b")
            try:
                n_tokens = len(tok(prompt).input_ids)
                n_words = len(_word_re.findall(prompt)) or len(prompt.split()) or 1
                tpw = max(1.0, min(2.0, n_tokens / n_words))
            except Exception:
                tpw = args.tokens_per_word

        new_prompt, replaced = normalize_prompt(
            prompt,
            tokenizer=tok,
            max_total_len=args.max_total_len,
            output_cap=args.output_cap,
            tokens_per_word=tpw,
            round_bucket=args.round,
            soft_clamp=args.soft_clamp,
            soft_round_bucket=(args.soft_round if args.soft_round > 0 else None),
        )
        # Tally replacements
        if replaced:
            for old, new in replaced:
                stats[(old, new)] += 1

        # Mark as changed if prompt text modified (covers soft clamp too)
        if new_prompt != prompt:
            changed += 1
            if len(examples) < 10:
                examples.append((prompt[:160], new_prompt[:160], replaced or []))
        obj["prompt"] = new_prompt
        out_lines.append(json.dumps(obj, ensure_ascii=False))

    print("Total prompts:", len(out_lines))
    print("Changed prompts:", changed)
    if stats:
        print("Replaced word-count values (old->new : count):")
        for (old, new), cnt in sorted(stats.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f"  {old} -> {new} : {cnt}")
    if examples:
        print("\nExamples (before -> after):")
        for b, a, rep in examples:
            print("-", b)
            print("+", a)
            print("  replaced:", rep)

    if not args.dry_run:
        with open(args.outfile, "w", encoding="utf-8") as fout:
            fout.write("\n".join(out_lines) + "\n")
        print("\nWrote:", args.outfile)


if __name__ == "__main__":
    main()
