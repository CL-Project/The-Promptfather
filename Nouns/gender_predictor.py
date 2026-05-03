"""
gender_predictor.py
====================
Train a character n-gram logistic regression classifier to predict
gender for consonant-final Hindi nouns, then apply it to the 4,491
excluded words that couldn't be assigned a paradigm class.

HOW IT WORKS
------------
Consonant-final nouns are the hard case — their gender can't be read
off from their phonological shape the way vowel-final nouns can.
But gender isn't *random* either: suffix patterns correlate with gender
across the vocabulary.  For example:
  - words ending in -न are often masculine (कारण, चलन, पालन)
  - words ending in -त are often feminine (बात, रात, बात)
  - words ending in -र are often masculine (घर, नगर, सागर)

A logistic regression over character n-grams of the last 1–4 characters
learns these correlations from the 12,921 consonant-final nouns that
already have known gender (M2 = masculine, F3 = feminine).

⚠  KNOWN LIMITATIONS (not fixable in code alone)
-------------------------------------------------
1. The model is trained on native Hindi vocabulary and is weakest on
   loanwords (Arabic, Persian, English) — which are exactly the words
   being predicted.  Reported cross-validation accuracy is measured on
   held-out M2/F3 words (already well-behaved), not on this harder
   excluded population.  True accuracy on the target set is unknown and
   almost certainly lower.

2. There is no gold-standard list to check predictions against.
   Treat all output — including predicted_high entries — as hypotheses
   for human review, not as authoritative classifications.

INPUT FILES  (all relative paths — run from the Nouns/ directory)
-----------
    data/noun_lexicon_final.tsv    — 20,953 nouns with certain paradigm class
    data/noun_lexicon_merged.tsv   — raw merged lexicon including ex_noun words

OUTPUT FILES
------------
    data/noun_lexicon_predicted.tsv  — high+medium confidence predictions only
    data/noun_lexicon_review.tsv     — low-confidence predictions (p < 0.65);
                                       requires manual review before use
    data/noun_lexicon_abstain.tsv    — entries where the model abstains (p < 0.55);
                                       predictions no better than a coin flip
    data/noun_lexicon_complete.tsv   — expanded lexicon + high/medium predictions

USAGE
-----
    cd Nouns/
    python gender_predictor.py

DEPENDENCIES
------------
    scikit-learn, numpy  (pip install scikit-learn numpy)
"""

import csv
import unicodedata
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# ── Unicode / grapheme helpers ───────────────────────────────────────────────

def normalize(text: str) -> str:
    """NFC + nukta stripping, matching the noun analyzer's normalization."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u0921\u093c", "\u0921")  # ड़ → ड
    text = text.replace("\u0922\u093c", "\u0922")  # ढ़ → ढ
    return text


def devanagari_grapheme_clusters(text: str) -> list[str]:
    """
    Split a Devanagari string into grapheme clusters.

    Each cluster is a base character (consonant or independent vowel)
    followed by all its combining diacritics (matras, virama, anusvara,
    chandrabindu, visarga, etc.).

    This is needed to correctly reverse a string without splitting a
    consonant from its matra:
        'रात'[::-1]  → 'त', 'ा', 'र'  (wrong — matra detaches)
        reversed clusters → ['त', 'ार', 'र']  — still wrong in isolation,
        but combining each cluster keeps त+ा together as a unit.

    Note: This is a lightweight approximation. A production system should
    use the `grapheme` or `regex` library for full Unicode segmentation.
    """
    clusters: list[str] = []
    for ch in text:
        if clusters and unicodedata.combining(ch):
            # Attach combining character to the preceding cluster.
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def reverse_by_graphemes(text: str) -> str:
    """Reverse a Devanagari string by grapheme cluster, not by code point."""
    return "".join(reversed(devanagari_grapheme_clusters(text)))


def suffix_from_reversed_ngram(ngram: str) -> str:
    """
    Recover the original suffix from a reversed-string n-gram.

    The vectoriser operates on grapheme-reversed words (see word_to_feature_string).
    A feature like 'ताब' (from reversed 'बात') represents the suffix '-बात'.
    To display it, we reverse the n-gram's grapheme clusters.

    FIX (issue 6): the old code used feat[i][::-1] which reverses code points,
    splitting consonants from their matras and producing unreadable output for
    any n-gram that spans a matra.  We now reverse by grapheme cluster instead.
    """
    return reverse_by_graphemes(ngram)


# Characters that mark a vowel-final word.
_VOWEL_FINAL_CHARS = frozenset(
    "ािीुूेैोौँंः"    # combining matras + anusvara + visarga
    "अआइईउऊएऐओऔ"    # independent vowels
    "ऄअॠॡ"          # rare independent vowels
)


def is_consonant_final(word: str) -> bool:
    """Return True if *word* ends in a consonant (no vowel matra or independent vowel)."""
    if not word:
        return False
    return word[-1] not in _VOWEL_FINAL_CHARS


# ── Paradigm class → gender label ───────────────────────────────────────────

MASCULINE_CLASSES = {"M1", "M2", "M3", "M4"}
FEMININE_CLASSES  = {"F1", "F2", "F3", "F4", "F5"}
CONSONANT_MASCULINE = {"M2"}
CONSONANT_FEMININE  = {"F3"}


def paradigm_to_gender(paradigm: str) -> str | None:
    """Map a paradigm class to 'M' or 'F'. Returns None for non-consonant-final classes."""
    if paradigm in CONSONANT_MASCULINE:
        return "M"
    if paradigm in CONSONANT_FEMININE:
        return "F"
    return None


# ── Feature extraction ───────────────────────────────────────────────────────

def word_to_feature_string(word: str) -> str:
    """
    Return the input string for the CountVectorizer.

    FIX (issue 6 prerequisite): we now reverse by grapheme cluster so that
    the vectoriser's character n-grams correspond to true Devanagari suffixes,
    even when the word contains consonant+matra sequences.

    Example:
        'रात'  → grapheme clusters: ['र', 'ा', 'त'] → reversed: ['त', 'ा', 'र']
        joined: 'तार'
        The bigram 'ता' captures the suffix ending 'ात' correctly.

    OLD (buggy for words with matras): word[::-1]
    """
    return reverse_by_graphemes(word)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_training_data(tsv_path: str) -> tuple[list[str], list[str]]:
    """
    Read noun_lexicon_final.tsv and extract consonant-final words only.

    Returns (words, labels) where label ∈ {'M', 'F'}.
    """
    words, labels = [], []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            word     = normalize(row["word"])
            paradigm = row["paradigm_class"].strip()
            gender   = paradigm_to_gender(paradigm)
            if gender is None:
                continue
            if not is_consonant_final(word):
                continue
            words.append(word)
            labels.append(gender)
    return words, labels


def load_excluded_words(merged_tsv: str, expanded_tsv: str) -> list[str]:
    """
    Find the consonant-final ex_noun words that were excluded from the lexicon.

    FIX (issue 5): After normalization strips nuktas, two different source
    entries (e.g. खिड़की and खिडकी) can map to the same normalized form.
    The old code would emit duplicate entries that could then receive different
    predictions and create conflicts in the output.

    We now deduplicate by normalized form and log any conflicts so the caller
    is aware.

    Strategy:
      1. Load all words already in the expanded lexicon (certain + heuristic).
      2. Load all ex_noun words from the merged lexicon.
      3. Deduplicate by normalized form, logging collisions.
      4. Return consonant-final ex_noun words not already classified.
    """
    already_classified: set[str] = set()
    with open(expanded_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            already_classified.add(normalize(row["word"]))

    # Collect all candidate ex_noun words, grouped by normalized form.
    # Value = list of original (pre-normalize) forms seen for that key.
    seen: dict[str, list[str]] = defaultdict(list)
    with open(merged_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_word = row["word"]
            word     = normalize(raw_word)
            paradigm = row["paradigm_class"].strip()
            if paradigm != "ex_noun":
                continue
            if not word:
                continue
            if word in already_classified:
                continue
            if not is_consonant_final(word):
                continue
            seen[word].append(raw_word)

    # Report collisions and deduplicate.
    excluded: list[str] = []
    collision_count = 0
    for norm_word, originals in seen.items():
        if len(originals) > 1:
            collision_count += 1
            unique_originals = sorted(set(originals))
            warnings.warn(
                f"Nukta collision: {unique_originals} all normalize to "
                f"'{norm_word}'. Keeping one entry; verify gender manually.",
                stacklevel=2,
            )
        excluded.append(norm_word)  # always exactly one entry per normalized form

    if collision_count:
        print(f"\n  ⚠  {collision_count} nukta collision(s) detected and deduplicated.")
        print(f"     Review warnings above and verify those entries manually.")

    return excluded


# ── Model building ────────────────────────────────────────────────────────────

def build_pipeline(max_ngram: int = 4, C: float = 1.0) -> Pipeline:
    """
    Scikit-learn pipeline: reversed-string char n-gram vectoriser → logistic regression.
    """
    vectoriser = CountVectorizer(
        analyzer="char",
        ngram_range=(1, max_ngram),
        min_df=2,
        binary=True,
    )
    classifier = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    return Pipeline([("vec", vectoriser), ("clf", classifier)])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(pipeline: Pipeline, words: list[str], labels: list[str], n_folds: int = 5) -> None:
    """
    Stratified k-fold cross-validation + final fit on full training set.

    FIX (issue 2): prints an explicit warning that cross-validation accuracy
    is measured on the M2/F3 training population, not on the excluded words
    being predicted.  The true accuracy on the excluded set is unknown.
    """
    X = [word_to_feature_string(w) for w in words]
    y = np.array(labels)

    print(f"\n{'─' * 60}")
    print(f"  Training data: {len(words):,} consonant-final nouns")
    dist = Counter(labels)
    print(f"  Class distribution: M={dist['M']:,}  F={dist['F']:,}")
    print(f"{'─' * 60}")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"\n{n_folds}-fold cross-validation accuracy:")
    print(f"  {scores.mean():.4f} ± {scores.std():.4f}  "
          f"(min={scores.min():.4f}, max={scores.max():.4f})")
    for i, s in enumerate(scores, 1):
        bar = "█" * int(s * 40)
        print(f"  Fold {i}: {s:.4f}  {bar}")

    # Distribution-shift warning — issue 2.
    print(f"""
  ⚠  ACCURACY WARNING
     The figures above measure performance on held-out M2/F3 words —
     words already successfully classified, which follow regular Hindi
     phonological patterns.  The excluded (ex_noun) words being predicted
     are systematically harder: they were excluded *because* they don't
     fit those patterns, and most are loanwords from Arabic, Persian, or
     English.  True accuracy on the target population is unknown and is
     almost certainly lower than the cross-validation figure above.
     Treat all predictions as hypotheses, not authoritative labels.
""")

    cv_iter = list(cv.split(X, y))
    train_idx, test_idx = cv_iter[-1]
    X_arr = np.array(X)

    pipeline.fit(X_arr[train_idx], y[train_idx])
    y_pred = pipeline.predict(X_arr[test_idx])
    y_true = y[test_idx]

    print(f"Classification report (fold {n_folds} held-out set, "
          f"n={len(test_idx):,}):")
    print(classification_report(y_true, y_pred, target_names=["F", "M"]))

    cm = confusion_matrix(y_true, y_pred, labels=["F", "M"])
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"         pred_F  pred_M")
    print(f"  true_F  {cm[0,0]:>5}   {cm[0,1]:>5}")
    print(f"  true_M  {cm[1,0]:>5}   {cm[1,1]:>5}")

    print("\nTop 15 n-grams most predictive of FEMININE:")
    _print_top_features(pipeline, label="F", n=15)
    print("\nTop 15 n-grams most predictive of MASCULINE:")
    _print_top_features(pipeline, label="M", n=15)


def _print_top_features(pipeline: Pipeline, label: str, n: int = 15) -> None:
    """
    Print the n-grams with the highest logistic regression weight for *label*.

    FIX (issue 6): the old code reversed the n-gram with feat[i][::-1], which
    reverses Unicode code points.  For n-grams that include a matra (a combining
    character), this produces a string where the matra precedes its base consonant
    — an invalid Devanagari sequence that doesn't correspond to any real suffix.

    We now use suffix_from_reversed_ngram(), which reverses by grapheme cluster
    so the matra stays attached to its consonant.
    """
    clf  = pipeline.named_steps["clf"]
    vec  = pipeline.named_steps["vec"]
    feat = vec.get_feature_names_out()
    coefs     = clf.coef_[0]
    sign      = 1 if label == "M" else -1
    top_idx   = np.argsort(sign * coefs)[::-1][:n]
    for i in top_idx:
        suffix = suffix_from_reversed_ngram(feat[i])
        print(f"    -{suffix:<6}  weight={coefs[i]:+.3f}")


# ── Confidence tiers ─────────────────────────────────────────────────────────

# FIX (issues 3 & 4): introduce a hard abstain threshold and separate the
# low-confidence tier from the lexicon output entirely.
#
# Old behaviour: every word got a prediction; predicted_low entries were
# written to the main lexicon with no special handling.
#
# New behaviour:
#   p ≥ 0.80  → predicted_high    → written to lexicon
#   p ≥ 0.65  → predicted_medium  → written to lexicon
#   p ≥ 0.55  → predicted_low     → written to REVIEW file only, NOT to lexicon
#   p < 0.55  → abstain           → written to ABSTAIN file; no gender assigned
#
# The 0.55 abstain threshold is a heuristic: a probability of 0.53 is barely
# better than chance.  The model should not commit to a label in that range.

THRESHOLD_HIGH   = 0.80
THRESHOLD_MEDIUM = 0.65
THRESHOLD_LOW    = 0.55  # below this → abstain


def classify_confidence(p_win: float) -> str:
    """Map a winning probability to a confidence label."""
    if p_win >= THRESHOLD_HIGH:
        return "predicted_high"
    if p_win >= THRESHOLD_MEDIUM:
        return "predicted_medium"
    if p_win >= THRESHOLD_LOW:
        return "predicted_low"
    return "abstain"


# ── Prediction ────────────────────────────────────────────────────────────────

def assign_paradigm(gender: str, word: str) -> str:
    """Consonant-final masculine → M2, consonant-final feminine → F3."""
    return "M2" if gender == "M" else "F3"


def predict_and_save(
    pipeline:      Pipeline,
    excluded:      list[str],
    output_path:   str,
    review_path:   str,
    abstain_path:  str,
) -> dict[str, tuple[str, str, float]]:
    """
    Predict gender for all excluded consonant-final words.

    FIX (issues 3 & 4):
    - Words with p < 0.55 are written to *abstain_path* with no gender label.
    - Words with 0.55 ≤ p < 0.65 are written to *review_path* only; they are
      NOT included in the returned dict and will NOT be merged into the lexicon.
    - Only high and medium confidence predictions are returned and eventually
      merged into noun_lexicon_complete.tsv.

    Returns {word: (paradigm_class, confidence_label, probability)} for
    high+medium predictions only.
    """
    X = [word_to_feature_string(w) for w in excluded]
    probs   = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["clf"].classes_)   # ['F', 'M']
    m_idx   = classes.index("M")

    lexicon_results: dict[str, tuple[str, str, float]] = {}

    with (
        open(output_path,  "w", encoding="utf-8") as f_main,
        open(review_path,  "w", encoding="utf-8") as f_review,
        open(abstain_path, "w", encoding="utf-8") as f_abstain,
    ):
        f_main.write("word\tparadigm_class\tconfidence\tprobability\n")
        f_review.write(
            "word\tparadigm_class\tconfidence\tprobability\n"
            "# These entries have p < 0.65 — barely above chance.\n"
            "# Do NOT add to the lexicon without manual verification.\n"
        )
        f_abstain.write(
            "word\tprobability\n"
            "# The model abstained on these words (p < 0.55).\n"
            "# No gender is predicted; manual assignment required.\n"
        )

        for word, prob_vec in zip(excluded, probs):
            p_m   = prob_vec[m_idx]
            p_win = max(prob_vec)
            conf  = classify_confidence(p_win)

            if conf == "abstain":
                f_abstain.write(f"{word}\t{p_win:.4f}\n")
                continue

            gender   = "M" if p_m >= 0.5 else "F"
            paradigm = assign_paradigm(gender, word)

            if conf == "predicted_low":
                # Goes to review file only — not to the main lexicon.
                f_review.write(f"{word}\t{paradigm}\t{conf}\t{p_win:.4f}\n")
                continue

            # predicted_high or predicted_medium → main output + lexicon.
            f_main.write(f"{word}\t{paradigm}\t{conf}\t{p_win:.4f}\n")
            lexicon_results[word] = (paradigm, conf, p_win)

    return lexicon_results


def merge_into_complete_lexicon(
    expanded_tsv:  str,
    predicted:     dict[str, tuple[str, str, float]],
    output_path:   str,
) -> None:
    """
    Merge high+medium confidence predicted words into the expanded lexicon.

    FIX (issue 4): only entries in *predicted* are merged; predicted_low and
    abstain entries are intentionally excluded.  The caller must inspect
    noun_lexicon_review.tsv and noun_lexicon_abstain.tsv separately.
    """
    existing = {}
    with open(expanded_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            w = normalize(row["word"])
            existing[w] = (row["paradigm_class"], row["confidence"], "")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("word\tparadigm_class\tconfidence\tprobability\n")

        for word, (cls, conf, prob) in sorted(existing.items()):
            f.write(f"{word}\t{cls}\t{conf}\t{prob}\n")

        added = 0
        for word, (cls, conf, prob) in sorted(predicted.items()):
            if word not in existing:
                f.write(f"{word}\t{cls}\t{conf}\t{prob:.4f}\n")
                added += 1

    print(f"\nComplete lexicon written to: {output_path}")
    print(f"  Existing entries:          {len(existing):,}")
    print(f"  Predicted additions:       {added:,}")
    print(f"  Total:                     {len(existing) + added:,}")
    print(f"  (Low-confidence and abstain entries are in separate review files.)")


# ── Summary statistics ────────────────────────────────────────────────────────

def print_prediction_summary(
    predicted:    dict[str, tuple[str, str, float]],
    review_path:  str,
    abstain_path: str,
) -> None:
    """Print a breakdown of predictions by confidence tier."""
    tiers = Counter(conf for _, (_, conf, _) in predicted.items())
    gender_counts = Counter(cls for _, (cls, _, _) in predicted.items())

    # Count lines in the review and abstain files (subtract header lines).
    def count_data_lines(path: str, header_lines: int = 1) -> int:
        try:
            with open(path, encoding="utf-8") as f:
                return max(0, sum(1 for ln in f if not ln.startswith("#")) - header_lines)
        except FileNotFoundError:
            return 0

    n_review  = count_data_lines(review_path)
    n_abstain = count_data_lines(abstain_path)
    total_processed = len(predicted) + n_review + n_abstain

    print(f"\n{'─' * 60}")
    print(f"  Predictions for {total_processed:,} excluded consonant-final nouns")
    print(f"{'─' * 60}")

    print(f"\n  Disposition breakdown:")
    for tier in ("predicted_high", "predicted_medium"):
        n   = tiers.get(tier, 0)
        pct = 100 * n / total_processed if total_processed else 0
        bar = "█" * int(pct / 2)
        print(f"    {tier:<22} {n:>5}  ({pct:5.1f}%)  {bar}  → merged into lexicon")
    print(f"    {'predicted_low':<22} {n_review:>5}  "
          f"({100*n_review/total_processed if total_processed else 0:5.1f}%)  "
          f"  → {review_path} (review required)")
    print(f"    {'abstain':<22} {n_abstain:>5}  "
          f"({100*n_abstain/total_processed if total_processed else 0:5.1f}%)  "
          f"  → {abstain_path} (no prediction)")

    print(f"\n  Gender distribution (high+medium only):")
    print(f"    Predicted masculine (M2): {gender_counts['M2']:>5}")
    print(f"    Predicted feminine  (F3): {gender_counts['F3']:>5}")

    print(f"\n  Sample high-confidence predictions:")
    shown = 0
    for word, (cls, conf, prob) in sorted(predicted.items()):
        if conf == "predicted_high" and shown < 20:
            gender = "M" if cls == "M2" else "F"
            print(f"    {word:<12} → {cls}  ({gender}, p={prob:.3f})")
            shown += 1

    if n_review:
        print(f"\n  ⚠  {n_review} low-confidence entries written to {review_path}.")
        print(f"     These were NOT added to the lexicon.  Review manually.")
    if n_abstain:
        print(f"\n  ⚠  {n_abstain} entries abstained (p < {THRESHOLD_LOW}).  "
              f"See {abstain_path}.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    FINAL_TSV    = "data/noun_lexicon_final.tsv"
    MERGED_TSV   = "data/noun_lexicon_merged.tsv"
    EXPANDED_TSV = "data/noun_lexicon_expanded.tsv"
    PRED_TSV     = "data/noun_lexicon_predicted.tsv"
    REVIEW_TSV   = "data/noun_lexicon_review.tsv"     # NEW: low-confidence
    ABSTAIN_TSV  = "data/noun_lexicon_abstain.tsv"    # NEW: model abstains
    COMPLETE_TSV = "data/noun_lexicon_complete.tsv"

    print("=" * 60)
    print("  Hindi Noun Gender Predictor")
    print("  Consonant-final nouns: M2 vs F3 classification")
    print("=" * 60)

    print("\n[1/4] Loading training data from noun_lexicon_final.tsv …")
    train_words, train_labels = load_training_data(FINAL_TSV)
    print(f"      {len(train_words):,} consonant-final nouns loaded.")
    dist = Counter(train_labels)
    print(f"      M2 (masculine): {dist['M']:,}   F3 (feminine): {dist['F']:,}")

    print("\n[2/4] Training and evaluating classifier …")
    pipeline = build_pipeline(max_ngram=4, C=1.0)
    evaluate(pipeline, train_words, train_labels, n_folds=5)

    X_all = [word_to_feature_string(w) for w in train_words]
    pipeline.fit(X_all, train_labels)
    print("\n      Final model fitted on full training set.")

    print("\n[3/4] Loading excluded consonant-final words …")
    excluded = load_excluded_words(MERGED_TSV, EXPANDED_TSV)
    print(f"      {len(excluded):,} consonant-final ex_noun words to classify.")

    if not excluded:
        print("      No excluded words found — check file paths and formats.")
        return

    print("\n[4/4] Predicting gender and writing output files …")
    predicted = predict_and_save(
        pipeline, excluded, PRED_TSV, REVIEW_TSV, ABSTAIN_TSV
    )
    print(f"      High+medium predictions written to: {PRED_TSV}")
    print(f"      Low-confidence entries written to:  {REVIEW_TSV}")
    print(f"      Abstain entries written to:         {ABSTAIN_TSV}")

    print_prediction_summary(predicted, REVIEW_TSV, ABSTAIN_TSV)

    merge_into_complete_lexicon(EXPANDED_TSV, predicted, COMPLETE_TSV)

    print("\n" + "=" * 60)
    print("  Done.")
    print(f"  noun_lexicon_predicted.tsv — {len(predicted):,} high/medium entries")
    print(f"  noun_lexicon_review.tsv    — low-confidence; review before use")
    print(f"  noun_lexicon_abstain.tsv   — no prediction possible")
    print(f"  noun_lexicon_complete.tsv  — full lexicon (high/medium only)")
    print("=" * 60)


if __name__ == "__main__":
    main()
