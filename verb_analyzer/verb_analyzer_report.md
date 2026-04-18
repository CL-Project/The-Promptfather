# Hindi Verb Analyzer — Problems & Fixes Report

**Project:** The Promptfather — Hindi Morphological Analyzer  
**Component:** Verb Analyzer  

---

## Background

The verb analyzer is a tool that takes an inflected Hindi verb — for example, खाया (ate) — and figures out its base form (खा), along with grammatical details like tense, gender, and number. During testing, we found several issues ranging from missing results to outright wrong answers. This report documents each problem and how it was fixed.

---

## Problem 1: Some Words Returned No Results At All

**What we saw:** Typing खाएं or खाऊं into the analyzer returned nothing, even though these are perfectly normal Hindi verb forms meaning "should eat" and "I will eat" respectively.

**Why it happened:** Hindi script can be stored in computers in more than one way — two strings can look completely identical on screen but be encoded differently under the hood. The analyzer's internal word lists were stored one way, and the words typed in were arriving in a slightly different encoding, so the comparison always failed silently.

**How it was fixed:** We added a normalization step at the very beginning of the analysis process, so that any word coming in gets converted to a standard encoding before anything else happens. This same technique was already being used successfully in the noun analyzer. We also applied the same normalization to all the internal word lists so everything speaks the same language.

---

## Problem 2: Duplicate Results Appearing

**What we saw:** Analyzing पढ़ूंगा (I will read) returned 4 results, two of which were completely identical — the same lemma, the same suffix, the same grammatical features, just listed twice.

**Why it happened:** The analyzer was supposed to keep track of results it had already seen and skip them. But the way it was checking for duplicates had a flaw — two results that were effectively identical could still slip through as "different" because of how the comparison was being done.

**How it was fixed:** Two things were done. First, the duplicate-checking comparison was simplified so it works correctly. Second, a cleanup step was added at the very end of the analysis process that sweeps through all results one final time and removes any remaining duplicates before anything is returned. This acts as a safety net regardless of what happened earlier.

---

## Problem 3: A Valid Reading Was Being Missed (खाए)

**What we saw:** Analyzing खाए (ate, masculine plural) only returned subjunctive readings. The perfective (past) reading — which is equally valid and arguably more common — was completely missing.

**Why it happened:** The analyzer processes possible meanings in a fixed priority order. The word ए was being caught by the subjunctive check before the perfective check ever got a chance to run. The code did try to add the perfective reading back as an afterthought, but the logic was inverted — it only added it when the subjunctive was the alternative, not when the subjunctive was the primary.

**How it was fixed:** A dedicated list of suffixes that are genuinely ambiguous was created. For these specific cases, the analyzer no longer runs through the priority chain at all — instead it directly returns all valid readings at once. For खाए, this means both the perfective and the subjunctive readings are now returned together, which is linguistically correct since the word truly can mean either thing depending on context.

---

## Problem 4: Nonsense Lemmas Being Generated

**What we saw:** Analyzing खाता (habitually eats) returned खात as a possible base form. Analyzing गया (went) returned ग and गय as possible base forms. None of these are real Hindi words, let alone real verb stems.

**Why it happened:** The analyzer works by stripping endings off words to recover the base form. It has a large list of possible endings to try, and it tries all of them without checking whether the result makes any linguistic sense. Very short, generic endings like ा were matching and producing one- or two-character fragments that aren't real stems.

**How it was fixed:** A plausibility check was added. Before accepting any recovered base form, the analyzer now checks that it meets a minimum length requirement. Results that are implausibly short are discarded immediately. The check is intentionally simple for now — it catches the worst offenders without being so strict that it rejects legitimate forms. It can be tightened further as more testing reveals edge cases.

---

## Problem 5: A Specific Wrong Lemma (पढ़ूंगा → पढ़ै)

**What we saw:** One of the results for पढ़ूंगा (I will read) showed पढ़ै as the base form. This is not a real Hindi word at all.

**Why it happened:** One of the rules in the underlying data file had a mistake — it was stripping a future-tense ending and adding back ै, which is a vowel sound that doesn't appear at the end of any real Hindi verb stem. Because the analyzer blindly trusted its data files, this bad rule produced a nonsense result.

**How it was fixed:** A warning system was added to the data file loader. Whenever it loads a rule that adds back a single vowel character as the stem ending, it now flags it as suspicious. Running this once produced a list of questionable rules for manual review, making it easy to identify and remove or correct the bad entries in the data file itself.

---

## Problem 6: No Way to Know Which Results to Trust

**What we saw:** Even after the above fixes, the analyzer was still returning a mix of high-quality results (like correctly identifying गया as an irregular past form of जाना) and lower-quality speculative results (like the जunk lemmas from Problem 4). There was no way for someone using the analyzer to tell which was which.

**How it was fixed:** Each result now carries a confidence label — high, medium, or low. Irregular forms that are directly looked up in a known list get marked high. Results from suffix rules where the ending is long and specific get marked medium. Short, generic suffix matches get marked low. Results are also now sorted so that the most trustworthy answers appear first. A downstream application can choose to show only high-confidence results, or show everything but display the confidence level to the user.

---

## Verification

A set of 13 test cases was written covering the full range of verb forms — perfective, imperfective, future, infinitive, imperative, subjunctive, and irregular. After all fixes were applied, all 13 passed. These tests now serve as a permanent safety net: any future change to the analyzer can be checked against them instantly to make sure nothing that was working before has been broken.

---

## Summary

| Problem | Impact | Fixed |
|---|---|---|
| Encoding mismatch | Valid words returned no results | Yes |
| Duplicate results | Same answer listed multiple times | Yes |
| Missing perfective reading for खाए | Correct meaning not returned | Yes |
| Nonsense lemmas from short suffixes | Wrong base forms in results | Yes |
| Bad rule producing पढ़ै | Wrong base form for future forms | Yes |
| No way to distinguish reliable vs speculative results | All results looked equally valid | Yes |
