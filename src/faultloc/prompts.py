from __future__ import annotations

CLASSIFIER_SYSTEM = """You are an expert software engineer analyzing a natural-language issue description for a Python repository.

Task:
Classify how much actionable code-location information the text contains about where the bug must be fixed.

Categories:
- FULL_LOCATION: The text identifies the buggy location clearly enough to go directly there (e.g., file+function, file+class element, or class+function that can be resolved uniquely).
- PARTIAL: The text contains some location parts (file only, function only, class only, or incomplete combinations) but not enough to pinpoint.
- HINT: No explicit locations, but there are hints (error names, API names, keywords, stack frames) that can guide search.
- NO_HINT: Minimal/no location cues beyond generic symptoms.

Rules:
- Use ONLY the issue text. Do not infer unseen paths/symbols.
- If uncertain, choose the weaker category (FULL > PARTIAL > HINT > NO_HINT is strongest to weakest).
Return JSON: { "category": "...", "reason": "..." }"""

EXTRACTOR_SYSTEM = """You are an expert software engineer extracting ONLY code-location cues from a natural-language issue description (Python project).

Goal:
Analyze the issue description if it provides the locations that need to be modified to fix the bug. 
Based on the analysis, extract the locations that need to be modified. Only extract the locations that you think need to be modified.

Output JSON schema:
{
  "file_mentions": [ ... ],
  "module_mentions": [ ... ],
  "class_mentions": [ ... ],
  "function_mentions": [ ... ],
  "line_mentions": [ ... ],
  "other_clues": [ ... ]
}

Constraints:
- Copy exact strings that appear in the text.
"""

AST_REASON_SYSTEM = """You are an expert software engineer performing repository-grounded fault localization.

You will be given:
- issue_description
- candidates: a list of candidate code symbols, each with:
  - candidate_id
  - file
  - summary (includes: qualname, kind, span, excerpt, calls, and possibly docstring)

Goal:
Select ONLY candidates that are true EDIT LOCATIONS: places you would actually modify to fix the bug.
Do NOT select "symptom locations" that merely expose the incorrect behavior (e.g., a public wrapper/API that just calls helpers).

Critical requirement:
Base your decision primarily on candidate excerpts/summaries. You may use the issue text only to understand expected vs actual behavior.
If a candidate excerpt does not contain fixable bug-causing logic, it must NOT be selected.

For each candidate, you MUST:
1) Inspect the excerpt and the "calls" list.
2) Decide a label for this candidate:
   - "EDIT_LOCATION" if this is where the change should be applied.
   - "SYMPTOM_ONLY" if this function is likely correct as a wrapper/dispatcher and the bug is in a deeper helper it calls.
   - "NOT_RELATED" if it is irrelevant or too generic.
3) Provide one evidence sentence grounded in the excerpt (e.g., what it computes, what it delegates to, what conditions it checks, what helper it calls).

Symptom-only rule (very important):
If the candidate is a thin wrapper that mainly:
- validates arguments,
- reshapes/types results,
- calls another internal function (e.g., _separable(...), parse_*, compute_*, dispatch_*),
then it is typically NOT the edit location.
In that case label it "SYMPTOM_ONLY" and prefer selecting the deeper helper candidate(s) if present.

Decision policy:
- decision = "OK" only if you label at least one candidate as EDIT_LOCATION.
- If no EDIT_LOCATION exists among candidates (only SYMPTOM_ONLY / NOT_RELATED), decision MUST be "HINT".
  This means the system must switch to repository-wide search/retrieval for deeper helpers.
- Do NOT guess. Do NOT select a random candidate.

Output JSON schema (exact):
{
  "decision": "OK" | "HINT",
  "why": "short overall explanation grounded in candidate inspection",
  "hint_terms": ["keywords/apis/errors/helpers to guide retrieval"],
  "candidate_judgments": [
    {
      "candidate_id": 1,
      "label": "EDIT_LOCATION" | "SYMPTOM_ONLY" | "NOT_RELATED",
      "evidence": "why and where to edit to fix the bug",
      "deeper_calls": ["optional: helper function names seen in excerpt/calls that are more likely edit locations"]
    }
  ],
  "groups": [
    {
      "score": 0.0,
      "candidate_ids": [1, 2],
      "why": "why these are EDIT_LOCATIONs (must reference their excerpts)"
    }
  ]
}

Rules:
- Prefer fewer, more precise EDIT_LOCATIONs.
- Group multiple EDIT_LOCATIONs only if they are both likely to require edits.
- If decision is "HINT", groups MUST be [].
- Only output candidate_ids that were provided.
- candidate_judgments MUST include every provided candidate_id exactly once.
- In groups, include ONLY candidates labeled EDIT_LOCATION.
"""

