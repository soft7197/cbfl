from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..artifacts import ArtifactLogger
from ..budget import BudgetManager
from ..config import CONFIG
from ..embeddings import CodeBERTEmbedder
from ..openai_client import OpenAIClient
from ..types import BudgetState, Category, CueBundle, FLResult
from ..repo_index.index import build_repo_index
from ..cues.extractor import CueExtractor
from ..cues.normalizer import normalize_cues
from ..retrieval.retriever import HybridRetriever
from .full_location import run_full_location
from .partial import run_partial_location
from .hint import run_hint_or_nohint
from .no_hint import expand_files_with_import_graph
from ..classify.resolver_informed import resolver_informed_classify

def run_faultloc_for_instance(
    instance_id: str,
    problem_statement: str,
    repo_path: Path,
    artifacts_root: Path,
    topk: int,
    client: Optional[OpenAIClient] = None,
    evaluate_patch: Optional[str] = None,
) -> Dict[str, Any]:
    client = client or OpenAIClient()
    embedder = CodeBERTEmbedder(
        model_name=CONFIG.embed_model,
        device=CONFIG.embed_device,
        batch_size=CONFIG.embed_batch_size,
        max_length=CONFIG.embed_max_length,
    )
    logger = ArtifactLogger(artifacts_root, instance_id)

    budget_state = BudgetState(
        time_limit_sec=CONFIG.time_limit_sec,
        cost_limit_usd=CONFIG.cost_limit_usd,
        started_at=time.monotonic(),
    )
    budget = BudgetManager(budget_state)
    logger.write("00_meta.json", {"instance_id": instance_id, "repo_path": repo_path.as_posix(), "budget": budget.snapshot()})
    logger.write_text("00_problem_statement.txt", problem_statement)

    # 1) Build repo index (cached per instance_id)
    repo_index = build_repo_index(repo_path)
    logger.write("04_repo_index/file_index.json", repo_index["file_index"])
    logger.write("04_repo_index/import_graph.json", repo_index["import_graph"])
    # symbol index can be large; store as-is (you can later chunk if needed)
    logger.write("04_repo_index/symbol_index.json", repo_index["symbol_index"])

    budget.ensure_within_budget()

    # 2) Extract cues (LLM, temp=0)
    extractor = CueExtractor(client, CONFIG.llm_model, CONFIG.extraction_temperature)
    ext_out = extractor.extract(problem_statement)
    logger.write("01_extractor.json", ext_out)
    budget.add_cost_from_usage(ext_out.get("usage"), CONFIG.price_input_per_1k, CONFIG.price_output_per_1k)
    budget.ensure_within_budget()
    cues = CueExtractor.to_bundle(ext_out["result"])

    # 3) Normalize / derive variants
    norm = normalize_cues(cues, problem_statement)
    norm_dict = {
        "file_candidates": norm.file_candidates,
        "module_candidates": norm.module_candidates,
        "class_candidates": norm.class_candidates,
        "function_candidates": norm.function_candidates,
        "line_numbers": norm.line_numbers,
    }
    logger.write("02_cues_normalized.json", {"normalized": norm_dict, "raw": cues.__dict__})
    budget.ensure_within_budget()

    # 4) Resolver-informed classification (deterministic)
    category, cls_details = resolver_informed_classify(norm_dict, repo_index)

    # If no location parts, decide HINT vs NO_HINT using extracted other_clues
    if category == Category.NO_HINT:
        if cues.other_clues:
            category = Category.HINT
            cls_details["decision"] = {"rule": "other_clues_present"}
    logger.write("03_classifier.json", {"category": category.value, "details": cls_details})
    budget.ensure_within_budget()

    # 5) Category-specific pipeline
    candidates = []
    retrieval_meta = None
    retrieval_results = None

    if category == Category.FULL_LOCATION:
        candidates = run_full_location(problem_statement, norm_dict, repo_index, topk=topk, max_locs=CONFIG.max_locations_per_group)
    elif category == Category.PARTIAL:
        candidates, partial_fallback = run_partial_location(
            client=client,
            model=CONFIG.llm_model,
            problem_statement=problem_statement,
            normalized=norm_dict,
            repo_root=repo_path,
            repo_index=repo_index,
            topk=topk,
            max_locs=CONFIG.max_locations_per_group,
            allow_ast_reasoning=budget_state.allow_ast_reasoning,
        )

        if partial_fallback is not None:
            # Reroute to HINT pipeline (hybrid retrieval)
            logger.write("03b_partial_fallback.json", partial_fallback)

            category = Category.HINT
            logger.write("03_classifier.json", {"category": category.value, "details": {**cls_details, "decision": {"rule": "partial_ast_fallback_to_hint"}}})

            retriever = HybridRetriever(embedder)
            retrieval_meta = retriever.build(
                repo_path,
                repo_index,
                budget,
                max_files=CONFIG.max_files_for_embedding,
                max_funcs=CONFIG.max_functions_for_embedding,
            )
            logger.write("05_retrieval_build.json", {"meta": retrieval_meta, "budget": budget.snapshot()})
            budget.ensure_within_budget()

            # Use hint_terms if provided to enrich the retrieval query
            hint_terms = partial_fallback.get("hint_terms") or []
            query_text = problem_statement
            if hint_terms:
                query_text = problem_statement + "\n\nHint terms:\n- " + "\n- ".join(map(str, hint_terms))

            retrieval_results = retriever.query(query_text, top_files=50, top_funcs=120)
            logger.write("05_retrieval.json", retrieval_results)

            candidates = run_hint_or_nohint(
                problem_statement,
                repo_path,
                repo_index,
                retrieval_results,
                topk=topk,
                max_locs=CONFIG.max_locations_per_group,
            )

    else:
        # HINT / NO_HINT: build retrieval index once per repo
        retriever = HybridRetriever(embedder)
        retrieval_meta = retriever.build(
            repo_path,
            repo_index,
            budget,
            max_files=CONFIG.max_files_for_embedding,
            max_funcs=CONFIG.max_functions_for_embedding,
        )
        logger.write("05_retrieval_build.json", {"meta": retrieval_meta, "budget": budget.snapshot()})
        budget.ensure_within_budget()

        retrieval_results = retriever.query(problem_statement, top_files=50, top_funcs=120)
        logger.write("05_retrieval.json", retrieval_results)

        if category == Category.NO_HINT and budget_state.allow_retrieval_expand:
            # expand files using import graph
            expanded_files = expand_files_with_import_graph(retrieval_results.get("file_hits", []), repo_index)
            logger.write("05_retrieval_expanded_files.json", {"expanded_files": expanded_files})
            # Rebuild retrieval_results with expanded files boosted (simple approach)
            # For now, just let downstream use original results; expanded files can be consumed later if you want.
        candidates = run_hint_or_nohint(problem_statement, repo_path, repo_index, retrieval_results, topk=topk, max_locs=CONFIG.max_locations_per_group)

    # 6) Save candidates
    logger.write("07_candidates.json", {
        "instance_id": instance_id,
        "category": category.value,
        "candidates": [candidate_group_to_dict(g) for g in candidates],
        "budget": budget.snapshot(),
    })

    # 7) Optional evaluation
    eval_out = None
    if evaluate_patch:
        from ..eval.evaluator import evaluate_against_patch
        eval_out = evaluate_against_patch(repo_path, repo_index, candidates, evaluate_patch)
        logger.write("09_evaluation.json", eval_out)

    return {
        "instance_id": instance_id,
        "category": category.value,
        "topk": topk,
        "candidates_path": str((logger.root / "07_candidates.json").as_posix()),
        "evaluation": eval_out,
    }

def candidate_group_to_dict(g) -> Dict[str, Any]:
    return {
        "rank": g.rank,
        "score": g.score,
        "rationale": g.rationale,
        "locations": [
            {
                "type": l.location_type.value,
                "file": l.file_path,
                "qualname": l.qualname,
                "span": {"start": l.span.start_line, "end": l.span.end_line},
                "anchor_line": l.anchor_line,
                "confidence": l.confidence,
                "evidence": l.evidence,
            }
            for l in g.locations
        ]
    }
