# Copyright 2024 Google LLC
# ... (Copyright header) ...
"""Rates a single atomic fact for accuracy."""

import dataclasses
import re
from typing import Any

# --- CUSTOM IMPORTS ---
# We import the engine we created in File 1
from bm25 import bm25_engine 
# ----------------------

# pylint: disable=g-bad-import-order
from common import modeling
from common import shared_config
from common import utils
from eval.safe import config as safe_config
# Removed: from eval.safe import query_serper (No longer needed)
# pylint: enable=g-bad-import-order
from pydantic import BaseModel, Field
from typing import Literal

SUPPORTED_LABEL = 'Supported'
NOT_SUPPORTED_LABEL = 'Not Supported'

_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_ORIGINAL_TOPIC_PLACEHOLDER = '[ORIGINAL_TOPIC]'
_NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does not \
support the factual accuracy of the given STATEMENT.
3. To do this, you are allowed to issue ONE Google search query that you think will \
allow you to find additional useful evidence. \
4. Your query should aim to obtain new information that does not appear in the \
KNOWLEDGE. If you have previous search results, look at the previous queries you \
have made and try to construct a new query that is meaningfully different from the \
previous queries.
5. You will provide your query in a JSON object with the following fields:
   - reasoning: your full reasoning process for constructing the query
   - search_query: the actual query to be issued
6. The STATEMENT was gathered from a biography about "{_ORIGINAL_TOPIC_PLACEHOLDER}" I \
would highly recommend using this in one of your queries.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_FINAL_ANSWER_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either "{SUPPORTED_LABEL}" or \
"{NOT_SUPPORTED_LABEL}". Wrap your final answer in square brackets.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


@dataclasses.dataclass()
class GoogleSearchResult:
    query: str
    result: str


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


def call_search(
    search_query: str,
    search_type: str = safe_config.search_type,
    num_searches: int = safe_config.num_searches,
    serper_api_key: str = shared_config.serper_api_key, # Argument kept for compatibility, but unused
    search_postamble: str = '', 
) -> str:
    """Call BM25 Index to get the search result."""
    
    # Clean up query
    search_query = search_query.strip()
    if search_postamble:
        search_query += f' {search_postamble}'


    # --- REPLACEMENT LOGIC ---
    # Instead of calling Serper, we call our local BM25 engine
    # We assume num_searches corresponds to 'k' results
    try:
        results = bm25_engine.search(search_query, k=num_searches)
        
        # If no results found
        if not results:
            return "No relevant documents found in the local index."
            
        return results
        
    except Exception as e:
        print(f"Error during BM25 search: {e}")
        return "Error retrieving documents."
    # -------------------------


class SearchQueryFormat(BaseModel):
    reasoning: str = Field(description="Your full reasoning process for constructing the query")
    search_query: str = Field(description="The actual query to be issued")

def maybe_get_next_search(
    atomic_fact: str,
    original_topic: str,
    past_searches: list[GoogleSearchResult],
    model: modeling.Model,
    debug: bool = safe_config.debug_safe,
) -> GoogleSearchResult | None:
    """Get the next query from the model."""
    knowledge = '\n'.join([s.result for s in past_searches])
    knowledge = 'N/A' if not knowledge else knowledge
    full_prompt = _NEXT_SEARCH_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = full_prompt.replace(_ORIGINAL_TOPIC_PLACEHOLDER, original_topic)
    full_prompt = utils.strip_string(full_prompt)
    model_response = model.generate(full_prompt, response_format=SearchQueryFormat)
    query = model_response.search_query

    if model_response and query:
        return GoogleSearchResult(query=query, result=call_search(query))

    return None

class FinalAnswerFormat(BaseModel):
    reasoning: str = Field(description="Your full reasoning process for determining the final answer")
    final_answer: Literal["Supported", "Not Supported"] = Field(description="The final answer to the question")

def maybe_get_final_answer(
    atomic_fact: str,
    searches: list[GoogleSearchResult],
    model: modeling.Model,
    debug: bool = safe_config.debug_safe,
) -> FinalAnswer | None:
    """Get the final answer from the model."""
    knowledge = '\n'.join([search.result for search in searches])
    full_prompt = _FINAL_ANSWER_FORMAT.replace(
        _STATEMENT_PLACEHOLDER, atomic_fact
    )
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    model_response = model.generate(full_prompt, response_format=FinalAnswerFormat)
    answer = model_response.final_answer
    # answer = utils.extract_first_square_brackets(model_response)
    # answer = re.sub(r'[^\w\s]', '', answer).strip()

    if model_response and answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
        return FinalAnswer(response=model_response.reasoning, answer=answer)

    return None


def check_atomic_fact(
    atomic_fact: str,
    original_topic: str,
    rater: modeling.Model,
    max_steps: int = safe_config.max_steps,
    max_retries: int = safe_config.max_retries,
    debug: bool = safe_config.debug_safe,
) -> tuple[FinalAnswer | None, dict[str, Any]]:
    """Check if the given atomic fact is supported."""
    search_results = []

    for _ in range(max_steps):
        next_search, num_tries = None, 0

        while not next_search and num_tries <= max_retries:
            next_search = maybe_get_next_search(atomic_fact, original_topic, search_results, rater)
            num_tries += 1

        if next_search is None:
            utils.maybe_print_error('Unsuccessful parsing for `next_search`')
            break
        else:
            search_results.append(next_search)

    search_dicts = {
        'google_searches': [dataclasses.asdict(s) for s in search_results]
    }
    final_answer, num_tries = None, 0

    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer = maybe_get_final_answer(
            atomic_fact, searches=search_results, model=rater, debug=debug
        )

    if final_answer is None:
        utils.maybe_print_error('Unsuccessful parsing for `final_answer`')

    return final_answer, search_dicts