"""Minimal wrapper for VLLMQwenModel to avoid dependency issues."""

import asyncio
import sys
import os
import uuid
from typing import List
import httpx
from time import sleep

port = 8000
# Monkey patch for langfun compatibility with newer openai
import openai
from openai import OpenAI
if not hasattr(openai, 'error'):
    # Create a fake error module for old langfun compatibility
    class FakeErrorModule:
        class OpenAIError(Exception):
            pass
        class RateLimitError(OpenAIError):
            pass
        class APIError(OpenAIError):
            pass
        class Timeout(OpenAIError):
            pass
        class APIConnectionError(OpenAIError):
            pass
        class InvalidRequestError(OpenAIError):
            pass
        class AuthenticationError(OpenAIError):
            pass
        class ServiceUnavailableError(OpenAIError):
            pass
    
    openai.error = FakeErrorModule()

if not hasattr(openai, 'openai_object'):
    # Create a fake openai_object for old langfun compatibility
    class OpenAIObject(dict):
        def __getattr__(self, key):
            return self.get(key)
    openai.openai_object = type('module', (), {'OpenAIObject': OpenAIObject})()

# Add paths for atomic_facts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
lff_root = os.path.join(project_root, "long_form_factuality")
for path in [project_root, lff_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from pydantic import BaseModel, Field
import json
class AtomicFacts(BaseModel):
    atomic_facts: List[str]

    
class VLLMAtomizationModel:
    """Simple thread-safe wrapper for vLLM's synchronous LLM class."""
    
    def __init__(self):
        """
        Args:
            llm_engine: vLLM LLM instance (synchronous)
        """
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            # Increase connection pool size for high concurrency
            http_client=httpx.Client(limits=httpx.Limits(max_keepalive_connections=1000, max_connections=1000))
        )
    
    def generate(self, prompt: str) -> str:
        structured_output = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that breaks down sentences into atomic facts and answers in a json format. IMPORTANT: Only put actual facts in here, if the sentance does not contain any facts, answer with an EMPTY list."},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "guided_json": AtomicFacts.model_json_schema() },
        )
        try:
            as_string = ""
            for fact in json.loads(structured_output.choices[0].message.content)["atomic_facts"]:
                fact_string = fact+"\n"
                if "- " not in fact_string:
                    fact_string = "- " + fact_string
                as_string += fact_string
            return as_string
        except Exception as e:
            print(e)
            return ""


class VLLMRaterModel:
    """Simple thread-safe wrapper for vLLM's synchronous LLM class."""
    
    def __init__(self):
        """
        """
        MAX_REQUEST_TIMEOUT = 1800  # 30 minutes
        long_timeout = httpx.Timeout(
            timeout=MAX_REQUEST_TIMEOUT,  # Total timeout for the entire request
            connect=10.0,                # Connection establishment timeout (seconds)
            read=MAX_REQUEST_TIMEOUT,    # Read/inactivity timeout (seconds)
            write=10.0                   # Write timeout (seconds)
        )
        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
            # Increase connection pool size for high concurrency
            http_client=httpx.Client(
                limits=httpx.Limits(max_keepalive_connections=1500, 
                max_connections=1500),
                timeout=long_timeout
            )
        )

    def generate(self, prompt: str, response_format: BaseModel, debug: bool = False) -> str:
        if debug:
            print("FULL PROMPT: ", prompt)
        try:
            structured_output = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that is either tasked with finding a good search query or deciding if a fact is supported or not. You will be given a prompt and a response format. Pay close attention to the response format and do not deviate from it."},
                    {"role": "user", "content": prompt}
                ],
                extra_body={
                    "guided_json": response_format.model_json_schema() },
            )
        except Exception:
            sleep(5)
            return self.generate(prompt, response_format)
        # validate the output
        try:
            final_answer = response_format.model_validate_json(structured_output.choices[0].message.content)
        except Exception:
            # print("Error validating output: ")
            # print("Retrying...")
            return self.generate(prompt, response_format)
        if debug:
            print("OUTPUT:")
            for key in final_answer.model_fields.keys():
                print(f"{key}: {getattr(final_answer, key, None)}")
        return final_answer


# Keep the old class for backwards compatibility but mark it as deprecated
class VLLMQwenModel:
    """DEPRECATED: Use SimpleLLMWrapper instead with the synchronous LLM class."""

    def __init__(
        self,
        engine,
        loop: asyncio.AbstractEventLoop,
        sampling_params,
    ) -> None:
        self.engine = engine
        self.loop = loop
        self.sampling_params = sampling_params

    async def _process_request_async(self, prompt: str):
        """The actual async logic that communicates with vLLM."""
        request_id = str(uuid.uuid4())

        results_generator = self.engine.generate(prompt, self.sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            return []

        return [o.text for o in final_output.outputs]

    def generate(self, prompt: str) -> List[str]:
        """
        Blocking wrapper for async generation.
        """
        if self.loop.is_closed():
            raise RuntimeError("The background asyncio loop is closed.")

        future = asyncio.run_coroutine_threadsafe(
            self._process_request_async(prompt),
            self.loop
        )
        
        return future.result()


def get_atomic_facts(response: str, model: VLLMAtomizationModel):
    """
    Wrapper for get_atomic_facts that matches the original SAFE implementation exactly.
    
    This reimplements the atomic facts generation without importing common.modeling
    to avoid the langfun dependency issues.
    """
    import json
    import itertools
    import re
    import spacy
    import rank_bm25
    import nltk
    from nltk import tokenize
    
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    # SAFE instruction prefix (exactly as in original)
    ATOMIC_FACT_INSTRUCTION = """\
Instructions:
1. You are given a sentence. Your task is to break the sentence down into a \
list of atomic facts.
2. An atomic fact is a sentence containing a singular piece of information.
3. Each atomic fact in the outputted list should check a different piece of \
information.
4. Use the previous examples to learn how to do this.
5. You should only output the atomic facts as a list, with each item starting \
with "- ". Do not include other formatting.
6. Your task is to do this for the last sentence that is given.

"""
    
    # Load spacy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy model...")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    # Load demons (examples) for prompting
    demon_path = os.path.join(lff_root, "third_party", "factscore", "demos", "demons.json")
    with open(demon_path, 'r') as f:
        demons = json.load(f)
    
    # BM25 for finding relevant examples
    tokenized_corpus = [doc.split(' ') for doc in demons.keys()]
    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
    
    # Split into paragraphs, then sentences (matching original)
    paragraphs = [para.strip() for para in response.split('\n') if para.strip()]
    sentences = []
    
    for paragraph in paragraphs:
        curr_sentences = tokenize.sent_tokenize(paragraph)
        sentences += curr_sentences
    
    # Parameters - using fewer examples to avoid confusion
    # The original uses 7+1 examples but smaller models get confused
    k = 1  # number of top matching demos
    
    atoms = {}
    
    # Process each sentence
    for sentence in sentences:
        if sentence in atoms:
            continue
        
        # Get top k matching demonstrations
        tokenized_query = sentence.split(' ')
        top_matchings = bm25.get_top_n(tokenized_query, list(demons.keys()), k)
        
        # Construct prompt with just the most relevant example
        prompt = ''
        
        # Add top matching example
        for match in top_matchings:
            prompt += 'Please breakdown the following sentence into independent facts: {}\n'.format(match)
            for fact in demons[match]:
                prompt += '- {}\n'.format(fact)
            prompt += '\n'
        
        # Add the target sentence
        prompt += 'Please breakdown the following sentence into independent facts: {}\n'.format(sentence)
        
        # Add instruction prefix
        prompt_to_send = ATOMIC_FACT_INSTRUCTION + prompt
        
        # Generate with model (temperature=0 as in original)
        try:
            responses = model.generate(prompt_to_send, temperature=0.0, top_p=1.0, max_tokens=512)
            if responses:
                output = responses[0]
                
                # Parse output exactly as in original text_to_sentences()
                sentences_from_output = output.split('- ')[1:]
                sentences_from_output = [
                    sentence[:sentence.find('\n')] if '\n' in sentence else sentence
                    for sentence in sentences_from_output
                ]
                sentences_from_output = [
                    sent.strip()[:-1] if sent.strip() and sent.strip()[-1] == '\n' else sent.strip()
                    for sent in sentences_from_output
                ]
                
                # Add period to last fact if missing
                if sentences_from_output and sentences_from_output[-1] and sentences_from_output[-1][-1] != '.':
                    sentences_from_output[-1] = sentences_from_output[-1] + '.'
                
                # Try markdown list format if no facts found
                if not sentences_from_output:
                    sentences_from_output = output.split('* ')[1:]
                    sentences_from_output = [
                        sentence[:sentence.find('\n')] if '\n' in sentence else sentence
                        for sentence in sentences_from_output
                    ]
                    sentences_from_output = [
                        sent.strip()[:-1] if sent.strip() and sent.strip()[-1] == '\n' else sent.strip()
                        for sent in sentences_from_output
                    ]
                    if sentences_from_output and sentences_from_output[-1] and sentences_from_output[-1][-1] != '.':
                        sentences_from_output[-1] = sentences_from_output[-1] + '.'
                
                atoms[sentence] = sentences_from_output
            else:
                atoms[sentence] = []
        except Exception as e:
            print(f"Error generating facts for sentence: {e}")
            atoms[sentence] = []
    
    # Build output pairs matching original format
    atomic_facts_pairs = []
    for sent in sentences:
        # Skip common filler sentences (matching original logic)
        if (sent.startswith('Sure') or sent.startswith('Please') or 
            sent.startswith('This sentence does not contain any facts')):
            atomic_facts_pairs.append((sent, []))
        else:
            atomic_facts_pairs.append((sent, atoms.get(sent, [])))
    
    # Convert to dict format
    facts_as_dict = [
        {'sentence': sentence, 'atomic_facts': identified_atomic_facts}
        for sentence, identified_atomic_facts in atomic_facts_pairs
    ]
    
    all_atomic_facts_list = list(
        itertools.chain.from_iterable([f['atomic_facts'] for f in facts_as_dict])
    )
    
    return {
        'num_claims': len(all_atomic_facts_list),
        'sentences_and_atomic_facts': atomic_facts_pairs,
        'all_atomic_facts': facts_as_dict,
    }

