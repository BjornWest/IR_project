import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "long_form_factuality"))

from long_form_factuality.eval.safe import get_atomic_facts
from long_form_factuality.common.modeling import OllamaQwenModel

two_atomic_facts = "I have never been to France, Francis is the most common name in Angola"

response = get_atomic_facts.main(two_atomic_facts, OllamaQwenModel(model_name="qwen2.5:7b"))
print(response)


