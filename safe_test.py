import pathlib
import sys
import os

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
LFF_ROOT = PROJECT_ROOT / "long_form_factuality"  # <-- 2. Define the library root
sys.path.append(str(LFF_ROOT))

from long_form_factuality.eval.safe import get_atomic_facts
from long_form_factuality.common.modeling import OllamaQwenModel

two_atomic_facts = (
    "I have never been to France, Francis is the most common name in Angola"
)

# 3. Save your original directory
original_cwd = os.getcwd()

try:
    # 4. Change to the directory the library expects
    os.chdir(LFF_ROOT)

    # Now this call will work
    response = get_atomic_facts.main(
        two_atomic_facts, OllamaQwenModel(model_name="qwen2.5:7b")
    )
    print(response)

finally:
    # 5. Change back to your original directory after the call
    os.chdir(original_cwd)
