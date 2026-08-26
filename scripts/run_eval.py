import sys
sys.path.insert(0, ".") # if running standalone adjust as needed

from eval import run_comparison
from eval_dataset import EVAL_PROMPTS

CONTROL_SYSTEM_MESSAGE = """You are a helpful assistant. Improve this prompt.""" # should this be a better prompt?

PERSONA_SYSTEM_MESSAGE = """You are a prompt engineering expert. Your job is to improve 
    prompts so they produce better results from language models.
    
    You will receive an original prompt and a goal describing what the prompt should 
    accomplish.
    Return your response as a JSON object with exactly these fields:
    - "optimized_prompt": the improved version of the prompt
    - "changes": a brief explanation of what you improved and why
    
    Return ONLY the JSON object. No markdown formatting, no extra text."""

if __name__ == "__main__":
    result = run_comparison(CONTROL_SYSTEM_MESSAGE, PERSONA_SYSTEM_MESSAGE, EVAL_PROMPTS)
    print(f"\np-value: {result['p_value']:.4f}")
    print(f"median difference: {result['median_difference']:+.2f}")
    print(f"compared: {result['n_compared']}, skipped: {result['n_skipped']}")
    print(f"\n{result['conclusion']}\n")
    for r in result["per_prompt_results"][:3]:
        print(f"   [{r['prompt'][:40]}] A={r['score_a']} B={r['score_b']} - {r['reasoning']}")