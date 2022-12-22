from functools import lru_cache
import time

example_prompt = """Why was Louis XIV considered an absolute monarch?"""


@lru_cache(typed=True)
def get_gpt_response(prompt, length=1024):
    """Get GPT Completion using the prompt"""
    return f"Time is {time.gmtime(0)}"


ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")

ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")

ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")

ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")

ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")

ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")

ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")

ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")

ai_genenerated = get_gpt_response(prompt=example_prompt).lstrip()
print(f"Cache info is {get_gpt_response.cache_info()}")
