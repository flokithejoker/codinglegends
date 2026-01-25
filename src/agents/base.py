"""Base agent class using LangChain's tool-calling agent pattern."""

import json
import re

# this function comes from codeseeker and is needed to insert json into prompts
def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    def sanitize_value(val):
        # Recursively sanitize strings within nested structures
        if isinstance(val, str):
            # Replace non-printable characters with a space
            return re.sub(r"[^\x20-\x7E]", " ", val)
        return val

    sanitized_value = sanitize_value(value)
    return json.dumps(sanitized_value, ensure_ascii=False)

class BaseAgent:
    """Base agent class."""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, input):
        """Run the agent with the given input text."""
        raise NotImplementedError("Subclasses must implement this method.")