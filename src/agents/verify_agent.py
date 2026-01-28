from pydantic import BaseModel, Field

from src.agents.base import BaseAgent


class VerifyOutput(BaseModel):
    code_ids: list[int] = Field(description="IDs of verified codes, or [0] if none apply")


class VerifyAgent(BaseAgent):
    output_schema = VerifyOutput

    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(provider, model, prompt_name="verify_codeseeker", **kwargs)
