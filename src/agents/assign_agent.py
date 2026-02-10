from pydantic import BaseModel, Field

from agents.base import BaseAgent


class AssignOutput(BaseModel):
    code_ids: list[int] = Field(description="IDs of assignable codes, or [0] if none apply")


class AssignAgent(BaseAgent):
    output_schema = AssignOutput

    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(provider, model, prompt_name="assign_codeseeker", **kwargs)
