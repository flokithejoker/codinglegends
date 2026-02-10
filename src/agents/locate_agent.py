from pydantic import BaseModel, Field

from agents.base import BaseAgent


class LocateOutput(BaseModel):
    term_ids: list[int] = Field(description="IDs of relevant terms, or [0] if none apply")


class LocateAgent(BaseAgent):
    output_schema = LocateOutput

    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(provider, model, prompt_name="locate_codeseeker", **kwargs)
