from pydantic import BaseModel, Field

from agents.base import BaseAgent


class AnalyseOutput(BaseModel):
    terms: list[str] = Field(description="Normalized medical terms extracted from the clinical note")


class AnalyseAgent(BaseAgent):
    output_schema = AnalyseOutput

    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(provider, model, prompt_name="analyse_codeseeker", **kwargs)
