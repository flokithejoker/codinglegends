from pydantic import BaseModel, Field


class RetrievedTerm(BaseModel):
    """A candidate term returned by the retrieval layer."""

    term_id: str
    code: str
    path: str
    source: str
    score: float = Field(default=0.0)


class RetrievedCode(BaseModel):
    """A candidate ICD code returned by the retrieval layer."""

    code: str
    description: str
    path: str
    source: str
    score: float = Field(default=0.0)


class GuidelineChunk(BaseModel):
    """A compact guideline text chunk."""

    content: str
