"""Request and response schemas for the Podcast Highlight Extractor API."""

from pydantic import BaseModel, Field


class PodcastInput(BaseModel):
    """Input schema: podcast title and full transcript text."""

    title: str = Field(..., description="The podcast episode title")
    transcript: str = Field(..., min_length=1, description="Full transcript text")


class PodcastOutput(BaseModel):
    """Output schema: title and extracted highlight sentences."""

    title: str
    highlights: list[str] = Field(..., description="Top N most relevant sentences")
