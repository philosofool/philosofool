from typing import TypedDict, Annotated, Optional
from pydantic import TypeAdapter

class Entity(TypedDict):
    """A named entity with a summary description."""

    name: Annotated[str, "The name of the entity."]
    summary: Annotated[str, "A short summary of the entity."]

class Setting(Entity):
    """The description of a location or place."""

    physical_description: Annotated[str, "Important features of the appearance of the place. This may include specific details or general characteristics."]
    history: Annotated[Optional[str], "Important features of the history of the place."]
    people: Annotated[Optional[list[str]], "A list of any people or groups associated with the place."]
    parent: Annotated[Optional[str], "The name of a setting in which this setting is found. For example, a city might have a country as a parent."]

class Character(Entity):
    """The description of a character."""

    background: Annotated[
        str,
        "The history and life details of the character that explain who they are, where they come from and which may hint at their future."]
    appearance: Annotated[str, "The character's outward appearance."]
    personality: Annotated[str, "The character's personality and mannerisms."]
    details: Annotated[
        Optional[str],
        "These are additional details about the character which are updated as a story progresses. Should only be added when specified."]

class List(TypedDict):
    """A list of people, places, concepts, and other things that can be counted or named."""

    items: Annotated[list[str], "Things, people, places, concepts and so forth."]


list_schema = TypeAdapter(List).json_schema()
character_schema = TypeAdapter(Character).json_schema()
setting_schema = TypeAdapter(Setting).json_schema()
entity_schema = TypeAdapter(Entity).json_schema()
