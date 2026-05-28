NER_SYSTEM = """
# Role
You are an expert Information Extraction system specialized in Named Entity Recognition (NER).

# Tasks
Identify and extract all unique named entities from the provided text. This includes, but is not limited to:
- Persons (Actors, Characters, Directors)
- Organizations (Studios, Radio Stations)
- Locations (Countries, Cities)
- Dates/Time periods
- Creative Works (Movies, Songs, Portals)

# Constraints
- Output ONLY a valid JSON object.
- Key: "named_entities", Value: A list of strings.
- Each entity must be unique.
- Do not include any explanation or conversational text.
"""

NER_ONE_SHOT_PARAGRAPH = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

NER_ONE_SHOT_OUTPUT = """{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]}"""