"""Prompt templates for subagent analysis and transformation tasks.

WS9: Subagent Execution (LLM Integration)
WS9.3 & WS9.4: Analysis and Transformation Prompts

This module contains all prompt templates used by the subagent executor
to instruct LLMs for various analysis and transformation tasks.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

# =============================================================================
# Analysis Task Prompts (WS9.3)
# =============================================================================


ANALYSIS_SYSTEM_BASE = """You are a document analysis assistant. You will analyze a text chunk and extract specific information based on the task.

CRITICAL RULES:
1. Respond ONLY with valid JSON - no markdown code blocks, no explanations.
2. Analyze ONLY the provided text chunk - do not make assumptions about content outside it.
3. Be precise with line references - use 0-indexed line numbers relative to the chunk.
4. If you cannot find the requested information, return empty arrays/null values rather than guessing.
"""


ANALYSIS_PROMPTS = {
    "characters": """Analyze this document chunk and extract all characters mentioned.

For each character, identify:
- Name (primary name used)
- Aliases (nicknames, titles, alternative names)
- Role (protagonist, antagonist, supporting, mentioned)
- Key traits or descriptions mentioned
- Relationships with other characters (if shown in this chunk)
- Mentions with line numbers

Respond with JSON in this exact format:
{
    "characters": [
        {
            "name": "string",
            "aliases": ["string"],
            "role": "protagonist|antagonist|supporting|mentioned",
            "traits": ["string"],
            "relationships": [{"character": "string", "relationship": "string"}],
            "mentions": [{"line": 0, "context": "brief quote"}]
        }
    ]
}""",

    "plot": """Analyze this document chunk for plot elements.

Identify:
- Key events or scenes (what happens)
- Conflict points (sources of tension)
- Character decisions/actions (important choices)
- Setting details (where/when)
- Foreshadowing or callbacks (hints at future/past events)
- Tension level of this chunk

Respond with JSON in this exact format:
{
    "plot_points": [
        {
            "type": "event|conflict|decision|revelation|description",
            "summary": "brief description of what happens",
            "characters_involved": ["name1", "name2"],
            "significance": "low|medium|high",
            "line_start": 0,
            "line_end": 10
        }
    ],
    "tension_level": "low|building|high|climax|resolution",
    "setting": "description of setting if mentioned"
}""",

    "style": """Analyze the writing style in this document chunk.

Assess:
- Narrative voice (person: first/second/third, tense: past/present/future)
- Tone (formal, casual, dark, humorous, lyrical, etc.)
- Sentence structure (simple, complex, varied)
- Dialogue style and frequency
- Description density (sparse, moderate, rich)
- Pacing (fast, moderate, slow)
- Notable stylistic techniques (metaphor, alliteration, stream-of-consciousness, etc.)

Respond with JSON in this exact format:
{
    "style": {
        "voice": {
            "person": "first|second|third",
            "tense": "past|present|future|mixed"
        },
        "tone": ["string"],
        "sentence_complexity": "simple|moderate|complex|varied",
        "dialogue_frequency": "none|sparse|moderate|heavy",
        "description_density": "sparse|moderate|rich",
        "pacing": "fast|moderate|slow",
        "techniques": ["string"]
    }
}""",

    "summary": """Summarize this document chunk concisely.

Focus on:
- Main events or developments
- Key character actions and dialogue
- Important information revealed
- Setting changes or transitions

Keep the summary to 2-4 sentences that capture the essential content.

Respond with JSON in this exact format:
{
    "summary": "Your 2-4 sentence summary here."
}""",

    "themes": """Identify themes present in this document chunk.

Look for:
- Major themes (love, death, justice, redemption, identity, etc.)
- Motifs (recurring symbols, images, or ideas)
- Subtext or underlying messages
- Cultural or historical references

Respond with JSON in this exact format:
{
    "themes": ["theme1", "theme2"],
    "motifs": [
        {"symbol": "what is repeated", "meaning": "what it might represent"}
    ],
    "subtext": "underlying message if any, or null"
}""",
}


def get_analysis_prompt(analysis_type: str, custom_prompt: str | None = None) -> str:
    """Get the analysis prompt for a given type.
    
    Args:
        analysis_type: Type of analysis (characters, plot, style, summary, themes, custom)
        custom_prompt: Custom prompt for 'custom' type
        
    Returns:
        The full prompt text
    """
    if analysis_type == "custom" and custom_prompt:
        return f"""{ANALYSIS_SYSTEM_BASE}

{custom_prompt}

Respond with valid JSON containing your analysis results."""
    
    prompt = ANALYSIS_PROMPTS.get(analysis_type)
    if not prompt:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    return f"{ANALYSIS_SYSTEM_BASE}\n\n{prompt}"


# =============================================================================
# Transformation Task Prompts (WS9.4)
# =============================================================================


TRANSFORM_SYSTEM_BASE = """You are a document transformation assistant. You will transform text according to specific instructions while preserving the story's essence.

CRITICAL RULES:
1. Respond ONLY with valid JSON - no markdown code blocks, no explanations.
2. Transform the COMPLETE provided text - do not omit any content.
3. Preserve the fundamental story/meaning while applying the transformation.
4. Maintain approximately the same length (within 20% of original).
5. Keep dialogue intact unless the transformation specifically requires changing it.
"""


TRANSFORM_PROMPTS = {
    "style_rewrite": """Rewrite this text in a different style.

Target style: {target_style}
{style_guidelines}

Instructions:
- Maintain the same plot, events, and information
- Adapt vocabulary, sentence structure, and tone to match the target style
- Preserve character voices and distinct speaking patterns where appropriate
- Keep the same overall length (within 20%)

Respond with JSON in this exact format:
{{
    "transformed_content": "The complete transformed text",
    "style_changes": ["list of key style changes made"]
}}""",

    "setting_change": """Transform this text by changing the setting/location.

Original setting: {old_setting}
New setting: {new_setting}
Cultural adaptations: {cultural_details}

Instructions:
- Replace references to the original setting with the new setting
- Adapt cultural details, landmarks, customs appropriately
- Maintain narrative consistency and character behavior
- Handle proper nouns and place references thoughtfully

Respond with JSON in this exact format:
{{
    "transformed_content": "The complete transformed text",
    "replacements": 0,
    "adaptations": ["list of cultural adaptations made"]
}}""",

    "tense_change": """Transform this text by changing the narrative tense.

From: {from_tense} tense
To: {to_tense} tense

Instructions:
- Convert all narrative/prose verbs from {from_tense} to {to_tense} tense
- Keep dialogue unchanged (characters speak in natural tense)
- Maintain temporal relationships between events
- Handle irregular verbs correctly

Respond with JSON in this exact format:
{{
    "transformed_content": "The complete transformed text",
    "verbs_changed": 0
}}""",

    "pov_change": """Transform this text by changing the point of view.

From: {from_pov}
To: {to_pov}
Focal character: {focal_character}

Instructions:
- Change pronouns and perspective accordingly
- Adjust internal thoughts to match the new POV:
  - First person: "I thought..." 
  - Third limited: "She thought..."
  - Third omniscient: Show multiple characters' thoughts
- For limited POV, you can only know the focal character's thoughts
- Maintain narrative consistency throughout

Respond with JSON in this exact format:
{{
    "transformed_content": "The complete transformed text",
    "pov_adjustments": ["list of key POV changes made"]
}}""",

    "character_rename": """Rename a character throughout this text.

Old name: {old_name}
New name: {new_name}
Aliases to rename: {aliases}
Pronoun updates: {pronoun_instructions}

Instructions:
- Replace all occurrences of the old name with the new name
- Handle possessive forms (e.g., "{old_name}'s" → "{new_name}'s")
- Update any specified aliases consistently
- Update pronouns if instructed
- Preserve dialogue attribution and character voice

Respond with JSON in this exact format:
{{
    "transformed_content": "The complete transformed text",
    "replacements": 0,
    "locations": [0, 5, 12]
}}""",

    "custom": """Transform this text according to the following instructions:

{custom_prompt}

Respond with JSON in this exact format:
{{
    "transformed_content": "The complete transformed text",
    "changes_made": ["list of key changes made"]
}}""",
}


def get_transform_prompt(
    transform_type: str,
    params: dict[str, Any],
) -> str:
    """Get the transformation prompt for a given type.
    
    Args:
        transform_type: Type of transformation
        params: Parameters to format into the prompt
        
    Returns:
        The full prompt text
    """
    template = TRANSFORM_PROMPTS.get(transform_type)
    if not template:
        raise ValueError(f"Unknown transformation type: {transform_type}")
    
    # Build format args based on type
    format_args: dict[str, str] = {}
    
    if transform_type == "style_rewrite":
        format_args["target_style"] = params.get("target_style", "")
        guidelines = params.get("style_guidelines", "")
        format_args["style_guidelines"] = f"Additional guidelines: {guidelines}" if guidelines else ""
        
    elif transform_type == "setting_change":
        format_args["old_setting"] = params.get("old_setting", "")
        format_args["new_setting"] = params.get("new_setting", "")
        format_args["cultural_details"] = params.get("cultural_details", "Adapt as appropriate")
        
    elif transform_type == "tense_change":
        # Support both target_tense (simpler) and explicit from_tense/to_tense
        to_tense = params.get("to_tense") or params.get("target_tense", "")
        from_tense = params.get("from_tense", "")
        # If only target_tense provided, infer the opposite as from_tense
        if not from_tense and to_tense:
            from_tense = "present" if to_tense == "past" else "past"
        format_args["from_tense"] = from_tense
        format_args["to_tense"] = to_tense
        
    elif transform_type == "pov_change":
        # Support both target_pov (simpler) and explicit from_pov/to_pov
        to_pov = params.get("to_pov") or params.get("target_pov", "")
        from_pov = params.get("from_pov", "")
        # If only target_pov provided, use "current" as placeholder
        if not from_pov and to_pov:
            from_pov = "current"
        format_args["from_pov"] = from_pov
        format_args["to_pov"] = to_pov
        format_args["focal_character"] = params.get("focal_character", "the protagonist")
        
    elif transform_type == "character_rename":
        format_args["old_name"] = params.get("old_name", "")
        format_args["new_name"] = params.get("new_name", "")
        format_args["aliases"] = ", ".join(params.get("aliases", [])) or "none"
        if params.get("update_pronouns"):
            format_args["pronoun_instructions"] = "Update pronouns accordingly"
        else:
            format_args["pronoun_instructions"] = "Keep pronouns unchanged"
            
    elif transform_type == "custom":
        format_args["custom_prompt"] = params.get("custom_prompt", "")
    
    try:
        formatted = template.format(**format_args)
    except KeyError:
        formatted = template
    
    return f"{TRANSFORM_SYSTEM_BASE}\n\n{formatted}"


# =============================================================================
# Response Schema Definitions
# =============================================================================


ANALYSIS_SCHEMAS = {
    "characters": {
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "role": {"type": "string", "enum": ["protagonist", "antagonist", "supporting", "mentioned"]},
                        "traits": {"type": "array", "items": {"type": "string"}},
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "character": {"type": "string"},
                                    "relationship": {"type": "string"}
                                }
                            }
                        },
                        "mentions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "line": {"type": "integer"},
                                    "context": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        "required": ["characters"]
    },
    "plot": {
        "type": "object",
        "properties": {
            "plot_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "summary": {"type": "string"},
                        "characters_involved": {"type": "array", "items": {"type": "string"}},
                        "significance": {"type": "string", "enum": ["low", "medium", "high"]},
                        "line_start": {"type": "integer"},
                        "line_end": {"type": "integer"}
                    },
                    "required": ["summary"]
                }
            },
            "tension_level": {"type": "string"},
            "setting": {"type": "string"}
        },
        "required": ["plot_points"]
    },
    "style": {
        "type": "object",
        "properties": {
            "style": {
                "type": "object",
                "properties": {
                    "voice": {
                        "type": "object",
                        "properties": {
                            "person": {"type": "string"},
                            "tense": {"type": "string"}
                        }
                    },
                    "tone": {"type": "array", "items": {"type": "string"}},
                    "sentence_complexity": {"type": "string"},
                    "dialogue_frequency": {"type": "string"},
                    "description_density": {"type": "string"},
                    "pacing": {"type": "string"},
                    "techniques": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "required": ["style"]
    },
    "summary": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"}
        },
        "required": ["summary"]
    },
    "themes": {
        "type": "object",
        "properties": {
            "themes": {"type": "array", "items": {"type": "string"}},
            "motifs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "meaning": {"type": "string"}
                    }
                }
            },
            "subtext": {"type": ["string", "null"]}
        },
        "required": ["themes"]
    }
}


TRANSFORM_SCHEMAS = {
    "style_rewrite": {
        "type": "object",
        "properties": {
            "transformed_content": {"type": "string"},
            "style_changes": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["transformed_content"]
    },
    "setting_change": {
        "type": "object",
        "properties": {
            "transformed_content": {"type": "string"},
            "replacements": {"type": "integer"},
            "adaptations": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["transformed_content"]
    },
    "tense_change": {
        "type": "object",
        "properties": {
            "transformed_content": {"type": "string"},
            "verbs_changed": {"type": "integer"}
        },
        "required": ["transformed_content"]
    },
    "pov_change": {
        "type": "object",
        "properties": {
            "transformed_content": {"type": "string"},
            "pov_adjustments": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["transformed_content"]
    },
    "character_rename": {
        "type": "object",
        "properties": {
            "transformed_content": {"type": "string"},
            "replacements": {"type": "integer"},
            "locations": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["transformed_content"]
    },
    "custom": {
        "type": "object",
        "properties": {
            "transformed_content": {"type": "string"},
            "changes_made": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["transformed_content"]
    }
}


__all__ = [
    "ANALYSIS_SYSTEM_BASE",
    "ANALYSIS_PROMPTS",
    "TRANSFORM_SYSTEM_BASE",
    "TRANSFORM_PROMPTS",
    "ANALYSIS_SCHEMAS",
    "TRANSFORM_SCHEMAS",
    "get_analysis_prompt",
    "get_transform_prompt",
]
