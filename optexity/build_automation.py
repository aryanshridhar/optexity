"""Compile a cached agentic run into a deterministic automation """

import logging

from pydantic import ValidationError

from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.automation import Automation

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You compile recordings of browser agents into deterministic automations. "
    "You reply with a single lean JSON object and nothing else: only url, "
    "parameters, and nodes, with no defaulted or null fields."
)

# The mapping is not left to the model's judgement: every recordable browser-use action
# has exactly one InteractionAction field, and `RECORDABLE` in
# browser_use/action_cache/action_cache.py is the same table seen from the other side.
ACTION_MAPPING = """\
| trace action    | interaction_action field | payload |
|-----------------|--------------------------|---------|
| click           | click_element      | command, prompt_instructions |
| input           | input_text         | command, prompt_instructions, input_text (the value that was typed) |
| select_dropdown | select_option      | command, prompt_instructions, select_values (list with the chosen option text) |
| upload_file     | upload_file        | file_path |
| navigate        | go_to_url          | url, new_tab |
| go_back         | go_back            | {} |
| send_keys       | key_press          | command, prompt_instructions, type (a LOWERCASE key name such as "enter", "tab", "escape", or a list for a combination such as ["ctrl", "a"]) |
| switch          | switch_tab         | tab_index |
| close           | close_current_tab  | {} |
| scroll          | scroll             | down (bool), amount (-1 for a full page) |\
"""

# Lean excerpt of optexity/examples/login_cookies.json — command + prompt_instructions
# + payload only. Placeholders appear only because this original declared them.
EXAMPLE_AUTOMATION = """\
{
    "url": "https://dev.dashboard.optexity.com/login",
    "parameters": {
        "input_parameters": {"email": ["test@gmail.com"], "password": ["12345678"]},
        "generated_parameters": {}
    },
    "nodes": [
        {
            "type": "action_node",
            "interaction_action": {
                "input_text": {
                    "command": "get_by_role(\\"textbox\\", name=\\"Email\\")",
                    "prompt_instructions": "Enter the email address {email[0]} into the Email field.",
                    "input_text": "{email[0]}"
                }
            }
        },
        {
            "type": "action_node",
            "interaction_action": {
                "input_text": {
                    "command": "get_by_role(\\"textbox\\", name=\\"Password\\")",
                    "prompt_instructions": "Enter the password into the Password field.",
                    "input_text": "{password[0]}"
                }
            }
        },
        {
            "type": "action_node",
            "interaction_action": {
                "click_element": {
                    "command": "get_by_role(\\"button\\", name=\\"Sign In\\", exact=True)",
                    "prompt_instructions": "Click the Sign In button."
                }
            }
        }
    ]
}\
"""

PROMPT_TEMPLATE = """\
Compile the recording below into a deterministic Optexity automation.

## ORIGINAL AUTOMATION

Keep `url` and `parameters` byte-for-byte. Do not add, rename, or infer parameters.
Replace every `agentic_task` node with the deterministic action nodes derived from the
trace. Do not keep the agentic_task.

{original}

## TRACE

One JSON object per action the agent actually performed, in order. Fields:

- `action`     the browser action name
- `params`     its arguments (`text` is the typed value for input)
- `intent`     the agent's next-goal thought; often about a *later* field — do not
               copy it into prompt_instructions
- `url`        the page URL when the action ran
- `element`    the target's attributes (use `name` / `ax_role` to describe it)
- `verified`   Playwright locators PROVEN against the live page, best first.
               Copy `verified[0]` verbatim as `command`.
- `unverified` use one only when `verified` is empty

{trace}

## OUTPUT

A single JSON object. No prose, no markdown fences, no extra keys.

Top-level object is exactly: `url`, `parameters`, `nodes`.

### Nodes

Every kept step is only:

{{"type": "action_node", "interaction_action": {{"<field>": {{...}}}}}}

Exactly one `<field>`, from this table. The payload contains ONLY the columns listed —
nothing else (no xpath, no skip_*, no fill_or_type, no click_before_input, no
press_enter, no is_slider, no max_tries, no timeouts, no sleeps, no browser_channel).

{mapping}

Do not add clicks, sleeps, extractions, assertions, or a leftover agentic_task.

### command

A Playwright expression with any leading `page.` stripped, e.g.
`get_by_role("button", name="Sign In")` or `locator("input[name='email']")`.
Copy `verified[0]` as-is. Do not invent a locator. Do not append `.first` unless it is
already on `verified[0]`. Never set both `command` and `xpath`.

### prompt_instructions

Required on every element action. One short human sentence naming THIS element's
field, from `element.attributes` / `ax_role` — not from `intent`.
Examples: "Enter the email into the Email field." / "Click the Sign In button."

### Pruning

Drop exploratory scrolls, orientation navigations, and actions that were immediately
undone (click then go_back). Keep every fill, click, select, or keypress that produced
the end state, in recorded order. One node per kept trace row.

### Parameters

Copy `parameters` from the original unchanged.

- If the original already declares an `input_parameters` key whose recorded value
  matches what was typed, write `"{{name[0]}}"` in `input_text` (and in
  prompt_instructions if you mention the value).
- If `input_parameters` is empty, or a typed value is not a declared parameter, put
  the recorded literal in `input_text`. Do not invent parameter names.

## EXAMPLE OUTPUT

This login workflow declared `email` and `password` in input_parameters, so those
fields use placeholders. An original with empty input_parameters would use literals
instead.

{example}
"""


def build_prompt(original: str, trace: str) -> str:
    return PROMPT_TEMPLATE.format(
        original=original,
        trace=trace,
        mapping=ACTION_MAPPING,
        example=EXAMPLE_AUTOMATION,
    )


def _json_payload(completion: str) -> str:
    """Strip a markdown fence if the model added one despite being told not to."""
    text = completion.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json") :]
    return text.strip()


def compile_automation(
    original: str, trace: str, model_name: str | None = None, max_attempts: int = 3
) -> Automation:
    """One LLM call, then validate. On a validation error, hand the error back and retry.

    The model is doing two jobs that are awkward in Python but easy for it: deciding
    which recorded steps mattered, and writing the natural-language fallback for each
    step. The locators themselves it only copies - those were already proven in-page.
    """
    model = get_llm_model_with_fallback(None, model_name, use_structured_output=False)
    prompt = build_prompt(original, trace)
    last_error: ValidationError | None = None

    for attempt in range(1, max_attempts + 1):
        completion, usage = model.get_model_response(prompt, SYSTEM_INSTRUCTION)
        logger.info(f"Attempt {attempt}: {usage.calculated_total_tokens} tokens")
        try:
            return Automation.model_validate_json(_json_payload(completion))
        except ValidationError as e:
            last_error = e
            logger.warning(f"Attempt {attempt} did not validate: {e}")
            prompt = (
                f"{build_prompt(original, trace)}\n\n"
                f"## PREVIOUS ATTEMPT REJECTED\n\n"
                f"You returned:\n\n{_json_payload(completion)}\n\n"
                f"It failed schema validation:\n\n{e}\n\n"
                f"Return the corrected lean JSON object. Copy the original "
                f"`parameters` unchanged; do not invent keys."
            )

    raise SystemExit(
        f"Could not produce a valid automation in {max_attempts} attempts: {last_error}"
    )
