import os
import re
import sys
from dotenv import load_dotenv
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is required in environment variables.")

spec_path = os.path.join(os.path.dirname(__file__), "Product-Spec-Email-Router.txt")
with open(spec_path, "r", encoding="utf-8") as f:
    product_spec = f.read()

# ---------------------------------------------------------------------------
# Action planning knowledge: bias the planner toward an Email Router plan that
# explicitly produces user stories, then features, then engineering tasks.
# ---------------------------------------------------------------------------
knowledge_action_planning = (
    "A complete product plan for the Email Router is composed of exactly three "
    "high-level steps, produced in this order:\n"
    "Step 1: Generate the user stories for the Email Router product.\n"
    "Step 2: Generate the product features for the Email Router product, grouped "
    "from the user stories.\n"
    "Step 3: Generate the engineering tasks for the Email Router product, "
    "derived from the user stories.\n"
    "When asked to plan the Email Router product, return ONLY these three short "
    "high-level steps as a numbered list. Do NOT include the actual user stories, "
    "feature descriptions, or task details inside the plan. Do NOT add sub-steps. "
    "Each step must be a single short sentence describing what to produce."
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# ---------------------------------------------------------------------------
# Product Manager - user stories
# ---------------------------------------------------------------------------
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_product_manager,
    knowledge_product_manager,
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona="You are an evaluation agent that checks the answers of other worker agents",
    evaluation_criteria=(
        "The answer should be stories that follow the following structure: As a [type of user], "
        "I want [an action or feature] so that [benefit/value]."
    ),
    worker_agent=product_manager_knowledge_agent,
    max_interactions=3,
)

# ---------------------------------------------------------------------------
# Program Manager - product features
# ---------------------------------------------------------------------------
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups. "
    "Use the product spec and the story generation pattern to list product features. "
    f"{product_spec}"
)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_program_manager,
    knowledge_program_manager,
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona="You are an evaluation agent that checks the answers of other worker agents",
    evaluation_criteria=(
        "The answer should be product features that follow the following structure: "
        "Feature Name: A clear, concise title that identifies the capability\n"
        "Description: A brief explanation of what the feature does and its purpose\n"
        "Key Functionality: The specific capabilities or actions the feature provides\n"
        "User Benefit: How this feature creates value for the user"
    ),
    worker_agent=program_manager_knowledge_agent,
    max_interactions=3,
)

# ---------------------------------------------------------------------------
# Development Engineer - engineering tasks
# ---------------------------------------------------------------------------
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story. "
    "Use the product spec and generated user stories to produce implementation tasks. "
    f"{product_spec}"
)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_dev_engineer,
    knowledge_dev_engineer,
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona="You are an evaluation agent that checks the answers of other worker agents",
    evaluation_criteria=(
        "The answer should be tasks following this exact structure: "
        "Task ID: A unique identifier for tracking purposes\n"
        "Task Title: Brief description of the specific development work\n"
        "Related User Story: Reference to the parent user story\n"
        "Description: Detailed explanation of the technical work required\n"
        "Acceptance Criteria: Specific requirements that must be met for completion\n"
        "Estimated Effort: Time or complexity estimation\n"
        "Dependencies: Any tasks that must be completed first"
    ),
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=3,
)

routing_agent = RoutingAgent(openai_api_key, [])


def product_manager_support_function(query):
    evaluation = product_manager_evaluation_agent.evaluate(query)
    return evaluation.get("final_response", "")


def program_manager_support_function(query):
    evaluation = program_manager_evaluation_agent.evaluate(query)
    return evaluation.get("final_response", "")


def development_engineer_support_function(query):
    evaluation = development_engineer_evaluation_agent.evaluate(query)
    return evaluation.get("final_response", "")


# Route descriptions are written with vocabulary that maximally distinguishes
# them in embedding space. Each description names the deliverable shape ("As a
# ... user story", "Feature Name:", "Task ID:") so that semantic similarity
# selects the correct specialist for each step.
routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": (
            "Use the Product Manager route to write user stories for the Email Router product. "
            "The Product Manager only produces user stories in the agile format "
            "'As a [type of user], I want [an action or feature] so that [benefit/value].' "
            "Choose this route whenever the step asks to identify, list, define, draft, "
            "review, or generate user stories, personas, user needs, or user-centered "
            "requirements. Do NOT choose this route for product features or engineering tasks."
        ),
        "func": product_manager_support_function,
    },
    {
        "name": "Program Manager",
        "description": (
            "Use the Program Manager route to define product features for the Email Router. "
            "The Program Manager groups related user stories into product features written "
            "with the labels 'Feature Name:', 'Description:', 'Key Functionality:', and "
            "'User Benefit:'. Choose this route whenever the step asks to define, list, "
            "group, organize, or generate product features, capabilities, or feature sets. "
            "Do NOT choose this route for user stories or engineering tasks."
        ),
        "func": program_manager_support_function,
    },
    {
        "name": "Development Engineer",
        "description": (
            "Use the Development Engineer route to define engineering tasks for the Email Router. "
            "The Development Engineer writes implementation tasks for each user story using the "
            "labels 'Task ID:', 'Task Title:', 'Related User Story:', 'Description:', "
            "'Acceptance Criteria:', 'Estimated Effort:', and 'Dependencies:'. Choose this "
            "route whenever the step asks to define, break down, or generate engineering tasks, "
            "development tasks, implementation tasks, technical work items, or backlog tasks. "
            "Do NOT choose this route for user stories or product features."
        ),
        "func": development_engineer_support_function,
    },
]


# ---------------------------------------------------------------------------
# Lightweight deterministic tie-break: if the step text obviously names one of
# the deliverables, force the matching route. This keeps the routing layer
# robust when embedding similarities sit close together.
# ---------------------------------------------------------------------------
# Order matters: the most specific deliverable wins. Engineering tasks are
# checked first, then product features, then user stories. This prevents a
# step like "engineering tasks derived from the user stories" from being
# misrouted to the Product Manager just because the phrase "user stories"
# appears in the step text.
KEYWORD_ROUTES = [
    (
        "Development Engineer",
        re.compile(
            r"\b(engineering|development|implementation|technical|backlog|dev)\s+tasks?\b"
            r"|\btasks?\s+(for|from|derived)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Program Manager",
        re.compile(r"product[\s-]*features?|\bfeatures?\b", re.IGNORECASE),
    ),
    (
        "Product Manager",
        re.compile(r"user[\s-]*stor(y|ies)|\bpersonas?\b", re.IGNORECASE),
    ),
]


def route_step(step_text: str):
    """Route a workflow step using a keyword override falling back to embeddings."""
    for route_name, pattern in KEYWORD_ROUTES:
        if pattern.search(step_text):
            for agent in routing_agent.agents:
                if agent["name"] == route_name:
                    print(f"  -> Routed to: {route_name} (keyword match)")
                    return route_name, agent["func"](step_text)
    # Fallback to semantic routing
    result = routing_agent.route(step_text)
    return None, result


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


class _Tee:
    """Mirror stdout to a log file so the run can be re-captured easily."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


log_path = os.path.join(os.path.dirname(__file__), "agentic_workflow_output.txt")
log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, log_file)

print("\n*** Workflow execution started ***\n")

# Email Router-first workflow prompt: explicitly request all three deliverables.
workflow_prompt = (
    "Build a comprehensive product plan for the Email Router product. "
    "The plan must produce user stories, then product features, then engineering "
    "tasks. Return only the high-level steps required to deliver this plan."
)
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
raw_workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)


def _is_high_level_step(line: str) -> bool:
    """Drop introductory headers and inlined deliverable content."""
    lower = line.lower().strip()
    if not lower:
        return False
    # Discard introductory or trailing meta lines
    if lower.endswith(":") and len(lower) < 80:
        return False
    if lower.startswith(("here are", "the following", "below are")):
        return False
    # Discard inlined deliverables that occasionally appear as steps
    if lower.startswith("as a "):
        return False
    if lower.startswith(("feature name", "description:", "key functionality",
                        "user benefit", "task id", "task title",
                        "related user story", "acceptance criteria",
                        "estimated effort", "dependencies")):
        return False
    return True


workflow_steps = [s for s in raw_workflow_steps if _is_high_level_step(s)]
# Safety net: if the planner over-filtered, fall back to a fixed three-step plan.
if len(workflow_steps) < 3:
    workflow_steps = [
        "Generate the user stories for the Email Router product.",
        "Generate the product features for the Email Router product, grouped from the user stories.",
        "Generate the engineering tasks for the Email Router product, derived from the user stories.",
    ]

print("Workflow steps:")
for step in workflow_steps:
    print(f"- {step}")

# Track outputs by deliverable so we can produce a consolidated report at the end.
deliverables = {
    "Product Manager": [],
    "Program Manager": [],
    "Development Engineer": [],
}

completed_steps = []
for step in workflow_steps:
    print(f"\nProcessing step: {step}")
    forced_route, routed_result = route_step(step)
    completed_steps.append(routed_result)
    print(f"Result:\n{routed_result}\n")

    # Best-effort attribution of each step output to a deliverable bucket.
    if forced_route:
        deliverables[forced_route].append(routed_result)
    else:
        text_lower = (routed_result or "").lower()
        if "task id" in text_lower:
            deliverables["Development Engineer"].append(routed_result)
        elif "feature name" in text_lower:
            deliverables["Program Manager"].append(routed_result)
        elif "as a " in text_lower:
            deliverables["Product Manager"].append(routed_result)

print("\n*** Workflow completed ***")

# ---------------------------------------------------------------------------
# Final consolidated Email Router deliverable
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("FINAL CONSOLIDATED EMAIL ROUTER PLAN")
print("=" * 78)


def _print_section(title: str, items):
    print(f"\n--- {title} ---\n")
    if not items:
        print("(no content produced for this section)")
        return
    for i, content in enumerate(items, 1):
        if len(items) > 1:
            print(f"[{title} output #{i}]")
        print(content.strip())
        print()


_print_section("USER STORIES (Product Manager)", deliverables["Product Manager"])
_print_section("PRODUCT FEATURES (Program Manager)", deliverables["Program Manager"])
_print_section("ENGINEERING TASKS (Development Engineer)", deliverables["Development Engineer"])

print("=" * 78)
print("End of consolidated Email Router plan")
print("=" * 78)

if not completed_steps:
    print("No workflow steps were completed.")

# Restore stdout and close log file
sys.stdout = sys.__stdout__
log_file.close()
print(f"Run log written to: {log_path}")
