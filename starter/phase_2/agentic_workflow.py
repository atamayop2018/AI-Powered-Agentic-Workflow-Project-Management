import os
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

knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

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

routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. Does not define features or tasks.",
        "func": product_manager_support_function,
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining product features from grouped user stories. Does not define engineering tasks.",
        "func": program_manager_support_function,
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining engineering tasks for user stories. Does not define product strategy or feature scope.",
        "func": development_engineer_support_function,
    },
]

print("\n*** Workflow execution started ***\n")
workflow_prompt = "What would the development tasks for this product be?"
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print("Workflow steps:")
for step in workflow_steps:
    print(f"- {step}")

completed_steps = []
for step in workflow_steps:
    print(f"\nProcessing step: {step}")
    routed_result = routing_agent.route(step)
    completed_steps.append(routed_result)
    print(f"Result:\n{routed_result}\n")

print("\n*** Workflow completed ***")
if completed_steps:
    print("Final structured output from the workflow:")
    print(completed_steps[-1])
else:
    print("No workflow steps were completed.")
