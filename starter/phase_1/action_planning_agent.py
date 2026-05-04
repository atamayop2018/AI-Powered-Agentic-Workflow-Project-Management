from workflow_agents.base_agents import ActionPlanningAgent
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the environment.")

knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product described in the specification. "
    "Features are defined by grouping related user stories. "
    "Tasks are defined for each story and represent the engineering work required to develop the product."
)

action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)
response_steps = action_planning_agent.extract_steps_from_prompt("One morning I wanted to have scrambled eggs")
print("Extracted action steps:")
for step in response_steps:
    print(f"- {step}")
