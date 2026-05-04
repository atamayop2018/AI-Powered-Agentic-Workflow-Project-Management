from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the environment.")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

print(direct_agent_response)
print("The agent used the general knowledge embedded in the GPT-3.5-Turbo model to answer this factual prompt.")
