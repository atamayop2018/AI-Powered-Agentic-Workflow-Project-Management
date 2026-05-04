from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the environment.")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_agent.respond(prompt)

print(augmented_agent_response)
# The agent likely used general factual knowledge from the LLM model.
# The persona prompt shaped the response style, encouraging a polite, professor-like answer that begins with "Dear students,".
