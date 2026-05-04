from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the environment.")

texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    "You are a college professor",
    "You know everything about Texas"
)
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    "You are a college professor",
    "You know everything about Europe"
)
math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    "You are a college math professor",
    "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
)

routing_agent = RoutingAgent(openai_api_key, [])
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x),
    }
]

routing_agent.agents = agents

print("Prompt: Tell me about the history of Rome, Texas")
print(routing_agent.route("Tell me about the history of Rome, Texas"))
print()
print("Prompt: Tell me about the history of Rome, Italy")
print(routing_agent.route("Tell me about the history of Rome, Italy"))
print()
print("Prompt: One story takes 2 days, and there are 20 stories")
print(routing_agent.route("One story takes 2 days, and there are 20 stories"))
