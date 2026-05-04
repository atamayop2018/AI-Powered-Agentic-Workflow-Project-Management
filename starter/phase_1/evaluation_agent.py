from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the environment.")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

persona_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_eval,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=10,
)

result = evaluation_agent.evaluate(prompt)
print(result)
