import os
from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


def make_openai_client(api_key):
    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


class DirectPromptAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        client = make_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()


class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        client = make_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}. Forget all previous context."
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()


class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, input_text):
        client = make_openai_client(self.openai_api_key)
        system_content = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}. "
            f"Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()


class RAGKnowledgePromptAgent:
    """Retrieval-Augmented Generation agent.

    Splits a knowledge text into overlapping chunks, computes embeddings for
    those chunks, and answers prompts by retrieving the most relevant chunk
    via cosine similarity before passing it to the LLM along with a persona.
    """

    def __init__(self, openai_api_key, persona, chunk_size=500, chunk_overlap=100):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unique_filename = f"chunks-{uuid.uuid4()}.csv"
        self.embeddings_filename = f"embeddings-{uuid.uuid4()}.csv"
        self.chunks = []
        self.embeddings_df = None

    def get_embedding(self, text):
        client = make_openai_client(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def chunk_text(self, text):
        separator_pattern = re.compile(r"[.!?]\s+")
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= self.chunk_size:
            chunks = [{"chunk_id": 0, "text": text, "chunk_size": len(text),
                       "start_char": 0, "end_char": len(text)}]
            self.chunks = chunks
            pd.DataFrame(chunks).to_csv(self.unique_filename, index=False,
                                        encoding="utf-8", quoting=csv.QUOTE_ALL)
            return chunks

        chunks = []
        start = 0
        chunk_id = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            if end < text_length:
                # Try to split at a sentence boundary
                window = text[start:end]
                matches = list(separator_pattern.finditer(window))
                if matches:
                    end = start + matches[-1].end()
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_size": len(chunk_text),
                    "start_char": start,
                    "end_char": end,
                })
                chunk_id += 1
            if end == text_length:
                break
            start = max(end - self.chunk_overlap, start + 1)

        self.chunks = chunks
        pd.DataFrame(chunks).to_csv(self.unique_filename, index=False,
                                    encoding="utf-8", quoting=csv.QUOTE_ALL)
        return chunks

    def calculate_embeddings(self):
        if not self.chunks:
            raise ValueError("No chunks available. Call chunk_text() first.")
        rows = []
        for chunk in self.chunks:
            embedding = self.get_embedding(chunk["text"])
            rows.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": embedding,
            })
        self.embeddings_df = pd.DataFrame(rows)
        self.embeddings_df.to_csv(self.embeddings_filename, index=False,
                                  encoding="utf-8", quoting=csv.QUOTE_ALL)
        return self.embeddings_df

    def find_prompt_in_knowledge(self, prompt):
        if self.embeddings_df is None or self.embeddings_df.empty:
            raise ValueError("Embeddings not calculated. Call calculate_embeddings() first.")
        prompt_embedding = np.array(self.get_embedding(prompt))
        best_score = -1.0
        best_chunk_text = ""
        for _, row in self.embeddings_df.iterrows():
            chunk_embedding = np.array(row["embedding"])
            similarity = np.dot(prompt_embedding, chunk_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(chunk_embedding)
            )
            if similarity > best_score:
                best_score = similarity
                best_chunk_text = row["text"]

        client = make_openai_client(self.openai_api_key)
        system_content = (
            f"You are {self.persona}, a knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer the prompt: {best_chunk_text}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


class EvaluationAgent:
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        client = make_openai_client(self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        final_response = None
        evaluation = ""

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i + 1} ---")
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            final_response = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{final_response}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {final_response}\n"
                f"Meet this criteria: {self.evaluation_criteria} "
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.persona},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {final_response}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )

        return {
            "final_response": final_response,
            "evaluation": evaluation,
            "iterations": i + 1
        }


class RoutingAgent:
    def __init__(self, openai_api_key, agents=None):
        self.openai_api_key = openai_api_key
        self.agents = agents or []

    def get_embedding(self, text):
        client = make_openai_client(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def route(self, user_input):
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1.0

        for agent in self.agents:
            agent_description = agent.get("description", "")
            if not agent_description:
                continue
            agent_emb = self.get_embedding(agent_description)
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        client = make_openai_client(self.openai_api_key)
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. "
            "You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        response_text = response.choices[0].message.content.strip()
        steps = []
        for line in response_text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r'^[0-9]+\.\s*', '', cleaned)
            cleaned = re.sub(r'^[-*]\s*', '', cleaned)
            if cleaned:
                steps.append(cleaned)
        return steps
