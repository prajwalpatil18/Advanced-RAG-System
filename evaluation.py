from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import Literal, List

import os
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant",temperature=0.0)


class context_ret_eval(BaseModel):
    context_relevance_score : float
    explanation : str

con_reteriver_eval = model.with_structured_output(context_ret_eval)

def context_relevator_eval(question, context):
    prompt = f"""
    You are a strict evaluator.

    User Question:
    {question}

    Retrieved Context:
    {context}

    Task:
    Evaluate how relevant the retrieved context is for answering the user question.

    Scoring Guidelines:
    - (0.7 to 1.0) = Context is highly relevant and directly answers the question
    - (0.3 to 0.5) = Context is partially relevant but missing important information
    - (0.0 to 0.3) = Context is irrelevant or unrelated

    Return ONLY valid JSON in the following format:
    {{
    "context_relevance_score": float,
    "explanation": string
    }}
    """

    response = con_reteriver_eval.invoke(prompt)
    return response



class context_suff_eval(BaseModel):
    sufficiency : Literal["Yes", "No", "Partial"]
    missing_information: str

con_sufficient_eval = model.with_structured_output(context_suff_eval)


def context_sufficient_eval(question, context):
    prompt = f"""
    You are an expert evaluator.

    User Question:
    {question}

    Retrieved Context:
    {context}

    Task:
    Decide whether the retrieved context contains enough information to fully answer the user question.

    Return ONLY valid JSON:
    {{
    "sufficiency": "Yes" | "Partial" | "No",
    "missing_information": "brief description if not Yes in string format"
    }}

    """

    response = con_sufficient_eval.invoke(prompt)
    return response



class faith_eval(BaseModel):
    faithful: Literal[True, False]
    unsupported_claims: List[str]

faithfulness_evaluation = model.with_structured_output(faith_eval)


def faithfulness_eval(answer, context):
    prompt = f"""
    You are a strict fact checker.

    Retrieved Context:
    {context}

    Model Answer:
    {answer}

    Task:
    Identify whether the answer contains any statements not supported by the retrieved context.

    Instructions:
    - Break the answer into individual claims.
    - Check each claim against the context.
    - If a claim is not supported, list it.

    Return ONLY valid JSON:
    {{
    "faithful": true | false,
    "unsupported_claims": ["claim1 in string format", "claim2 in string format"]
    }}
    """

    response = faithfulness_evaluation.invoke(prompt)
    return response



class answer_relevance_eval(BaseModel):
    answer_relevance_score: float
    explanation : str

ans_relevance_evaluation = model.with_structured_output(answer_relevance_eval)


def answer_relevance_evaluation(question, answer):
    prompt = f"""
    You are an evaluator.

    User Question:
    {question}

    Model Answer:
    {answer}

    Task:
    Evaluate how well the answer addresses the user's question.

    Return ONLY valid JSON:
    {{
    "answer_relevance_score": float,
    "explanation": string
    }}
    """

    response = ans_relevance_evaluation.invoke(prompt)
    return response



class ans_completeness_eval(BaseModel):
    completeness: Literal["Complete", "Partial", "Incomplete"]
    missing_parts: str
    

ans_completeness_evaluation = model.with_structured_output(ans_completeness_eval)


def answer_completeness_eval(question, answer):
    prompt = f"""
    You are a completeness evaluator.

    User Question:
    {question}

    Model Answer:
    {answer}

    Task:
    Determine whether the answer fully covers all aspects of the question.

    Return ONLY valid JSON:
    {{
    "completeness": "Complete" | "Partial" | "Incomplete",
    "missing_parts": "brief description if not Complete in string format"
    }}
    """

    response = ans_completeness_evaluation.invoke(prompt)
    return response


