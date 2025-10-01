import os
from openai import OpenAI
from pydantic_ai import Agent

# Setup OpenAI client
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def answer_question_manual(question, search_method, index, vindex):
    # (your hybrid / vector / text search selection + prompt building)
    ...
    response = openai_client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "system", "content": "You are a helpful assistant..."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content

def create_agent(system_prompt, tools):
    return Agent(
        name="faq_agent",
        instructions=system_prompt,
        tools=tools,
        model="gpt-4o-mini"
    )
