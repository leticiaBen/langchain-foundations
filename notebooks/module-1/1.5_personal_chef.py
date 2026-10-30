import base64
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

load_dotenv()

model = init_chat_model(model="gpt-5-mini")

# Read the image
with open("/Users/mac/langchain-ai/lca-lc-foundations/notebooks/module-1/resources/images.png", "rb") as f:
    img_bytes = f.read()

img_b64 = base64.b64encode(img_bytes).decode("utf-8")

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "What Can I cook with this?"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

tavily_client = TavilyClient()

@tool("Meal_search", description="From the list of ingredients, search the web for a meal that contains these ingredients")
def meal_search(query: str) -> Dict[str, Any]:
    return tavily_client.search(query)

tools = [meal_search]

config = {"configurable": {"thread_id": "3"}}

system_prompt = """You are a personal chef who helps the user cook with what they already have.

Follow these steps in order:

1. Identify the ingredients.
   - If the user sends a picture, list the ingredients you can see in it.
   - If the user sends a text list, use those ingredients directly.

2. Confirm before searching.
   - If the ingredients come from a picture, show the list and ask the user to confirm
     which ones they want to use. Do not search yet. Wait for their answer.
   - If the user wrote the ingredients themselves, skip this step.

3. Search for recipes.
   - Only after the ingredients are confirmed, use the Meal_search tool.
   - Your search must contain the confirmed ingredients, for example:
     "recipes with eggs, tomatoes, bell peppers and cheese".

4. Answer with 2 recipes. For each recipe, give:
   - the name of the dish
   - the ingredients needed, marking any the user does not have
   - the main steps, in a short numbered list
   - the cooking time

If you cannot see any ingredients in the picture, say so and ask the user to list them."""

# No checkpointer here: LangGraph Studio / `langgraph dev` provides persistence itself
chef = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

if __name__ == "__main__":
    # Local test run (python 1.5_personal_chef.py) needs its own memory between turns
    local_chef = create_agent(
        model=model,
        checkpointer=InMemorySaver(),
        tools=tools,
        system_prompt=system_prompt
    )

    response = local_chef.invoke(
        {"messages": [multimodal_question]},
        config,
    )

    print(response['messages'][-1].content)

    confirmation = HumanMessage(content="Yes, I want to use eggs, milk, salmon.")
    response = local_chef.invoke(
        {"messages": [confirmation]},
        config,
    )

    for msg in response["messages"]:
        msg.pretty_print()
