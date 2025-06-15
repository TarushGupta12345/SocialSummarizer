import os
from openai import OpenAI
import composio
import requests
from dotenv import load_dotenv
import datetime
from composio import ComposioToolSet, App
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI
from composio_langchain import ComposioToolSet, Action, App


load_dotenv()

openai_client = OpenAI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  

def get_cultural_stories() -> str:

    with open("cultural_prompt.txt", "r") as file:
        cultural_prompt = file.read()

    search_prompt = (
        f"Today's date is {datetime.now().strftime("%Y-%m-%d")}. {cultural_prompt}"
    )

    payload = {
            "model": "openai/gpt-4o-mini-search-preview",  
            "messages": [{"role": "user", "content": search_prompt}],
        }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",  
        "X-Title": "Social Summarizer",
        "Content-Type": "application/json",
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload
    )

    return response.json()["choices"][0]["message"]["content"]

def send_with_composio(input_stuff: str):
    llm = ChatOpenAI()
    prompt = hub.pull("hwchase17/openai-functions-agent")
    composio_toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))  
    
    integration = composio_toolset.get_integration(id="fa1cc231-219f-4b04-974f-b42fea845a3b")
    print(integration.expectedInputFields)
    tools = composio_toolset.get_tools(actions=['GMAIL_SEND_EMAIL'])

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    task = f"Send the following email to tarushgs@gmail.com: {input_stuff}"
    result = agent_executor.invoke({"input": task})
    print(result)

def main():
    send_with_composio(get_cultural_stories())

if __name__ == "__main__":
    main()