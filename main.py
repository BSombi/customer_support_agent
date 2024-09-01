from dotenv import load_dotenv
from crewai import Crew
from tasks import customer_support_tasks
from agents import customer_support_agents
from textwrap import dedent

import os
from utils import get_openai_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo-0125'
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

def main():
    load_dotenv()

    print("## Welcome to the Article Writing Agent Demo! ##")
    print('-----------------------------------')
    topic = input("What topic do you want to write about?\n")

    tasks = customer_support_tasks()
    agents = customer_support_agents()

    #create agents
    support_agent = agents.support_agent(customer)
    support_quality_assurance_agent = agents.support_quality_assurance_agent(customer)

    # create tasks
    plan = tasks.plan(topic, planner)
    check = tasks.check(topic, checker)
    write = tasks.write(topic, writer)
    edit = tasks.edit(editor)

    edit.context = [write]

    crew = Crew(
        agents=[planner, checker, writer, editor],
        tasks=[plan, check, write, edit],
        verbose=True
    )

    result = crew.kickoff(inputs={"topic": topic})

    print(result)

if __name__ == "__main__":
    main()