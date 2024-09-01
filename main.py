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

    print("## Welcome to the Customer Support Agent Demo! ##")
    print('-----------------------------------')
    customer = input("What is the company name?\n")
    person = input("What is the customer name?\n")
    inquiry = input("What is the inquiry about?\n")
    
    tasks = customer_support_tasks()
    agents = customer_support_agents()

    #create agents
    support_agent = agents.support_agent(customer)
    support_quality_assurance_agent = agents.support_quality_assurance_agent(customer)

    # create tasks
    inquiry_resolution = tasks.inquiry_resolution(customer, inquiry, person)
    quality_assurance_review = tasks.quality_assurance_review()

    inquiry_resolution.context = [quality_assurance_review]

    crew = Crew(
        agents=[support_agent, support_quality_assurance_agent],
        tasks=[inquiry_resolution, quality_assurance_review],
        verbose=True,
        memory=True
    )

    result = crew.kickoff(inputs=[{"customer": customer}, 
                                  {"person": person},
                                  {"inquiry": inquiry}])

    print(result)

if __name__ == "__main__":
    main()