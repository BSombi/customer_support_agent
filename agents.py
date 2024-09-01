
from crewai import Agent

class customer_support_agents():

    def support_agent(self, customer):
            return Agent(
                  role="Senior Support Representative",
                  goal="Be the most friendly and helpful "
                        "support representative in your team",
                    backstory=(
                        "You work at crewAI (https://crewai.com) and "
                        " are now working on providing "
                        "support to {customer}, a super important customer "
                        " for your company."
                        "You need to make sure that you provide the best support!"
                        "Make sure to provide full complete answers, "
                        " and make no assumptions."
                    ),
        allow_delegation=False,
        verbose=True
    )

    def support_quality_assurance_agent(self, customer):
        return Agent(
             role="Support Quality Assurance Specialist",
             goal="Get recognition for providing the "
             "best support quality assurance in your team",
             backstory=(
                    "You work at crewAI (https://crewai.com) and "
                    "are now working with your team "
                    "on a request from {customer} ensuring that "
                    "the support representative is "
                    "providing the best support possible.\n"
                    "You need to make sure that the support representative "
                    "is providing full"
                    "complete answers, and make no assumptions."
                ),
                verbose=True
            )