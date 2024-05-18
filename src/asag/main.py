from dotenv import load_dotenv
import os
import dspy
import typer

from .grading_model import ASAGCoT

# Load environment variables from .env file
load_dotenv()

# Retrieve the OPENAI_API_KEY from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model with the API key
lm_gpt_35_turbo = dspy.OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", max_tokens=700)
dspy.settings.configure(lm=lm_gpt_35_turbo)

app = typer.Typer()

@app.command()
def assess(student_answer: str, reference_answer: str, question: str):
    """
    Assess the student's answer by comparing it to the reference answer.
    """
    pipeline = ASAGCoT()
    pipeline.load("notebooks/asag_cot.json")

    output = pipeline(student_answer=student_answer, reference_answer=reference_answer, question=question)

    print(output.assessment)

if __name__ == "__main__":
    app()
