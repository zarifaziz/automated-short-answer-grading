import dspy

class AssessmentSignature(dspy.Signature):
    """
    Role: Academic Tutor
    Task: Make an assessment of the student's answer by comparing it with the reference answer.
    You are essentially assessing whether the student's answer was correct, or needs additional feedback.
    
    The value for assessment must be EXACTLY one of the following classes:
    "correct", "partially_correct_incomplete", "contradictory", "non_domain", "irrelevant"
    Include the underscores if the class contains it.

    Definitions of the classes:
    - correct: if the student answer is a complete and correct paraphrase of the reference answer
    - partially_correct_incomplete: if it is a partially correct answer containing some but not
    all information from the reference answer
    - contradictory: if the student answer explicitly contradicts the reference answer
    - irrelevant:  if the student answer is talking about domain content but not providing the necessary information
    - non_domain: if the student utterance does not include domain content, e.g., “I don’t know”, “what the book says”, “you are stupid”

    The "question" field doesn't provide too much context apart from the final few words of the problem.
    So don't focus on the question too much.
    """

    question: str = dspy.InputField()
    student_answer: str = dspy.InputField()
    reference_answer: str = dspy.InputField()

    assessment: str = dspy.OutputField(desc="the final assessment")

class ASAGCoT(dspy.Module):
    """Assess the student's answer to the question by comparing with the reference answer"""

    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(AssessmentSignature)

    def forward(self, question: str, student_answer: str, reference_answer: str):
        output = self.prog(question=question, student_answer=student_answer, reference_answer=reference_answer)
        
        output.assessment = str(output.assessment).lower()
        dspy.Suggest(
            output.assessment in ["correct", "partially_correct_incomplete", "contradictory", "non_domain", "irrelevant"],
            f"'{output.assessment}' must be exactly one of 'correct', 'partially_correct_incomplete', 'contradictory', 'non_domain', 'irrelevant'"
        )
        
        return output
