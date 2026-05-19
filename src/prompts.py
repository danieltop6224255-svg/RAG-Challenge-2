from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt

class RephrasedQuestionsPrompt:
    instruction = """
You are a question rephrasing system.
Your task is to break down a comparative question into individual questions for each company mentioned.
Each output question must be self-contained, maintain the same intent and metric as the original question, be specific to the respective company, and use consistent phrasing.
"""

    class RephrasedQuestion(BaseModel):
        """Individual question for a company"""
        company_name: str = Field(description="Company name, exactly as provided in quotes in the original question")
        question: str = Field(description="Rephrased question specific to this company")

    class RephrasedQuestions(BaseModel):
        """List of rephrased questions"""
        questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each company")

    pydantic_schema = '''
class RephrasedQuestion(BaseModel):
    """Individual question for a company"""
    company_name: str = Field(description="Company name, exactly as provided in quotes in the original question")
    question: str = Field(description="Rephrased question specific to this company")

class RephrasedQuestions(BaseModel):
    """List of rephrased questions"""
    questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each company")
'''

    example = r"""
Example:
Input:
Original comparative question: 'Which company had higher revenue in 2022, "Apple" or "Microsoft"?'
Companies mentioned: "Apple", "Microsoft"

Output:
{
    "questions": [
        {
            "company_name": "Apple",
            "question": "What was Apple's revenue in 2022?"
        },
        {
            "company_name": "Microsoft", 
            "question": "What was Microsoft's revenue in 2022?"
        }
    ]
}
"""

    user_prompt = "Original comparative question: '{question}'\n\nCompanies mentioned: {companies}"

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)




class SubQuestionsPrompt:
    instruction = """
You analyze user questions for decomposition.
Decide whether a question should be split into multiple independent sub-questions.
Only split when answering accurately requires separate retrieval/answering steps.
"""

    class SubQuestionsSchema(BaseModel):
        is_multi_question: bool = Field(description="Whether the question should be decomposed into multiple sub-questions")
        sub_questions: List[str] = Field(description="List of standalone sub-questions. Empty if no split is needed")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(SubQuestionsSchema), flags=re.MULTILINE)

    example = r"""
Example 1:
Question: "What is the revenue of company A in 2023?"
Output:
{
  "is_multi_question": false,
  "sub_questions": []
}

Example 2:
Question: "Compare total assets of Apple and Microsoft in 2022 and say who is higher"
Output:
{
  "is_multi_question": true,
  "sub_questions": [
    "What were Apple's total assets in 2022?",
    "What were Microsoft's total assets in 2022?"
  ]
}
"""

    user_prompt = 'Question: "{question}"'
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextPrompt:
    instruction = """
You are a RAG (Retrieval-Augmented Generation) answering system.
Your task is to answer the given question based only on the provided context pages.

Before giving a final answer, think step by step and rely only on explicit evidence from context.
If the answer is missing or ambiguous, return 'N/A'.
"""

    user_prompt = """
Here is the context:
\"\"\"
{context}
\"\"\"

---

Here is the question:
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis grounded in provided context.")
        reasoning_summary: str = Field(description="Concise summary of the reasoning process.")
        relevant_pages: List[int] = Field(description="List of context page numbers used for the answer.")
        final_answer: Union[str, Literal['N/A']] = Field(
            description="Final answer extracted from context. Return 'N/A' if unavailable."
        )

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)
    system_prompt = build_system_prompt(instruction)
    system_prompt_with_schema = build_system_prompt(instruction, pydantic_schema=pydantic_schema)


class AnswerWithRAGContextBooleanPrompt:
    instruction = AnswerWithRAGContextPrompt.instruction
    user_prompt = AnswerWithRAGContextPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested value, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")
        
        final_answer: Union[bool] = Field(description="""
A boolean value (True or False) extracted from the context that precisely answers the question.
If question ask about did something happen, and in context there is information about it, return False.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Question:
"Did W. P. Carey Inc. announce any changes to its dividend policy in the annual report?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks whether W. P. Carey Inc. announced changes to its dividend policy.\n2. The phrase 'changes to its dividend policy' requires careful interpretation. It means any adjustment to the framework, rules, or stated intentions that dictate how the company determines and distributes dividends.\n3. The context (page 12, 18) states that the company increased its annualized dividend to $4.27 per share in the fourth quarter of 2023, compared to $4.22 per share in the same period of 2022. Page 45 mentions further details about dividend.\n4. Consistent, incremental increases throughout the year, with explicit mentions of maintaining a 'steady and growing' dividend, indicates no changes to *policy*, though the *amount* increased as planned within the existing policy.",
  "reasoning_summary": "The context highlights consistent, small increases to the dividend throughout the year, consistent with a stated policy of providing a 'steady and growing' dividend. While the dividend *amount* changed, the *policy* governing those increases remained consistent. The question asks about *policy* changes, not amount changes.",
  "relevant_pages": [12, 18, 45],
  "final_answer": False
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNamesPrompt:
    instruction = AnswerWithRAGContextPrompt.instruction
    user_prompt = AnswerWithRAGContextPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested entity, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
Each entry should be extracted exactly as it appears in the context.

If the question asks about positions (e.g., changes in positions), return ONLY position titles, WITHOUT names or any additional information. Appointments on new leadership positions also should be counted as changes in positions. If several changes related to position with same title are mentioned, return title of such position only once. Position title always should be in singular form.
Example of answer ['Chief Technology Officer', 'Board Member', 'Chief Executive Officer']

If the question asks about names, return ONLY the full names exactly as they are in the context.
Example of answer ['Carly Kennedy', 'Brian Appelgate Jr.']

If the question asks about new launched products, return ONLY the product names exactly as they are in the context. Candidates for new products or products in testing phase not counted as new launched products.
Example of answer ['EcoSmart 2000', 'GreenTech Pro']

- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"What are the names of all new executives that took on new leadership positions in company?"

Answer:
```
{
    "step_by_step_analysis": "1. The question asks for the names of all new executives who took on new leadership positions in the company.\n2. Exhibit 10.9 and 10.10, as listed in the Exhibit Index on page 89, mentions new Executive Agreements with Carly Kennedy and Brian Appelgate.\n3. Exhibit 10.9, Employment Agreement with Carly Kennedy, states her start date as April 4, 2022, and her position as Executive Vice President and General Counsel.\n4. Exhibit 10.10, Offer Letter with Brian Appelgate shows that his new role within the company is Interim Chief Operations Officer, and he was accepting the offer on November 8, 2022.\n5. Based on the documents, Carly Kennedy and Brian Appelgate are named as the new executives.",
    "reasoning_summary": "Exhibits 10.9 and 10.10 of the annual report, described as Employment Agreement and Offer Letter, explicitly name Carly Kennedy and Brian Appelgate taking on new leadership roles within the company in 2022.",
    "relevant_pages": [
        89
    ],
    "final_answer": [
        "Carly Kennedy",
        "Brian Appelgate"
    ]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class ComparativeAnswerPrompt:
    instruction = """
You are a question answering system.
Your task is to analyze individual company answers and provide a comparative response that answers the original question.
Base your analysis only on the provided individual answers - do not make assumptions or include external knowledge.
Before giving a final answer, carefully think out loud and step by step.

Important rules for comparison:
- When the question asks to choose one of the companies (e.g., when comparing metrics), return the company name exactly as it appears in the original question
- If a company's metric is in a different currency than what is asked in the question, exclude that company from comparison
- If all companies are excluded (due to currency mismatch or other reasons), return 'N/A' as the final answer
- If all companies except one are excluded, return the name of the remaining company (even though there is no actual comparison possible)
"""

    user_prompt = """
Here are the individual company answers:
\"\"\"
{context}
\"\"\"

---

Here is the original comparative question:
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="Just leave empty")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
Company name should be extracted exactly as it appears in question.
Answer should be either a single company name or 'N/A' if no company is applicable.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"Which of the companies had the lowest total assets in USD at the end of the period listed in the annual report: "CrossFirst Bank", "Sleep Country Canada Holdings Inc.", "Holley Inc.", "PowerFleet, Inc.", "Petra Diamonds"? If data for the company is not available, exclude it from the comparison."

Answer:
```
{
  "step_by_step_analysis": "1. The question asks for the company with the lowest total assets in USD.\n2. Gather the total assets in USD for each company from the individual answers: CrossFirst Bank: $6,601,086,000; Holley Inc.: $1,249,642,000; PowerFleet, Inc.: $217,435,000; Petra Diamonds: $1,078,600,000.\n3. Sleep Country Canada Holdings Inc. is excluded because its assets are not reported in USD.\n4. Compare the total assets: PowerFleet, Inc. ($217,435,000) < Petra Diamonds ($1,078,600,000) < Holley Inc. ($1,249,642,000)  < CrossFirst Bank ($6,601,086,000).\n5. Therefore, PowerFleet, Inc. has the lowest total assets in USD.",
  "reasoning_summary": "The individual answers provided the total assets in USD for each company except Sleep Country Canada Holdings Inc. (excluded due to currency mismatch). Direct comparison shows PowerFleet, Inc. has the lowest total assets.",
  "relevant_pages": [],
  "final_answer": "PowerFleet, Inc."
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    system_prompt = """
You are a JSON formatter.
Your task is to format raw LLM response into a valid JSON object.
Your answer should always start with '{' and end with '}'
Your answer should contain only json string, without any preambles, comments, or triple backticks.
"""

    user_prompt = """
Here is the system prompt that defines schema of the json object and provides an example of answer with valid schema:
\"\"\"
{system_prompt}
\"\"\"

---

Here is the LLM response that not following the schema and needs to be properly formatted:
\"\"\"
{response}
\"\"\"
"""




class RerankingPrompt:
    system_prompt_rerank_single_block = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and retrieved text block related to that query. Your task is to evaluate and score the block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate block based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

    system_prompt_rerank_multiple_blocks = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and several retrieved text blocks related to that query. Your task is to evaluate and score each block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate blocks based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

class RetrievalRankingSingleBlock(BaseModel):
    """Rank retrieved text block relevance to a query."""
    reasoning: str = Field(description="Analysis of the block, identifying key information and how it relates to the query")
    relevance_score: float = Field(description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant")

class RetrievalRankingMultipleBlocks(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="A list of text blocks and their associated relevance scores."
    )
