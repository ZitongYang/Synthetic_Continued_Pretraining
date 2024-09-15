QUALITY_FEW_SHOT_COT_PROMPT = """## Example 1
### Question
In the context of "Les Misérables", written by Victor Hugo in 1862, what is the main setting of the novel? There is only one correct choice.
### Choices
A. London
B. Madrid
C. Paris
D. Rome
### Thought Process and Answer
Thought process: "Les Misérables" is primarily set in Paris, making C the correct choice. London, Madrid, and Rome are significant cities in other literary works but not in Victor Hugo's "Les Misérables". There is only one correct choice.
Answer: C.

## Example 2
### Question
In the context of "Brave New World", written by Aldous Huxley in 1932, what substance is widely used in the society to control citizens' happiness? There is only one correct choice.
### Choices
A. Gold
B. Soma
C. Silver
D. Iron
### Thought Process and Answer
Thought process: In Aldous Huxley's "Brave New World," Soma is used as a means to maintain social control by ensuring citizens' happiness, making B the correct choice. Gold, Silver, and Iron are not the substances used for this purpose in the book.
Answer: B.

## Example 3
### Question
In the context of "Romeo and Juliet", written by William Shakespeare in the early 1590s, what are the names of the two feuding families? There is only one correct choice.
Choices:
A. Montague and Capulet
B. Bennet and Darcy
C. Linton and Earnshaw
D. Bloom and Dedalus
### Thought Process and Answer
Thought process: In William Shakespeare's "Romeo and Juliet," the two feuding families are the Montagues and the Capulets, making A the correct choice. The Bennets and Darcys are in "Pride and Prejudice", the Lintons and Earnshaws in "Wuthering Heights", and Bloom and Dedalus in "Ulysses".
Answer: A.

## Example 4
### Question
In the context of "1984", written by George Orwell in 1949, what is the name of the totalitarian leader? There is only one correct choice.
### Choices
A. Big Brother
B. O'Brien
C. Winston Smith
D. Emmanuel Goldstein
### Thought Process and Answer
Thought process: In George Orwell's "1984," the totalitarian leader is known as Big Brother, making A the correct choice. O'Brien is a character in the novel, Winston Smith is the protagonist, and Emmanuel Goldstein is a rebel leader.
Answer: A.

## Example 5
### Question
In the context of "Moby-Dick", written by Herman Melville in 1851, what is the name of the ship's captain obsessed with hunting the titular whale? There is only one correct choice.
### Choices
A. Captain Hook
B. Captain Nemo
C. Captain Flint
D. Captain Ahab
### Thought Process and Answer
Thought process: In Herman Melville's "Moby-Dick," the ship's captain obsessed with hunting the whale is Captain Ahab, making D the correct choice. Captain Nemo is in "Twenty Thousand Leagues Under the Sea", Captain Flint in "Treasure Island", and Captain Hook in "Peter Pan".
Answer: D.

## Example 6
"""

OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES = """
As a knowledge analyzer, your task is to dissect and understand an article provided by the user. You are required to perform the following steps:
1. Summarize the Article: Provide a concise summary of the entire article, capturing the main points and themes.
2. Extract Entities: Identify and list all significant "nouns" or entities mentioned within the article. These entities should include but not limited to:
    * People: Any individuals mentioned in the article, using the names or references provided.
    * Places: Both specific locations and abstract spaces relevant to the content.
    * Object: Any concrete object that is referenced by the provided content.
    * Concepts: Any significant abstract ideas or themes that are central to the article's discussion.

Try to exhaust as many entities as possible. Your response should be structured in a JSON format to organize the information effectively. Ensure that the summary is brief yet comprehensive, and the list of entities is detailed and accurate.

Here is the format you should use for your response:

{
  "summary":  "<A concise summary of the article>",
  "entities": ["entity1", "entity2", ...]
}
"""

OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITY_SPECIFIC_QUESTIONS = """
As an examiner, you are tasked with creating reading comprehension questions for students based on a provided article and a specified entity referenced within it. Your role involves crafting questions and corresponding answers that fulfill the following criteria:

1. **Focus on the Entity**: Ensure all questions consistently center around the specified entity from the article.
2. **Encourage Deep Analysis**: Develop thought-provoking, open-ended questions that challenge students to think critically and analytically. Questions should:
   - Prompt students to reflect deeply, questioning the assumptions within the article.
   - Require students to evaluate evidence and consider alternative perspectives.
   - Encourage complex reasoning about the entity and its implications within the article's context.
3. **Comprehensive Answers**: For each question, provide a detailed solution that:
   - Explicitly connects back to the specified entity and its role or representation in the article.
   - Includes concrete references to specific paragraphs or sections of the article to support the answer.

Try to write as many questions as possible. Your response should be formatted to organize the questions and answers systematically. Here is the structure you should use:

### Questions and answers about <entity> in context of <title>
Question: <Question1 focusing on the entity>
Answer: <Detailed answer with references to the article>

Question: <Question2 focusing on the entity>
Answer: <Detailed answer with references to the article>
  ...

"""

OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves two main objectives:
1. Rephrasing Content: The user will identify two specific entities mentioned in the article. You are required to rephrase the content of the article twice:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
2. Analyzing Interactions: Discuss how the two specified entities interact within the context of the article.
3. Generating qeustions and answers: crafting questions and corresponding answers that fulfill the following criteria:
    - **Focus on the two Concepts/Terms**: Ensure all questions consistently center around the two concepts provided by the user.
    - **Encourage Deep Analysis**: Develop thought-provoking, open-ended questions that challenge students to think critically and analytically understand how the interaction between the two entities shape the article.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of Interaction between <entity1> and <entity2> in context of <title>
<Discussion on how the two entities interact within the article>

### Questions and answers about <entity1> and <entity2> in context of <title>
Question: <Question1 focusing on the two entities>
Answer: <Detailed answer with references to the article>

Question: <Question2 focusing on the two entities>
Answer: <Detailed answer with references to the article>

...
"""

OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves three main objectives:
1. Rephrasing Content: The user will identify three specific entities mentioned in the article. You are required to rephrase the content of the article three times:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
    * Lastly, emphasizing the third entity.
2. Analyzing Interactions: Discuss how these three specified entities interact within the context of the article.
3. Asking Questions: Generate some questions that involves all three entities and their interactions. Ensure your questions satisfies the following criteria:
    * Focus on the entities: Ensure all questions consistently center around the three entities specified.
    * Encourage Deep Analysis: Develop thought-provoking, open-ended questions that challenge students to think critically and analytically about the entities. Questions should:
        - Prompt students to reflect deeply on the meaning and implications of all three concepts.
        - Encourage complex reasoning about the concept’s broader implications in the context of the article.
Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of <title> in relation to <entity3>
<Rephrased content focusing on the third entity>

### Discussion of Interaction between <entity1>, <entity2> and <entity3> in context of <title>
<Discussion on how the three entities interact within the article>

### Question involving <entity1>, <entity2> and <entity3> in context of <title>
<Question1 that involves all three entities and their interactions>
<Answer to the Question1>

<Question2 that involves all three entities and their interactions>
<Answer to the Question2>

...
"""
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

from itertools import permutations

def uncapitalize_first(s):
    return s[0].lower() + s[1:]

def format_name(name):
    # Split the name by comma
    parts = name.split(',')

    # If there is a comma, we assume the format is "Lastname, Firstname"
    if len(parts) == 2:
        formatted_name = parts[1].strip() + ' ' + parts[0].strip()
    else:
        # If there's no comma, assume the format is already "Firstname Lastname"
        formatted_name = name.strip()

    return formatted_name

def second_last_character(input_string: str):
    try:
        # Remove newline characters from the end of the string
        modified_string = input_string.rstrip('\n')
        answer_index = LETTER_TO_INDEX[modified_string[-2]]
    except (KeyError, IndexError):
        answer_index = None
    return answer_index

def generate_all_answer_strings():
    choices = 'ABCD'
    all_answers = []
    for r in range(1, len(choices) + 1):
        for perm in permutations(choices, r):
            answer = ''.join(perm)
            all_answers.append(f'Answer: {answer}.\n')
    return all_answers