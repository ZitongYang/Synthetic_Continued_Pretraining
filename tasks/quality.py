from collections import Counter
from typing import List, Dict, Optional

import numpy as np

from tasks.task_abc import Question, Document, Task
from utils.io_utils import jload_list, jload
from utils.prompt_utils import (format_name, uncapitalize_first, second_last_character,
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES,
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITY_SPECIFIC_QUESTIONS,
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS,
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS,
                                QUALITY_FEW_SHOT_COT_PROMPT)


class QuALITYQuestion(Question):
    def __init__(self,
                 statement: str,
                 options: List[str],
                 answer: str,
                 ishard: bool,
                 attempts: List[Dict]=[dict()],
                 **kwargs):
        statement_dict = dict(content=statement, options=options)
        super().__init__(statement_dict, answer, attempts)
        self.ishard = ishard

    def _formatted_choice(self):
        formatted = ""
        for i, option in enumerate(self.statement['options']):
            # Convert 0, 1, 2, 3, ... to A, B, C, D, ...
            letter = chr(65 + i)
            formatted += f"{letter}. {option}\n"
        return formatted

    def prompt(
            self,
            document_context: Optional[str],
            add_thought_process: bool,
            sep_after_question: str
    ):
        formatted = "### Question\n"

        if document_context is None:
            formatted += f"{self.statement['content']} There is only one correct choice.{sep_after_question}"
        else:
            formatted += f"{document_context} {uncapitalize_first(self.statement['content'])} There is only one correct choice.{sep_after_question}"

        formatted += f"### Choices\n"
        formatted += self._formatted_choice()

        if add_thought_process:
            if sep_after_question == '\n\n':
                formatted += "\n"

            formatted += "### Thought Process and Answer\n"
            formatted += "Thought process:"

        return formatted

    def llama_parse_answer(self, raw_output: str):
        if raw_output is None:
            return dict()
        else:
            answer_index = second_last_character(raw_output)
            if answer_index is not None:
                answer_content = self.statement['options'][answer_index]
            else:
                answer_content = None
            return dict(reasoning=raw_output,
                        answer_index=answer_index,
                        answer_content=answer_content)

    def _iscorrect(self, attempt: Dict):
        return self.answer == chr(attempt['answer_index'] + 65)

    def iscorrect(self, attempt_index: int = 0):
        return self._iscorrect(self.attempts[attempt_index])

    def asdict(self):
        return dict(
            statement=self.statement['content'],
            options=self.statement['options'],
            answer=self.answer,
            ishard=self.ishard,
            attempts=self.attempts,
            formatted_prompt=self.formatted_prompt
        )

    def majority_vote(self, n_samples: int):
        if len(self.attempts)>0:
            self.attempts = self.attempts[:n_samples]
            indices = [attempt['answer_index'] for attempt in self.attempts]
            counts = Counter(indices)
            most_freq = max(counts, key=counts.get)
            for attempt in self.attempts:
                if attempt['answer_index'] == most_freq:
                    self.attempts = [attempt]
        else:
            self.attempts = [dict()]


class QuALITYArticle(Document):
    def __init__(self, text: str, questions: List[Dict],
                 title: str, author: str, year: str, topic: str, **kwargs):
        questions = [QuALITYQuestion(**qdict) for qdict in questions]
        super().__init__(text, questions)
        self.title = title
        self.author = author
        self.year = year
        self.topic = topic

    @property
    def uid(self):
        return ' by '.join([self.title, self.author])

    @property
    def content(self):
        """ Full article content """
        result = f"\"{self.title}\", {format_name(self.author)}, {self.year}."
        result += f"\n {self.text}"
        return result

    @property
    def _article_context(self):
        """ Context prefix for article free questioning"""
        return f"In the context of \"{self.title}\", written by {format_name(self.author)} in {self.year},"

    def question_prompts(self, add_document_context: bool, add_thought_process: bool, sep_after_question: str):
        """All questions for a given article.

        Args:
            add_document_context: bool, whether to prepend the article context to the questions.
                For example, we set this to False in our non-contextualized question evaluation.
            add_thought_process: bool, whether to add the thought process suffix to the questions.
                We set this to False in our query embeddings.
            sep_after_question: str, either '\n' or '\n\n' depending on the prompt format.

        Returns:
            List of questions.
         """
        prompts = []
        for q in self.questions:
            prompts.append(
                q.prompt(
                    self._article_context if add_document_context else None,
                    add_thought_process,
                    sep_after_question)
            )

        return prompts

    def asdict(self):
        return dict(title=self.title,
                    author=self.author,
                    year=self.year,
                    topic=self.topic,
                    text=self.text,
                    questions=[q.asdict() for q in self.questions])

class QuALITY(Task):
    """
    >>> from task import *
    >>> quality_raw = QuALITY('all')
    >>> len(quality_raw.articles)
    265
    """
    openai_system_generate_entities = OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES
    openai_system_generate_entity_specific_questions = OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITY_SPECIFIC_QUESTIONS
    openai_system_generate_two_entity_relations = OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS
    openai_system_generate_three_entity_relations = OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS
    llama_cot_prompt = QUALITY_FEW_SHOT_COT_PROMPT

    @staticmethod
    def _load_split(split: str):
        file_path = f'data/dataset/raw/QuALITY.v1.0.1.htmlstripped.{split}'
        return jload_list(file_path)

    def _create_documents(self):
        documents = []
        for adict in self._data:
            questions = []
            for qdict in adict['questions']:
                question = dict(statement=qdict['question'],
                                options=qdict['options'],
                                answer=chr(int(qdict['gold_label'])-1+65),
                                ishard=bool(qdict['difficult']))
                questions.append(question)
            questions = sorted(questions, key=lambda x: x['statement'])
            document = QuALITYArticle(
                              title=adict['title'],
                              author=adict['author'],
                              text=adict['article'],
                              year=adict['year'],
                              topic=adict['topic'],
                              questions=questions)
            documents.append(document)
        super().__init__('quality', sorted(documents, key=lambda x: x.title))

    def _dedup(self):
        deuped_documents = {}
        for document in self.documents:
            key = document.uid
            if key not in deuped_documents:
                deuped_documents[key] = document
            else:
                deuped_documents[key].questions += document.questions
        self.documents = list(deuped_documents.values())

    def __init__(self, split: str):
        self.split = split
        if split in ['train', 'dev']:
            self._data = QuALITY._load_split(split)
            self._create_article()
        elif split in ['all', '50']:
            self._data = QuALITY._load_split('train') + QuALITY._load_split('dev')
            self._create_documents()
            self._dedup()
            if split == '50':
                self.documents = self.documents[:50]
        elif split=='test':
            super().__init__('quality', None)
        else:
            raise ValueError(f"Invalid split: {split}")

    def load_attempts_json(self, file_path: str):
        loaded_articles_data = jload(file_path)

        attempted_articles = []
        for adict in loaded_articles_data:
            article = QuALITYArticle(**adict)
            attempted_articles.append(article)
        super().__init__('quality', sorted(attempted_articles, key=lambda x: x.title))

    def all_questions(self, add_document_context: bool, add_thought_process: bool, sep_after_question: str):
        prompts = []
        for document in self.documents:
            prompts += document.question_prompts(add_document_context, add_thought_process, sep_after_question)

        return prompts

    @staticmethod
    def _attempts_stats(attempt_index: int, documents: List[QuALITYArticle]):
        attempted_hard_q = 0
        attempted_non_hard_q = 0
        correct_hard_q = 0
        correct_non_hard_q = 0

        for article in documents:
            for question in article.questions:
                if question.attempts[attempt_index]:
                    try:
                        if question.attempts[attempt_index]['answer_index'] in [0, 1, 2, 3]:
                            if question.ishard:
                                attempted_hard_q += 1
                                if question.iscorrect(attempt_index):
                                    correct_hard_q += 1
                            else:
                                attempted_non_hard_q += 1
                                if question.iscorrect(attempt_index):
                                    correct_non_hard_q += 1
                    except KeyError:
                        print(f"KeyError: {question.attempts[attempt_index]}")

        return dict(attempted_hard_q=attempted_hard_q,
                    attempted_non_hard_q=attempted_non_hard_q,
                    correct_hard_q=correct_hard_q,
                    correct_non_hard_q=correct_non_hard_q)

    @staticmethod
    def _question_stats(documents: List[QuALITYArticle]):
        # compute the number of answers with index 0, 1, 2, 3
        answerhist = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for article in documents:
            for question in article.questions:
                answerhist[question.answer] += 1

        # compute the number of hard questions
        hard_q = 0
        non_hard_q = 0
        for article in documents:
            for question in article.questions:
                if question.ishard:
                    hard_q += 1
                else:
                    non_hard_q += 1
        return dict(answerhist=answerhist, hard_q=hard_q, non_hard_q=non_hard_q)

    @staticmethod
    def _performance_stats_for_documents(documents: List[QuALITYArticle]):
        question_stats = QuALITY._question_stats(documents)

        def calculate_one_attempt(attempts_stats: Dict):
            def _div_nan_if_zero(a, b):
                if a==0 or b==0:
                    return np.nan
                return a/b

            hard_attempt_rate = _div_nan_if_zero(attempts_stats['attempted_hard_q'], question_stats['hard_q'])
            hard_accuracy = _div_nan_if_zero(attempts_stats['correct_hard_q'], attempts_stats['attempted_hard_q'])
            non_hard_attempt_rate = _div_nan_if_zero(attempts_stats['attempted_non_hard_q'], question_stats['non_hard_q'])
            non_hard_accuracy = _div_nan_if_zero(attempts_stats['correct_non_hard_q'], attempts_stats['attempted_non_hard_q'])
            overall_attempt_rate = _div_nan_if_zero(attempts_stats['attempted_hard_q']+attempts_stats['attempted_non_hard_q'], question_stats['hard_q']+question_stats['non_hard_q'])
            overall_accuracy = _div_nan_if_zero(attempts_stats['correct_hard_q']+attempts_stats['correct_non_hard_q'], attempts_stats['attempted_hard_q']+attempts_stats['attempted_non_hard_q'])

            return dict(
                hard_attempt_rate=hard_attempt_rate,
                hard_accuracy=hard_accuracy,
                non_hard_attempt_rate=non_hard_attempt_rate,
                non_hard_accuracy=non_hard_accuracy,
                overall_attempt_rate=overall_attempt_rate,
                overall_accuracy=overall_accuracy
            )
        attempts_stats = QuALITY._attempts_stats(0, documents)
        one_attempt = calculate_one_attempt(attempts_stats)

        result = dict()
        for key, value in one_attempt.items():
            result[key] = dict(mean=value, std=0)
        return result

    def performance_stats(self):
        total_documents = len(self.documents)
        val_documents = self.documents[:int(0.2*total_documents)]
        test_documents = self.documents[int(0.2*total_documents):]
        result = dict(val=QuALITY._performance_stats_for_documents(val_documents),
                      test=QuALITY._performance_stats_for_documents(test_documents))
        return result