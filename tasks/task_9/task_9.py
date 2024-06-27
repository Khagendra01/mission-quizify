import streamlit as st
import os
import sys
import json

# Absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from tasks.task_8.task_8 import QuizGenerator

class QuizManager:
    def __init__(self, questions: list):
        self.questions = questions
        self.total_questions = len(questions)

    def get_question_at_index(self, index: int):
        valid_index = index % self.total_questions
        return self.questions[valid_index]

    def next_question_index(self, direction=1):
        if 'question_index' not in st.session_state:
            st.session_state['question_index'] = 0
        st.session_state['question_index'] = (st.session_state['question_index'] + direction) % self.total_questions

def parse_json_response(raw_response):
    if isinstance(raw_response, dict):
        return raw_response
    try:
        # Remove leading/trailing backticks and any surrounding whitespace
        cleaned_response = raw_response.strip().strip("```")
        # Remove leading 'json' identifier if present
        if cleaned_response.startswith("json"):
            cleaned_response = cleaned_response[4:].strip()
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

# Test Generating the Quiz
if __name__ == "__main__":
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "sample-mission-427316",
        "location": "us-central1"
    }

    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)

        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        question_bank = None

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                st.write(topic_input)
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                raw_question_bank = generator.generate_quiz()
                
                # Parse the raw responses
                question_bank = [parse_json_response(question) for question in raw_question_bank if parse_json_response(question)]

    if question_bank:
        screen.empty()
        quiz_manager = QuizManager(question_bank)
        
        if 'question_index' not in st.session_state:
            st.session_state['question_index'] = 0

        with st.container():
            st.header("Generated Quiz Question:")

            # Display the current question
            index_question = quiz_manager.get_question_at_index(st.session_state['question_index'])
            choices = [f"{choice['key']}) {choice['value']}" for choice in index_question['choices']]

            st.write(index_question['question'])
            answer = st.radio('Choose the correct answer', choices)

            answer_submitted = st.button("Submit Answer")
            next_question = st.button("Next Question")
            previous_question = st.button("Previous Question")

            if answer_submitted:
                correct_answer_key = index_question['answer']
                if answer.startswith(correct_answer_key):
                    st.success("Correct!")
                else:
                    st.error("Incorrect!")

            if next_question:
                quiz_manager.next_question_index(direction=1)
                st.experimental_rerun()

            if previous_question:
                quiz_manager.next_question_index(direction=-1)
                st.experimental_rerun()