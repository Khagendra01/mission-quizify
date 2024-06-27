import streamlit as st
import os
import sys
import json

# Set the page config at the very beginning
st.set_page_config(page_title="Quiz Builder")

# 确保 Python 可以找到 `tasks` 文件夹中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from tasks.task_3.task_3 import DocumentProcessor
    from tasks.task_4.task_4 import EmbeddingClient
    from tasks.task_5.task_5 import ChromaCollectionCreator
    from langchain_core.prompts import PromptTemplate
    from langchain_google_vertexai import VertexAI
    print("Modules imported successfully")
except ImportError as e:
    print(f"Error importing modules: {e}")

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        self.topic = topic if topic else "General Knowledge"
        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions
        self.vectorstore = vectorstore
        self.llm = None
        self.question_bank = []  # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
            You must respond as a JSON object with the following structure:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}
            
            Context: {context}
            """
    
    def init_llm(self):
        self.llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.8,  # Increased for less deterministic questions
            max_output_tokens=500
        )

    def is_valid_json(self, json_str):
        try:
            json.loads(json_str)
            return True
        except ValueError as e:
            return False

    def clean_response(self, response):
        # 去除多余的换行符和回车符
        return response.replace('\n', '').replace('\r', '')

    def generate_question_with_vectorstore(self):
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore not provided.")

        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        retriever = self.vectorstore.as_retriever()
        prompt = PromptTemplate.from_template(self.system_template)
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        chain = setup_and_retrieval | prompt | self.llm

        try:
            response = chain.invoke(self.topic)
            print("LLM Response:", response)
            
            # 确保响应是字符串，如果不是，则将其转换为字符串
            if isinstance(response, dict):
                response = json.dumps(response)
            elif not isinstance(response, str):
                response = str(response)

            cleaned_response = self.clean_response(response)
            print("Raw Response for Debugging:", cleaned_response)
            if self.is_valid_json(cleaned_response):
                response_json = json.loads(cleaned_response)
                print("Decoded Response:", response_json)
                return response_json
            else:
                print("Invalid JSON response.")
                return None
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"Raw Response for Debugging: {cleaned_response}")
            return None
        except Exception as e:
            print(f"Failed to get response from LLM: {e}")
            return None

    def generate_quiz(self) -> list:
        self.question_bank = []

        for _ in range(self.num_questions):
            question_dict = self.generate_question_with_vectorstore()
            if not question_dict:
                print("Skipping invalid question.")
                continue
            print("Raw Question Dict:", question_dict)

            if self.validate_question(question_dict):
                print("Successfully generated unique question")
                self.question_bank.append(question_dict)
            else:
                print("Duplicate or invalid question detected.")

        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        if "question" not in question:
            return False
        for q in self.question_bank:
            if q["question"] == question["question"]:
                return False
        return True

# Test Generating the Quiz
if __name__ == "__main__":
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "sample-mission-427316",
        "location": "us-central1"
    }
    
    try:
        processor = DocumentProcessor()
        processor.ingest_documents()
        embed_client = EmbeddingClient(**embed_config)  # Initialize from Task 4
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
        print("DocumentProcessor, EmbeddingClient, and ChromaCollectionCreator initialized successfully")
    except Exception as e:
        print(f"Error initializing components: {e}")
    
    st.header("Quiz Builder")
    
    with st.form("Load Data to Chroma"):
        st.subheader("Quiz Builder")
        st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
        
        topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
        questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            try:
                chroma_creator.create_chroma_collection()
                st.write(topic_input)
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question_bank = generator.generate_quiz()
                if question_bank and len(question_bank) > 0:
                    st.header("Generated Quiz Questions: ")
                    for question in question_bank:
                        st.json(question)
                else:
                    st.error("No questions generated. Please check the logs for errors.")
            except Exception as e:
                print(f"Error during quiz generation: {e}")
                st.error("An error occurred during quiz generation. Please check the logs.")