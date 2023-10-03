import requests

host = "http://localhost:5000/"
questions = [
    "Who is Ilya Sutskever?",
    "What did Elon Musk say about OpenAI?",
    "Is there any existencial threat with the development of AI?",
    "What is so interesting about the story of OpenAI?",
    "Why did ChatGPT become some famous?"
]
pdf_file_path = "./openai.pdf"


def clear_document():
    response = requests.delete(host + "clear_document")
    response.raise_for_status()


def run_queries():
    for question in questions:
        response = requests.post(
            host + "query_document",
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json; charset=utf-8'
            },
            json={"text": question}
        )
        print(response.content.decode())
        response.raise_for_status()


def upload_document():
    with open(pdf_file_path, 'rb') as file:
        response = requests.post(
            host + "upload_document",
            files={
                'document':
                ("document.pdf", file, 'application/pdf')
            }
        )


def run_requests():
    clear_document()
    run_queries()
    upload_document()
    run_queries()


if __name__ == "__main__":
    run_requests()
