# Interactive PDF Querying Application with AWS Bedrock and Streamlit

## Overview
This project is an interactive web application that allows users to query PDF documents using advanced machine learning models. The application integrates AWS Bedrock's Titan Embeddings Model and multiple LLMs (Claude and LLaMA2) to handle user queries. It also supports dynamically updating PDFs and vector stores, with the ability to save responses as PDF files.

## Technologies Used
- **Streamlit**: For developing the interactive web application interface.
- **AWS Bedrock**: Utilizing Titan Embeddings Model and Bedrock LLMs (Claude and LLaMA2).
- **LangChain**: Implementing PyPDFDirectoryLoader for data ingestion and RecursiveCharacterTextSplitter for text processing.
- **FAISS**: For efficient vector storage and retrieval.
- **FPDF**: To save query responses as PDF files.

## Features
- **Interactive Querying**: Users can input queries to be answered based on the content of the uploaded PDF documents.
- **Dynamic Updates**: Buttons to update PDFs and vector store dynamically.
- **PDF Saving**: Functionality to save LLM outputs as PDF files.
- **Real-Time Feedback**: Real-time feedback through Streamlit's user interface elements.

## Installation

### Prerequisites
- Python 3.6+
- AWS account with necessary permissions

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/pdf-querying-app.git
   cd pdf-querying-app
2. **virtual env**:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install the dependencies:

bash
Copy code
pip install -r requirements.txt
Set up AWS credentials:
Ensure your AWS credentials and region are correctly configured. You can do this by setting environment variables or using the AWS CLI.

bash
Copy code
aws configure
Run the application:

bash
Copy code
streamlit run app.py
