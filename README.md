# Information Retrieval System

This project implements an Information Retrieval (IR) system with advanced text processing and search capabilities. The system combines traditional IR techniques with modern deep learning approaches to provide efficient and accurate text search functionality.

## Features

- Text preprocessing and cleaning
- Inverted index construction
- TF-IDF and BM25 ranking
- Query expansion using RM3
- BERT-based semantic embeddings
- Interactive GUI for search queries
- Support for multiple document categories

## Prerequisites

- Python 3.x
- Java Development Kit (JDK) 22
- Required Python packages (install via pip):
  ```
  pyterrier
  nltk
  pandas
  torch
  tensorflow
  tensorflow-hub
  transformers
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. Set up Java environment:
   - Install JDK 22
   - Set JAVA_HOME environment variable to "C:\\Program Files\\Java\\jdk-22"

## Project Structure

- `main.py`: Main application file containing the IR system implementation
- `main.ipynb`: Jupyter notebook version of the implementation
- `text.csv`: Processed text data
- `archive_2/`: Directory containing the original text documents
- `mySecondIndex/`: Directory containing the built search index

## Usage

1. Run the main application:
   ```bash
   python main.py
   ```

2. The GUI will open with a search interface where you can:
   - Enter your search query
   - View ranked search results
   - See query expansion results
   - View term embeddings

## Search Features

The system implements several search features:
- Basic term matching
- TF-IDF ranking
- BM25 ranking
- Query expansion using RM3
- BERT-based semantic search

## Categories

The system supports documents from the following categories:
- Business
- Entertainment
- Food
- Graphics
- Historical
- Space
- Sport
- Technology

## Contributing

Feel free to submit issues and enhancement requests.

