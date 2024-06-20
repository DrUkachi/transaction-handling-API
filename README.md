# Transaction Semantic Similarity Search

This project implements a semantic similarity search system for transaction descriptions using transformer-based language models. It allows users to find transactions similar to a given input description based on semantic meaning rather than exact text matching.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to provide a scalable and efficient method to find transactions with similar descriptions using transformer-based language models. It involves tokenizing transaction descriptions, computing embeddings using a pre-trained model, and measuring semantic similarity using cosine similarity.

## Features

- **Pattern Matching:** This involves extracting user's names from a description in the transactions table and matching it to the users information.
- **Semantic Search:** Find transactions similar to a given description based on semantic meaning.
- **Efficient Embedding:** Utilize transformer models for computing embeddings efficiently.

## Setup Instructions

### Prerequisites

- Python 3.10+
- pip package manager
- Git

### Installation

1. Clone the repository:

   ```bash
   git clone <https://github.com/DrUkachi/transaction-handling-API/tree/main>
   cd transaction-semantic-similarity
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Transaction Data:**

   Ensure your transaction data is available in a suitable format (CSV).

2. **Run Semantic Similarity Search:**

   Run the `app.py` script to start the app

   ```bash
   python app.py 
   ```
3. **Use the Streamlit app to access the API:**

   Run the Streamlit app using the `main.py` script to start the app

   ```bash
   streamlit run app.py
   ```
   You should see this:

   ![Streamlit Application](<https://github.com/DrUkachi/nd0821-c3-starter-code/blob/main/screenshots/live-post.png>)

4. **You can also visit the local Swagger UI using the link:**
[API documentation](http://localhost:8000/docs).

## Directory Structure

```
transaction-handling-api/
│
├── app.py                # Main script for the api
├── README.md   # Project overview and instructions and description of solution
├── requirements.txt               # Python dependencies
└── transactions.csv                # transaction data
|___ users.csv                      # users data (example)
```

## Technologies Used

- **Transformers Library:** Hugging Face Transformers for transformer-based models.
- **Python Libraries:** pandas, numpy, scikit-learn for data handling and computation.
- **Git:** Version control for project management and collaboration.
- **FastAPI:** For converting the model to an API Web service. See deployed link [link](https://transaction-handling-api.onrender.com)

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any enhancements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Description of Solution 1

**Endpoint Definition**:
   - The endpoint `/match-users/` is defined as a POST method that expects a `transaction_id` as input.
   - It returns a JSON response with matched users sorted by relevance and includes the total number of matches.

2. **Dependencies**:
   - The endpoint uses a dependency - `(get_loaded_data)` to fetch data (`trans_data` and `users_data`) needed for processing. This assumes that `get_loaded_data` is a function returning loaded data, likely from some external source such as a database or file.

3. **Processing Logic**:
   - It first checks if the transaction with the provided `transaction_id` exists in `trans_data`. If not found, it raises a `404` HTTPException.
   - Extracts the transaction description and attempts to identify the user's name using several methods (`extract_name`, `extract_name_spacy`, `extract_name_distilbert`). These methods likely employ different techniques (like regex, NLP libraries) to extract names from the transaction description.
   - If a user name is successfully extracted, it proceeds to match this name against `users_data` using fuzzy matching (`fuzz.ratio` from `fuzzywuzzy` library) to calculate a similarity score (`match_metric`).
   - Matches are sorted in descending order of `match_metric` to prioritize more relevant matches.

4. **Response**:
   - The response JSON includes:
     - `"users"`: A list of dictionaries containing `id` and `match_metric` for each matched user, sorted by relevance.
     - `"total_number_of_matches"`: An integer indicating the total number of users matched.

5. **Error Handling**:
   - Handles exceptions and returns appropriate HTTP status codes (`404` for not found, `500` for internal server errors) with detailed error messages.

### Limitations of the Solution:

1. **Language Limitation**:
   - The solution assumes that all names are in Latin alphabet characters. It does not handle names in other languages or character sets, which could lead to mismatches if non-Latin characters are present.

2. **Name Extraction Issues**:
   - It might struggle with transaction descriptions that have unconventional formatting or where names are not clearly separated. For example, names written without spaces (e.g., "JohnSmith") could be challenging to extract accurately.

3. **Performance Considerations**:
   - Depending on the size of `users_data`, iterating through each user could become inefficient if `users_data` is very large. Optimization techniques like indexing or filtering based on initial criteria could improve performance.

### Improvements:

- **Multilingual Support**: Implement other methods to handle names in different languages using appropriate NLP libraries or techniques for name recognition and fuzzy matching, by training small sized and effecient LLMs for this particular task.
  
- **Enhanced Name Extraction**: Develop more robust techniques to handle diverse formats of transaction descriptions, such as names without spaces or with punctuation.

## Description of Solution 2:

The provided solution extends the FastAPI application with a new endpoint `/similar-transactions/` that uses language model embeddings to find transactions with descriptions similar to a given input string. Here’s a breakdown:

1. **Endpoint Definition**:
   - The endpoint `/similar-transactions/` is defined as a POST method that expects an `input_string` as input.
   - It returns a JSON response with transactions sorted by relevance based on their similarity to the `input_string`.

2. **Dependencies**:
   - It uses the following dependencies `(get_loaded_data)` to fetch transaction data (`trans_data`) needed for processing. The second element of the tuple (`_`) is unused here as it needs only the transaction data.

3. **Processing Logic**:
   - **Tokenization and Embedding Creation**:
     - Converts the `input_string` into tokens using a tokenizer (`tokenizer`), which is likely based on a pre-trained language model (e.g., BERT, RoBERTa). The tokenizer returns token IDs suitable for the model and is wrapped into a PyTorch tensor.
     - Utilizes a pre-trained language model (`model`) to obtain embeddings for the `input_string`. It computes the mean of the last hidden state embeddings (`input_output.last_hidden_state.mean(dim=1)`) to get a fixed-size representation of the input.
   
   - **Compute Similarity**:
     - Tokenizes and embeds all transaction descriptions (`trans_data['description']`). This involves padding and truncation to ensure consistent input lengths for efficient batch processing using PyTorch tensors.
     - Computes cosine similarity between the embedding of `input_string` and each transaction description embedding (`cosine_similarity`).
     - Assigns similarity scores to each transaction and adds them as a new column (`similarity_score`) to `trans_data`.

   - **Sorting and Response**:
     - Sorts `trans_data` based on `similarity_score` in descending order to prioritize transactions most similar to `input_string`.
     - Constructs the response JSON:
       - `"transactions"`: A list of dictionaries containing `id` and `description` of transactions sorted by similarity.
       - `"total_number_of_tokens_used"`: Number of tokens used in the input string's tokenizer representation (`input_tokens['input_ids'][0]`).

4. **Error Handling**:
   - Catches exceptions and returns a detailed error message with a `500` HTTP status code if any error occurs during processing.

### Limitations of the Solution:

1. **Contextual Understanding**:
   - **Semantic Meaning**: Although the solution uses language model embeddings, it may not capture nuanced semantic similarities effectively. Language models like BERT or RoBERTa encode text based on context, but they might not perfectly capture the intended semantic similarity for all use cases.
   
2. **Performance Considerations**:
   - **Computational Cost**: Calculating embeddings and similarity scores for potentially large datasets (`trans_data`) can be computationally intensive, especially in real-time applications or with large input strings.
   
3. **Model and Tokenizer Dependency**:
   - The solution assumes availability and compatibility with a specific tokenizer and language model (`model`). Changes in these components (like switching to a different model or tokenizer) could require substantial modifications.

4. **Handling Long Texts**:
   - The approach may struggle with very long input strings or descriptions, as transformers like BERT have a maximum token limit. Long texts may need to be segmented or truncated, potentially affecting similarity accuracy.

### Suggestions for Improvement:

- **Fine-tuning or Transfer Learning**: Consider fine-tuning a pre-trained language model on transaction-specific data to better capture domain-specific semantics.
  
- **Advanced Similarity Metrics**: Explore other similarity metrics beyond cosine similarity (e.g., BERTScore) that might better align with semantic similarity tasks.

- **Efficiency Enhancements**: Implement batching or parallelization strategies to optimize computation, especially for large datasets.

- **Handling Long Texts**: Develop strategies to handle long texts more effectively, such as hierarchical or attention-based approaches.

By addressing these limitations and considering improvements, the solution can be made more robust and suitable for a wider range of real-world applications where semantic similarity in transaction descriptions is crucial.


## Task 3: From PoC to Production

Taking a proof of concept (PoC) to production involves several key steps and considerations to ensure reliability, scalability, and maintainability. Below are some of my suggestions and improvements that can be made to transition from a PoC to a production-ready state for the Transaction Handling application, built with FastAPI.

### 1. **Deployment Strategy**

- **Containerization**: Use Docker to containerize your FastAPI application along with its dependencies. This ensures consistent deployment across different environments.

- **Orchestration**: Deploy containers using Kubernetes or Docker Compose for managing scalability, load balancing, and fault tolerance.

- **Deployment Automation**: Implement CI/CD pipelines (e.g., using Jenkins, GitLab CI/CD, or GitHub Actions) for automated testing, building, and deploying updates to production.

### 2. **Infrastructure Considerations**

- **Cloud Provider**: Choose a cloud provider (AWS, Azure, Google Cloud) based on your specific needs for scalability, region availability, compliance, and cost.

- **Database Management**: Use managed database services (e.g., AWS RDS, Azure SQL Database) for data storage, ensuring backups, replication, and scalability are managed effectively.

- **Monitoring and Logging**: Implement monitoring tools (e.g., Prometheus, Grafana) and centralized logging (e.g., ELK stack, Fluentd) to track application performance, errors, and usage metrics.

### 3. **Security**

- **Authentication and Authorization**: Implement OAuth2 authentication for securing API endpoints. Use JWT (JSON Web Tokens) for secure token-based authentication.

- **Data Encryption**: Ensure data at rest and in transit is encrypted using SSL/TLS. Utilize encryption libraries (e.g., Python's `cryptography` module) for sensitive data handling.

- **API Rate Limiting**: Implement rate limiting to prevent abuse and ensure fair usage of the API.

### 4. **Performance Optimization**

- **Caching**: Use caching mechanisms (e.g., Redis) for frequently accessed data to improve response times and reduce load on the database.

- **Query Optimization**: Optimize database queries using indexing, query optimization techniques, and database performance tuning.

- **Asynchronous Tasks**: Use asynchronous programming (e.g., FastAPI's `BackgroundTasks`, Celery) for long-running tasks to improve API responsiveness.

### 5. **Testing and Quality Assurance**

- **Unit Testing**: Write comprehensive unit tests using tools like pytest to validate individual components and functions.

- **Integration Testing**: Perform integration tests to ensure different parts of the application work together correctly.

- **Load Testing**: Conduct load testing (e.g., using tools like Apache JMeter, Locust) to simulate real-world usage and identify performance bottlenecks.

### 6. **Documentation and Monitoring**

- **API Documentation**: Provide clear, comprehensive API documentation using tools like Swagger UI or FastAPI's built-in documentation generator.

- **Operational Documentation**: Document deployment procedures, environment variables, configuration settings, and troubleshooting guides.

- **Health Checks**: Implement health checks for the application and its dependencies to monitor availability and performance.

### 7. **Maintenance and Support**

- **Version Control**: Use version control (e.g., Git) to manage model changes, code changes, branching strategies, and release process or management.

- **Incident Response**: Establish procedures for incident response and monitoring alerts to promptly address and mitigate issues.
