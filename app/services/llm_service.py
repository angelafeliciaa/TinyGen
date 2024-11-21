from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import re
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Azure Cognitive Search Credentials
search_service_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")
search_service_api_key = os.getenv("SEARCH_SERVICE_API_KEY")
index_name = os.getenv("SEARCH_INDEX_NAME")

credential = AzureKeyCredential(search_service_api_key)

# Initialize the SearchClient
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=credential
)

def generate_embedding(text):
    response = openai.Embedding.create(
        input=text,
        engine='text-embedding-ada-002' 
    )
    embedding = response['data'][0]['embedding']
    return embedding

def index_code_files(repo_content):
    documents = []
    for idx, (file_path, content) in enumerate(repo_content.items()):
        if not should_process_file(file_path):
            continue
        embedding = generate_embedding(content)
        documents.append({
            "id": str(idx),
            "file_path": file_path,
            "content": content,
            "embedding": embedding
        })
    # Upload documents to Azure AI Search
    results = search_client.upload_documents(documents=documents)
    print(f"Indexed {len(results)} documents.")


# could give examples of prompt and response, but that won't work for all cases, 
# although it would give very accurate results for the specific sample output we're given.

def should_process_file(file_path):
    # List of directories and file extensions to ignore
    ignore_dirs = ['.idea', '.git', '__pycache__', 'venv', 'env']
    ignore_extensions = ['.pyc', '.pyo', '.pyd', '.db', '.lock', '.toml', '.md', '.txt']
    ignore_filenames = ['requirements.txt', 'Pipfile', 'poetry.lock']

    # Check if the file is in an ignored directory
    if any(ignored_dir in file_path.split(os.sep) for ignored_dir in ignore_dirs):
        return False

    # Check if the file has an ignored extension
    if any(file_path.endswith(ext) for ext in ignore_extensions):
        return False

    # Check if the file is in the list of ignored filenames
    if os.path.basename(file_path) in ignore_filenames:
        return False

    return True

def find_relevant_files(repo_content: dict, prompt: str) -> dict:

    # Generate embedding for the prompt
    prompt_embedding = generate_embedding(prompt)

    # Perform vector search
    results = search_client.search(
        search_text="",  # Empty string because we're using vector search
        vectors=[
            {
                "value": prompt_embedding,
                "fields": "embedding",
                "k": 10  # Number of top results to return
            }
        ],
        select=["file_path", "content"],
        top=10  # Ensure this matches 'k' in the vector
    )

    relevant_files = {}

    for result in results:
        file_path = result['file_path']
        content = result['content']
        score = result['@search.score']  # You can use this if needed
        relevant_files[file_path] = {
            'content': content,
            'score': score
        }

    return relevant_files


def is_special_file(file_path: str) -> bool:
    """Check if the file requires special handling."""
    file_name = os.path.basename(file_path).lower()
    special_files = ['__init__.py', 'index.js']
    return file_name in special_files

def clean_code_block(code: str) -> str:
    # Remove leading and trailing whitespace
    code = code.strip()
    
    # Remove leading '''python or ''' if present
    code = re.sub(r'^\'\'\'(?:python)?\s*', '', code)
    
    # Remove trailing ''' if present
    code = re.sub(r'\s*\'\'\'$', '', code)
    
    # Remove any remaining ''' markers within the code
    code = code.replace("'''", "")
    
    return code.strip()

def generate_changes(top_files: dict, prompt: str) -> dict:
    changes = {}
    for file_path, file_info in top_files.items():
        is_special = is_special_file(file_path)
        
        system_message = (
            "You are a helpful assistant that modifies code based on user prompts. Follow these rules strictly:\n"
            "1. Only suggest changes that are directly related to the user's prompt.\n"
            "2. If you think there needs to be a change to fulfill the prompt, respond with the entire updated code for the file.\n"
            "3. If no changes are needed, respond with 'No changes needed.'\n"
            "4. Provide only the modified code, not explanations or comments about the changes.\n"
            "5. Do not use markdown formatting or code block syntax (like '''python or ''').\n"
            "6. Do not include any text before or after the code.\n"
            "7. The user prompt might give potential solutions or examples to solve the problem, but do not jump to conclusions over these examples. Carefully consider the best approach.\n"
        )
        
        if is_special:
            system_message += (
                "8. This is a special file (__init__.py or index.js). Be extra cautious:\n"
                "   - For __init__.py: Only modify if absolutely necessary. Prefer adding imports over adding logic.\n"
                "   - For index.js: Ensure changes don't break the module's main functionality or exports.\n"
            )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Given the following code in {file_path}:\n\n{file_info['content']}\n\nApply the following change if necessary: {prompt}\n\nProvide the entire updated code for the file:"}
            ],
            max_tokens=2048,
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        if content != "No changes needed.":
            # Clean the code block
            content = clean_code_block(content)
            changes[file_path] = content + '\n'  # Ensure there's a newline at the end

    return changes

def perform_reflection(changes: dict, prompt: str, max_iterations: int = 3) -> dict:
    for iteration in range(max_iterations):
        reflected_changes = {}
        changes_made = False
        
        for file_path, content in changes.items():
            is_special = is_special_file(file_path)
            
            system_message = (
                "You are a code reviewer assistant. Follow these rules strictly:\n"
                "1. Ensure code changes are relevant to the given prompt and in the correct file.\n"
                "2. Do not write comments, summaries, or explanations.\n"
                "3. If no changes are needed, return the original code exactly.\n"
                "4. If improvements are needed, provide only the corrected code.\n"
                "5. Do not add any text that isn't part of the code.\n"
            )
            
            if is_special:
                system_message += (
                    "6. This is a special file (__init__.py or index.js). Be extra cautious:\n"
                    "   - For __init__.py: Only approve changes if absolutely necessary. Prefer imports over logic.\n"
                    "   - For index.js: Ensure changes don't break the module's main functionality or exports.\n"
                )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Problem to solve:\n\n{prompt}\n\nReview these changes in {file_path}:\n\n{content}\n\nReturn the code as-is if appropriate, or provide corrected code if needed."}
                ],
                max_tokens=2048,
                temperature=0
            )
            reflection = response.choices[0].message.content.strip()
            
            # Clean the reflection
            cleaned_reflection = '\n'.join(
                line for line in reflection.split('\n')
                if not line.strip().startswith(('+', '-', '#')) and 'changes needed' not in line.lower()
            )
            
            reflected_changes[file_path] = cleaned_reflection.strip() or content
            
            if reflected_changes[file_path] != content:
                changes_made = True
        
        if not changes_made:
            # If no changes were made in this iteration, we're satisfied
            return changes
        
        # Update changes for the next iteration
        changes = reflected_changes
    
    # If we've reached max_iterations and still making changes, return the latest version
    return changes