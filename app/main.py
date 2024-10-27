from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
from .models.codegen import CodegenRequest
from .services import github_service, llm_service, diff_service, visualization
from app.services import supabase_service
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from urllib.parse import urljoin

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


app = FastAPI()

# Serve static files
static_dir = os.path.join(os.getcwd(), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve output files
output_dir = os.path.join(os.getcwd(), "outputs")
app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")

def upload_file_to_s3(file_path, s3_key):
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        # For public buckets
        s3_url = f"https://tinygen.s3.us-east-2.amazonaws.com/{s3_key}"
        # If bucket is private, use pre-signed URL
        # presigned_url = s3_client.generate_presigned_url(
        #     'get_object',
        #     Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
        #     ExpiresIn=3600
        # )
        return s3_url  # or presigned_url if using private bucket
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="Credentials not available")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return JSONResponse(status_code=404, content={"error": "index.html not found"})

@app.post("/generate")
async def generate_code(request: CodegenRequest):
    try:
        # Fetch repo content
        repo_content = github_service.fetch_repo_content(request.repoUrl)
        
        # Find relevant files
        relevant_files = llm_service.find_relevant_files(repo_content, request.prompt)
        
        # Generate initial changes only for relevant files
        initial_changes = llm_service.generate_changes(relevant_files, request.prompt)
        
        # Perform multiple reflections until satisfied
        final_changes = llm_service.perform_reflection(initial_changes, request.prompt, max_iterations=3)
        
        # Generate diff
        diff = diff_service.generate_diff(repo_content, final_changes)
        
        # Sanitize diff by removing null bytes
        sanitized_diff = diff.replace('\u0000', '')

         # Generate import visualization graph
        # graph_local_path = os.path.join(os.getcwd(), "import_graph.html") 
        # graph_path = visualization.visualize_import_graph(repo_content, graph_local_path)
        
        # # Upload the graph to S3
        # s3_key = f"import_graphs/{request.repoUrl.replace('/', '_')}_import_graph.html"
        # graph_url = upload_file_to_s3(graph_local_path, s3_key)
      
        # Generate import visualization graph
        graph_path = visualization.visualize_import_graph(repo_content, os.path.join(output_dir, "import_graph.html"))
    
        # Store prompt and response in Supabase
        supabase_service.log_generation(request.repoUrl, request.prompt, sanitized_diff)
        
        return JSONResponse(content={"diff": sanitized_diff, "graph_url": f"/outputs/import_graph.html"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)