from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
from .models.codegen import CodegenRequest
from .services import github_service, llm_service, diff_service, visualization
from app.services import supabase_service

load_dotenv()

app = FastAPI()

# Serve static files
static_dir = os.path.join(os.getcwd(), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve output files
output_dir = os.path.join(os.getcwd(), "outputs")
app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")

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
        
        # Rank and select top files
        top_files = llm_service.rank_and_select_files(relevant_files)
        
        # Generate initial changes only for relevant files
        initial_changes = llm_service.generate_changes(top_files, request.prompt)
        
        # Perform multiple reflections until satisfied
        final_changes = llm_service.perform_reflection(initial_changes, request.prompt, max_iterations=3)
        
        # Generate diff
        diff = diff_service.generate_diff(repo_content, final_changes)
        
        # Sanitize diff by removing null bytes
        sanitized_diff = diff.replace('\u0000', '')
        
        # Generate import visualization graph
        graph_path = visualization.visualize_import_graph(repo_content, os.path.join(output_dir, "import_graph.html"))
        
        # Store in Supabase
        supabase_service.log_generation(request.repoUrl, request.prompt, sanitized_diff)
        
        return JSONResponse(content={"diff": sanitized_diff, "graph_url": f"/outputs/import_graph.html"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)