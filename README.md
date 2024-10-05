# TinyGen

TinyGen, a toy version of Codegen, is an AI-powered code evolution catalyst - transforming ideas into precise diffs with a simple prompt!

## Live Demo

A live demo of TinyGen is available at [https://tinygen-31646a1cc468.herokuapp.com](https://tinygen-31646a1cc468.herokuapp.com). Navigate to https://tinygen-31646a1cc468.herokuapp.com/docs to see the API documentation.

## Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/angelafeliciaa/TinyGen.git
   cd tinygen
   ```

2. Create and activate a virtual environment:
   On Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   On macOS and Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

5. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

6. Open `http://localhost:8000` in your browser to use the application.

## Usage

Enter a natural language description of the changes you want to make to your code, and TinyGen will generate the corresponding diff.

## Technologies Used

- FastAPI
- OpenAI API
- Supabase

---
