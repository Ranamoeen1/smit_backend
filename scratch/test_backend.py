import os
import traceback
from dotenv import load_dotenv
load_dotenv()

from services.document_parser import parse_cv
from services.vector_store import store_cv_in_vector_db
from services.agent import run_agent_workflow

try:
    print("Parsing CV...")
    with open('sample_cv.docx', 'rb') as f:
        file_bytes = f.read()
    text = parse_cv(file_bytes, 'sample_cv.docx')
    
    print("Storing in Vector DB...")
    vs = store_cv_in_vector_db(text)
    
    print("Running Agent Workflow...")
    res = run_agent_workflow(vs)
    
    print("SUCCESS")
    print(res.model_dump_json(indent=2))
except Exception as e:
    print(f"FAILURE: {str(e)}")
    traceback.print_exc()
