#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:23:28 2024

@author: ssalie
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import json
import tempfile
import logging
from . import invoice_utils as iu #use this when building Dockerfile and comment out below
#import invoice_utils as iu

app = FastAPI()
logger = logging.getLogger(__name__)

def extract_data(file_path: str) -> dict:
    try:
        text = iu.text_extraction(file_path)
        llm_raw = iu.llm_key_extraction(text)
        llm_clean = iu.val_clean(llm_raw)
        output = json.dumps(llm_clean)
        return json.loads(output)
    except Exception as e:
        logger.error(f"Error extracting data from {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error processing file")

def process_file(file: UploadFile):
    """Process the file and return the extracted data."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    try:
        filename_data = extract_data(temp_file_path)
        return filename_data
    finally:
        os.remove(temp_file_path)

@app.post("/upload-pdf-return-invoice-data/")
async def upload_pdf_return_invoice_data(file: UploadFile = File(...)):    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
    
    output = process_file(file)
    return output