"""
app/main.py
Main application file for the PPT-PDFGen-AI service using FastAPI.
Defines API endpoints for searching, crawling, generating content using RAG, and creating downloadable PDF/PPT files.
Endpoints:      
- /search (POST): Perform web search and crawl results
- /ask (POST): Generate content using RAG pipeline
- /generate (POST): Generate PDF or PPT files based on web search results
- /download/{filename} (GET): Download a generated file
"""

import sys
import asyncio
import uvicorn
import subprocess
import os
import re
import uuid
import secrets
import requests

from datetime import datetime
from fastapi import FastAPI, Depends, Security, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.security.api_key import APIKey, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
from fastapi.responses import FileResponse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the configuration and logging modules
from app.rag import RAGRunner
from shared.config import Config
from app.crawler import WebCrawler
from app.file_generator import generate_ppt, generate_pdf
import shared.logger as logger

# Initialize the FastAPI application
app = FastAPI(
    title="AI PPT & PDF Generator",
    description="Generate AI-powered PPT and PDF enriched with insights extracted from relevant articles",
    version="1.0.0 (DCI 2025 Batch 2 Final Project)"
)

# Global crawler instance
crawler = None

# Request model definition
class SearchRequest(BaseModel):
    """Search request model

    Attributes:
        query: Search query string
        limit: Limit on the number of links returned results. Ex: 5 means only return top 5 links.
    """
    query: str
    limit: int = 5  # Default limit

class CrawlRequest(BaseModel):
    """
    Crawling request model

    Attributes:
        urls: List of URLs to crawl
        instruction: Crawling instruction, usually a search query
    """
    urls: list[str]
    instruction: str
    
class AskRequest(BaseModel):
    """
    RAG request model

    Attributes:
        Type: Content and document type (pdf/ppt)
        text_inject: Text from website crawling
        query: User and system prompt query
        Topic: Main topic user need
    """
    type: str
    text_inject: str
    query: str
    topic: str

class GenerateRequest(BaseModel):
    """
    Combines search + ask into one request.

    Attributes:
        prompt: Custom prompt for RAG generation
        topic: Main topic (for styling PDF/PPT output)
        type: Output type - "pdf" or "ppt"
        limit: Limit on the number of links returned results. Ex: 5 means only return top 5 links.
    """
    query: str
    topic: str
    type: str
    limit: int = 5  # Default limit

# Set up API key security
API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=True)

# Function to verify the key
async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    if api_key_header == Config.ACCESS_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY"
        )

@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler

    Executed when the FastAPI application starts; responsible for initializing the crawler and installing the necessary browser.
    """
    global crawler

    # Configuring log levels
    logger.setup_logger("INFO")
    logger.info("PPT-PDFGen-AI service is starting...")

    # Check and install browsers
    logger.info("Check Playwright Browser...")
    try:
        # Try installing your browser
        subprocess.run([sys.executable, "-m", "playwright",
                       "install", "chromium"], check=True)
        logger.info("Playwright browser is installed successfully or already exists")
    except subprocess.CalledProcessError as e:
        logger.error(f"Browser installation failed: {e}")
        raise

    # Initialize the crawler
    crawler = WebCrawler()
    await crawler.initialize()
    logger.info("Crawler initialization completed")

    # Ensure output directory exists
    FileService.ensure_output_directory()
    
    logger.info(f"The API service runs on: http://{Config.API_HOST}:{Config.API_PORT}")
    logger.info("PPT-PDFGen-AI service startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler

    Executed when the FastAPI application shuts down; responsible for releasing crawler resources.
    """
    global crawler
    if crawler:
        await crawler.close()
        logger.info("Crawler resources have been released")
    logger.info("PPT-PDFGen-AI crawler service is down.")

async def crawl(request: CrawlRequest):
    """
    Crawls multiple URLs and processes the content

    Args:
    request: crawl request containing URLs and instructions

    Returns:
    Dict: dictionary containing processed content, number of successes and failed URLs

    Raises:
    HTTPException: throws an exception when an error occurs during crawling
    """
    global crawler
    return await crawler.crawl_urls(request.urls, request.instruction)


# Utility functions (converted from static methods)
def extract_valid_urls(results: List[Dict], limit: int = 10) -> List[str]:
    """Extract valid URLs from search results with limit."""
    urls = []
    for result in results[:limit]:
        if "url" in result and result["url"]:
            urls.append(result["url"])
    return urls

def validate_search_results(results: List[Dict]) -> None:
    """Validate search results and raise appropriate exceptions."""
    if not results:
        logger.warning("No search results found")
        raise HTTPException(status_code=404, detail="No search results found")

def validate_urls(urls: List[str]) -> None:
    """Validate extracted URLs and raise appropriate exceptions."""
    if not urls:
        logger.warning("No valid URLs found")
        raise HTTPException(status_code=404, detail="No valid URLs found")

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra whitespace and replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename.strip())
    # Remove leading/trailing dots and underscores
    filename = filename.strip('._')
    # Limit length
    if len(filename) > 50:
        filename = filename[:50]
    # Ensure not empty
    if not filename:
        filename = "unnamed"
    return filename

# Unused
def check_groq_connection() -> Dict:
    """Check connectivity with Groq API by listing available models."""
    api_key = Config.GROQ_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq connection failed: {str(e)}")

def process_with_rag(request_data: Dict) -> Dict:
    """Process content using RAG pipeline."""
    try:
        rag = RAGRunner()
        result = rag.run(request_data)
        if not result:
            raise HTTPException(
                status_code=500, 
                detail="RAG pipeline failed to generate content"
            )
        return result
    except Exception as e:
        logger.error(f"RAG processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG processing error: {str(e)}")

class FileService:
    """Service class for file operations that actually need shared state/config."""
    
    SUPPORTED_TYPES = {"pdf", "ppt"}
    MEDIA_TYPES = {
        "pdf": "application/pdf",
        "ppt": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    }
    
    # File storage configuration - relative path that works in most environments
    OUTPUT_DIR = "output/generated_files"
    
    @classmethod
    def ensure_output_directory(cls):
        """Ensure output directory exists."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory ensured: {os.path.abspath(cls.OUTPUT_DIR)}")
    
    @classmethod
    def validate_file_type(cls, file_type: str) -> None:
        """Validate file type using class constants."""
        if file_type not in cls.SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"File type must be one of: {', '.join(cls.SUPPORTED_TYPES)}"
            )
    
    @classmethod
    def generate_filename(cls, topic: str, file_type: str) -> str:
        """Generate filename: topic_YYYYMMDD_HHMMSS.ext"""
        sanitized_topic = sanitize_filename(topic)
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{sanitized_topic}_{datetime_str}.{file_type}"
    
    @classmethod
    def get_full_file_path(cls, filename: str) -> str:
        """Get full file path in output directory."""
        return os.path.join(cls.OUTPUT_DIR, filename)
    
    @classmethod
    def generate_file(cls, generated_result: Dict, topic: str, file_type: str) -> str:
        """Generate file and return full file path."""
        try:
            cls.ensure_output_directory()
            
            # Generate filename with topic_datetime format
            filename = cls.generate_filename(topic, file_type)
            full_path = cls.get_full_file_path(filename)
            
            logger.info(f"Generating {file_type.upper()} file: {full_path}")
            
            if file_type == "pdf":
                logger.info("Generating PDF file...")
                # Pass the full path to the PDF generator
                output_file = generate_pdf(generated_result, output_path=full_path)
            elif file_type == "ppt":
                logger.info("Generating PPT file...")
                # Pass the full path to the PPT generator
                output_file = generate_ppt(generated_result, output_path=full_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Use the predetermined path instead of relying on generator return
            final_output_file = output_file if output_file else full_path
                
            # Ensure the file was created
            if not os.path.exists(final_output_file):
                raise Exception(f"File generation failed - file not created at {final_output_file}")
                
            logger.info(f"File generated successfully: {final_output_file}")
            return final_output_file
            
        except Exception as e:
            logger.error(f"Error generating {file_type.upper()} file: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate {file_type.upper()} file: {str(e)}"
            )
    
    @classmethod
    def get_file_info(cls, file_path: str) -> Dict[str, Any]:
        """Get file information for response."""
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
        file_stat = os.stat(file_path)
        file_name = os.path.basename(file_path)
        file_ext = Path(file_path).suffix.lstrip('.')
        
        return {
            "file_path": os.path.abspath(file_path),  # Return absolute path
            "filename": file_name,
            "file_size": file_stat.st_size,
            "file_type": file_ext,
            "media_type": cls.MEDIA_TYPES.get(file_ext, "application/octet-stream"),
            "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat()
        }
    
    @classmethod
    def create_file_response(cls, file_path: str) -> FileResponse:
        """Create file response for download."""
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        file_info = cls.get_file_info(file_path)
        
        return FileResponse(
            path=file_path,
            filename=file_info["filename"],
            media_type=file_info["media_type"],
            headers={
                "Content-Disposition": f"attachment; filename={file_info['filename']}",
                "Content-Length": str(file_info["file_size"])
            }
        )

async def search_and_crawl(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Perform web search and crawl results.
    
    Args:
        query: Search query string
        limit: Maximum number of URLs to process
        
    Returns:
        Dict containing search results and crawled content
        
    Raises:
        HTTPException: When search or crawling fails
    """
    logger.info(f"Starting search and crawl for: {query}")
    
    try:
        # Perform web search
        logger.info("Executing web search...")
        response = WebCrawler.make_searxng_request(query=query)
        
        # Validate and extract URLs
        results = response.get("results", [])
        validate_search_results(results)
        
        urls = extract_valid_urls(results, limit)
        validate_urls(urls)
        
        logger.info(f"Found {len(urls)} URLs, starting crawl process")
        
        # Crawl the URLs
        crawl_request = CrawlRequest(urls=urls, instruction=query)
        crawl_result = await crawl(crawl_request)
        
        # Validate crawl results
        if not crawl_result or not crawl_result.get("content"):
            raise HTTPException(
                status_code=404, 
                detail="No content could be extracted from search results"
            )
        
        logger.info("Search and crawl completed successfully")
        
        return {
            "results": results,
            "crawl_results": crawl_result,
            "content": crawl_result.get("content"),
            "success_count": crawl_result.get("success_count", 0),
            "failed_urls": crawl_result.get("failed_urls", []),
            "processed_urls": urls
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search and crawl error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")


def generate_content_with_rag(type: str, text_inject: str, query: str, topic: str) -> Dict[str, Any]:
    """
    Generate content using RAG pipeline.
    
    Args:
        text_content: Input text for RAG processing
        query: User query/prompt
        topic: Content topic
        type: Type of content to generate (pdf/ppt)
        
    Returns:
        Dict containing generated content
        
    Raises:
        HTTPException: When content generation fails
    """
    logger.info(f"Starting RAG content generation for topic: {topic}")
    
    try:
        # Validate content type if provided
        if type:
            FileService.validate_file_type(type)
        
        # Prepare RAG request
        rag_request_data = {
            "type": type,
            "text_inject": text_inject,
            "query": query,
            "topic": topic
        }
        
        if type:
            rag_request_data["type"] = type
        
        logger.info("Processing with RAG pipeline...")
        result = process_with_rag(rag_request_data)
        
        logger.info("RAG processing completed successfully")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")


def create_downloadable_file(generated_content: Dict, topic: str, file_type: str) -> Dict[str, Any]:
    """
    Create a downloadable file from generated content and return file info.
   
    Args:
        generated_content: Content generated by RAG
        topic: File topic for naming
        file_type: Type of file to generate (pdf/ppt)
       
    Returns:
        Dict containing file information and path
       
    Raises:
        HTTPException: When file creation fails
    """
    logger.info(f"Creating downloadable {file_type.upper()} file for topic: {topic}")
   
    try:
        # Validate file type
        FileService.validate_file_type(file_type)
       
        # Generate file
        logger.info(f"Generating {file_type.upper()} file...")
        output_file_path = FileService.generate_file(generated_content, topic, file_type)
       
        # Get file information
        file_info = FileService.get_file_info(output_file_path)
       
        logger.info(f"File creation completed: {output_file_path}")
       
        return {
            "success": True,
            "message": f"{file_type.upper()} file generated successfully",
            "file_info": file_info
        }
       
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File creation failed: {str(e)}")

# API Endpoints
@app.post("/search")
async def search_endpoint(request: SearchRequest, x_api_key: APIKey = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Search API endpoint that performs web search and crawling.

    Args:
        request: Search request containing query and configuration parameters

    Returns:
        Dict containing search results, crawled content, and metadata

    Raises:
        HTTPException: When search or crawling fails
    """
    logger.info(f"Received search request: {request.query}")
    
    try:
        result = await search_and_crawl(request.query, request.limit)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search endpoint failed: {str(e)}")


@app.post("/ask")
async def ask_endpoint(request: AskRequest, x_api_key: APIKey = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Generate content using RAG pipeline based on provided text and parameters.

    Processes input text using embeddings and LLM to generate either
    detailed articles or slide-style summaries.

    Args:
        request: Request containing text, type (pdf/ppt), topic, and query

    Returns:
        Dict containing the generated content result

    Raises:
        HTTPException: When content generation fails
    """
    logger.info(f"Received RAG request for topic: {request.topic}")
    
    try:
        # Extract content type if available
        type = getattr(request, 'type', None)
        
        result = generate_content_with_rag(
            type=request.type,
            text_inject=request.text_inject,
            query=request.query,
            topic=request.topic
        )
        
        return {"result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ask endpoint failed: {str(e)}")


@app.post("/generate")
async def generate_endpoint(request: GenerateRequest,x_api_key: APIKey = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Generate PDF or PPT files based on web search results.
    Returns file information instead of direct file download for Streamlit integration.
    
    Args:
        request: Generate request containing query, topic, type, and limit
        
    Returns:
        Dict containing file information and download path
        
    Raises:
        HTTPException: When any step in the generation process fails
    """
    # Create random process ID for progress tracking
    logger.info(f"Starting generation process: {request.query} -> {request.type}")
   
    try:        
        # Step 1: Search and crawl web content
        search_result = await search_and_crawl(request.query, request.limit)
       
        # Step 2: Generate content using RAG
        generated_content = generate_content_with_rag(
            text_inject=search_result["content"],
            query=request.query,
            topic=request.topic,
            type=request.type
        )
       
        # Step 3: Create downloadable file
        file_result = create_downloadable_file(
            generated_content=generated_content,
            topic=request.topic,
            file_type=request.type
        )
       
        # Extract filename from the generated file path
        generated_filename = os.path.basename(file_result["file_info"]["file_path"])
       
        # Complete the process
        result = {
            "success": True,
            "message": f"{request.type.upper()} file generated successfully",
            "file_info": {
                "file_path": file_result["file_info"]["file_path"],
                "filename": generated_filename,
                "file_size": file_result["file_info"]["file_size"],
                "file_type": request.type,
                "media_type": file_result["file_info"]["media_type"],
                "created_at": file_result["file_info"]["created_at"]
            },
            "download_url": f"/download/{generated_filename}"
        }
       
        logger.info(f"Generation process completed successfully")
       
        return result
       
    except HTTPException as e:
        raise
    except Exception as e:
        error_msg = f"Generation process failed: {str(e)}"
        logger.error(f"Generate endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/download/{filename}")
async def download_file(filename: str, x_api_key: APIKey = Depends(get_api_key)) -> FileResponse:
    """
    Download a generated file.
   
    Args:
        filename: Name of the file to download
       
    Returns:
        FileResponse for file download
    """
    file_path = FileService.get_full_file_path(filename)
   
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
   
    return FileService.create_file_response(file_path)

if __name__ == "__main__":
    # Program entry point
    logger.info("Start the PPT-PDFGen-AI service through the command line")
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)