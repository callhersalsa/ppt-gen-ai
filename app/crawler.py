"""
This module is responsible for setting up and executing web crawling tasks using
SearXNG and Crawl4AI.

Purpose:
- Integrates the SearXNG metasearch engine to collect search results from multiple sources.
- Uses Crawl4AI to crawl and extract content from URLs found via SearXNG.
- Cleans the further crawled text using BeautifulSoup.
"""

from typing import List, Dict, Any, Optional, Tuple
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    CrawlerMonitor,
    DisplayMode
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import markdown
from bs4 import BeautifulSoup
import re
import http.client
from codecs import encode
import json
from fastapi import HTTPException
from urllib.parse import urlencode

# Import the configuration and logging modules
from shared.config import Config
import shared.logger as logger

# Define the WebCrawler class for web crawling and content processing
class WebCrawler:
    """Web crawler class, which encapsulates the functions of web crawling and content processing"""

    def __init__(self):
        """Initialize the crawler instance"""
        self.crawler = None
        logger.info("Initialize the WebCrawler instance")

    async def initialize(self) -> None:
        """Initialize an AsyncWebCrawler instance

        This method must be called before using the crawler
        """
        # Configure the browser
        browser_config = BrowserConfig(
                headless=True,
                verbose=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        # Initialize the crawler
        self.crawler = await AsyncWebCrawler(config=browser_config).__aenter__()
        logger.info("AsyncWebCrawler initialization completed")

    async def close(self) -> None:
        """Close the crawler instance to release resources"""
        if self.crawler:
            await self.crawler.__aexit__(None, None, None)
            logger.info("AsyncWebCrawler is closed")

    @staticmethod
    def markdown_to_text_regex(markdown_str: str) -> str:
        """Convert Markdown text to plain text using regular expressions

        Args:
            markdown_str: Markdown formatted text

        Returns:
            str: Converted plain text
        """
        # Remove title symbol
        text = re.sub(r'#+\s*', '', markdown_str)

        # Remove links and images
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Remove bold, italics, and other emphasis symbols
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)

        # Remove list symbols
        text = re.sub(r'^[\*\-\+]\s*', '', text, flags=re.MULTILINE)

        # Remove code block
        text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.*?)`', r'\1', text)

        # Remove blockquote
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        
        # Change newline with the real newline
        text = text.replace("\\n", "\n")
        
        # Add space after -
        text = re.sub(r'\.(\w)', r'. \1', text)
        
        # Add new line for digit format
        text = re.sub(r'(?<=\n)(\d+\.)', r'\n\1', text)
        
        # Add new line before heading/worrd if no digit
        text = re.sub(r'(?<!\d)(?<=\n)([A-Z][^\n]+\([A-Z][a-z]+\))', r'\n\1', text)

        # Delete unnecessary comma in the first
        text = re.sub(r'^,+', '', text)

        return text.strip()

    @staticmethod
    def markdown_to_text(markdown_str: str) -> str:
        """Convert Markdown text to plain text using markdown and BeautifulSoup libraries

        Args:
            markdown_str: Markdown formatted text

        Returns:
            str: Converted plain text
        """
        html = markdown.markdown(markdown_str, extensions=['fenced_code'])
        
        # Extracting plain text with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        text = soup.get_text(separator="\n")  # Preserve paragraph wrapping

        # Clean up extra blank lines
        cleaned_text = "\n".join([line.strip()
                             for line in text.split("\n") if line.strip()])

        return cleaned_text

    @staticmethod
    def make_searxng_request(query: str, 
                            language: str = 'id', 
                            limit: int = 5,
                            disabled_engines: str = Config.DISABLED_ENGINES,
                            enabled_engines: str = Config.ENABLED_ENGINES) -> dict:
        """Send a search request to SearXNG

        Args:
            query: Search query string
            limit: Limit the number of results returned
            disabled_engines: Comma separated list of search engines to disable
            enabled_engines: Comma separated list of enabled search engines

        Returns:
            dict: Search results returned by SearXNG

        Raises:
            Exception: Throws an exception when the request fails
        """
        try:
            conn = http.client.HTTPConnection(Config.SEARXNG_HOST, Config.SEARXNG_PORT)

            # Proper form-encoded data
            form_data = {
                'q': query,
                'format': 'json',
                'language': language,
                'time_range': 'week',
                'safesearch': '2',
                'pageno': '1',
                'category_general': '1',
                'limit': str(limit)
            }
            body = urlencode(form_data)

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'PPTGen-AI/1.0.0',
                'Accept': 'application/json',
                'Cookie': f'disabled_engines={disabled_engines};enabled_engines={enabled_engines};method=POST',
            }

            logger.info(f"Send a search request to SearXNG: {query}")
            conn.request("POST", Config.SEARXNG_BASE_PATH, body, headers)
            res = conn.getresponse()
            data = res.read()
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            logger.error(f"SearXNG request failed: {str(e)}")
            raise Exception(f"Search request failed: {str(e)}")

    async def crawl_urls(self, urls: List[str], instruction: str) -> Dict[str, Any]:
        """Crawl multiple URLs and process the content

        Args:
            urls: List of URLs to crawl
            instruction: Crawling instructions, usually search queries

        Returns:
            Dict[str, Any]: A dictionary containing the processed content, number of successes, and failed URLs

        Raises:
            HTTPException: Throws an exception when all URL crawling fails
        """
        try:
            # Check if the crawler is initialized
            if not self.crawler:
                logger.warning("The crawler is not initialized and is automatically initialized")
                await self.initialize()

            # Configuring the Markdown Generator
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=Config.CONTENT_FILTER_THRESHOLD),
                options={
                    "ignore_links": True,
                    "ignore_images": True,
                    "escape_html": False,
                }
            )

            # Configure crawler running parameters
            run_config = CrawlerRunConfig(
                word_count_threshold=Config.WORD_COUNT_THRESHOLD,
                exclude_external_links=True,    # Remove external links
                remove_overlay_elements=True,   # Remove pop-up/modal boxes
                excluded_tags=['img', 'header', 'footer', 'iframe', 'nav'],      # Exclude image tags
                process_iframes=True,           # Handling iframes
                markdown_generator=md_generator, # Use the configured Markdown generator
                cache_mode=CacheMode.BYPASS,     # No cache
                exclude_social_media_links=True,  # Exclude all known social media domains
                exclude_social_media_domains=Config.SOCIAL_MEDIA_DOMAINS,
            )

            logger.info(f"Starting to crawl URLs: {', '.join(urls)}")
            results = await self.crawler.arun_many(urls=urls, config=run_config)

            # Create a list to store all successful URLs crawled
            all_results = []
            failed_urls = []
            retry_urls = []

            # First crawl process
            for i, result in enumerate(results):
                try:
                    if result is None:
                        logger.debug(f"The URL crawling result is None: {urls[i]}")
                        retry_urls.append(urls[i])
                        continue

                    if not hasattr(result, 'success'):
                        logger.debug(f"URL crawling results lack success attribute: {urls[i]}")
                        retry_urls.append(urls[i])
                        continue

                    if result.success:
                        if not hasattr(result, 'markdown') or not hasattr(result.markdown, 'fit_markdown'):
                            logger.debug(f"URL crawling results lack markdown content: {urls[i]}")
                            retry_urls.append(urls[i])
                            continue

                        # Add the markdown content of successful results to the list
                        result_with_source = result.markdown.fit_markdown + '\n\n'
                        all_results.append(result_with_source)
                        logger.info(f"Successfully crawled URL: {urls[i]}")
                    else:
                        logger.debug(f"Failed to crawl URL: {urls[i]}")
                        retry_urls.append(urls[i])
                except Exception as e:
                    # Record the URL that needs to be retried
                    retry_urls.append(urls[i])
                    error_msg = str(e)
                    logger.warning(f"First crawl attempt failed for URL: {urls[i]}, Error message: {error_msg}")

            # If there are URLs that need to be retried, perform a second crawl
            if retry_urls:
                logger.info(f"Retrying failed URLs: {', '.join(retry_urls)}")
                retry_results = await self.crawler.arun_many(urls=retry_urls, config=run_config)

                for i, result in enumerate(retry_results):
                    try:
                        if result is None:
                            logger.debug(f"Retry crawl result is None for URL: {retry_urls[i]}")
                            failed_urls.append(retry_urls[i])
                            continue

                        if not hasattr(result, 'success'):
                            logger.debug(f"Retry crawl result is missing 'success' attribute: {retry_urls[i]}")
                            failed_urls.append(retry_urls[i])
                            continue

                        if result.success:
                            if not hasattr(result, 'markdown') or not hasattr(result.markdown, 'fit_markdown'):
                                logger.debug(f"Retry crawl result is missing markdown content: {retry_urls[i]}")
                                failed_urls.append(retry_urls[i])
                                continue

                            # Add the successfully retried result to the list
                            result_with_source = result.markdown.fit_markdown + '\n\n'
                            all_results.append(result_with_source)
                            logger.info(f"Successfully crawled URL on retry: {retry_urls[i]}")
                        else:
                            logger.debug(f"Retry crawl still failed for URL: {retry_urls[i]}")
                            failed_urls.append(retry_urls[i])
                    except Exception as e:
                        # Record the final failed URL
                        failed_urls.append(retry_urls[i])
                        error_msg = str(e)
                        logger.error(f"Second crawl attempt failed for URL: {retry_urls[i]}, Error message: {error_msg}")

            if not all_results:
                logger.error("All URL crawls failed")
                raise HTTPException(status_code=500, detail="All URL crawls failed")

            # Join all successful results into one complete string using a separator
            combined_content = '\n\n==========\n\n'.join(all_results)

            # Convert to plain text
            plain_text = self.markdown_to_text_regex(self.markdown_to_text(combined_content))

            response = {
                "content": plain_text,
                "success_count": len(all_results),
                "failed_urls": failed_urls
            }

            logger.info(f"Crawling completed, Success: {len(all_results)}, Failed: {len(failed_urls)}")
            return response
        
        except Exception as e:
            logger.error(f"An exception occurred during the crawling process: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
