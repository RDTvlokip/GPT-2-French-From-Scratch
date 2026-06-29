import os
import re
import requests
import time
import logging
import sys
import html
import threading
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import Counter

CONFIG = {
    "data_dir": "data",
    "index_file_name": "downloaded_index.txt",
    "api_url": "https://fr.wikipedia.org/w/api.php",
    "user_agent": "UltraAdvancedWikipediaDownloader/3.2-EnhancedCleaner (rdtvlokip@gmail.com)",
    "request_timeout": 30,
    "retry_strategy": Retry(
        total=7,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 524],
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False
    ),
    "min_content_length": 250,
    "batch_size_random": 500,
    "batch_size_search": 50,
    "batch_size_category_members": 50,
    "batch_size_allcategories": 50,
    "api_sleep_interval": 0.5,
    "error_sleep_interval": 5,
    "max_workers": 8,
    "log_level": logging.INFO,
    "log_format": '%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s',
    "log_file": "wikipedia_downloader_enhanced.log",
    "template_removal_iterations": 5
}

logging.basicConfig(
    level=CONFIG["log_level"],
    format=CONFIG["log_format"],
    handlers=[
        logging.FileHandler(CONFIG["log_file"], encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class UltraEnhancedTextCleaner:
    def __init__(self, config):
        self.config = config
        self.initial_block_removal_patterns = self._compile_initial_block_removal_patterns()
        self.content_extraction_patterns = self._compile_content_extraction_patterns()
        self.markup_simplification_patterns = self._compile_markup_simplification_patterns()
        self.section_removal_patterns = self._compile_section_removal_patterns()
        self.final_formatting_patterns = self._compile_final_formatting_patterns()

    def _compile_initial_block_removal_patterns(self):
        patterns = [
            # (re.compile(r'', re.DOTALL), ''), # Ligne problématique supprimée
            (re.compile(r'<gallery.*?>.*?</gallery>', re.DOTALL | re.IGNORECASE), '\n'),
            (re.compile(r'\{\|(?:[^{}]*|\{\{[^{}]*\}\}|[^}])*\|\}', re.DOTALL | re.IGNORECASE), '\n'),
            (re.compile(r'<table.*?>.*?</table>', re.DOTALL | re.IGNORECASE), '\n'),
            (re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL | re.IGNORECASE), ''),
            (re.compile(r'<ref[^/]*/>', re.IGNORECASE), ''),
            (re.compile(r'<references.*?>.*?</references>', re.DOTALL | re.IGNORECASE), ''),
            (re.compile(r'\[\[(?:Fichier|File|Image|Vidéo|Media):.*?\]\]', re.IGNORECASE | re.DOTALL), ''),
            (re.compile(r'\[\[Catégorie:[^\]]+\]\]', re.IGNORECASE), ''),
            (re.compile(r'\{\{(?:DEFAULTSORT|Autorité|Authority control|Contrôle d\'autorité|Suivi des pages liées|Article détaillé|Loupe|Traduction/Référence|Clear|Clr|Colonnes|Début de colonnes|Fin de colonnes|Sommaire)(?:[^{}]*|\{\{[^{}]*\}\})*\}\}', re.IGNORECASE | re.DOTALL), ''),
            (re.compile(r'\{\{(?:Infobox|Taxobox|Boîte|Ficha|Encadré)(?:[^{}]*|\{\{[^{}]*\}\}[^{}]*)*\}\}', re.IGNORECASE | re.DOTALL), '\n'),
            (re.compile(r'\{\{(?:Palette|Navbox|Navigation|Bas de page)(?:[^{}]*|\{\{[^{}]*\}\})*\}\}', re.IGNORECASE | re.DOTALL), '\n'),
            (re.compile(r'\{\{(?:Ébauche|Stub|Sources|Travaux inédits|Style à revoir|À recycler|À vérifier)(?:[^{}]*|\{\{[^{}]*\}\})*\}\}', re.IGNORECASE | re.DOTALL), ''),
            (re.compile(r'\{\{(?:Coord|Coordinate|Location)(?:[^{}]*|\{\{[^{}]*\}\})*\}\}', re.IGNORECASE | re.DOTALL), ''),
            (re.compile(r'<math>.*?</math>', re.DOTALL | re.IGNORECASE), '[FORMULE MATH]'),
            (re.compile(r'<syntaxhighlight.*?>.*?</syntaxhighlight>', re.DOTALL | re.IGNORECASE), '[CODE]'),
            (re.compile(r'<timeline.*?>.*?</timeline>', re.DOTALL | re.IGNORECASE), '[TIMELINE]'),
            (re.compile(r'__NOTOC__|__FORCETOC__|__TOC__', re.IGNORECASE), ''),
            (re.compile(r'\{\{(?:sfn|harvnb|harvsp|citation|ref|rp)(?:\|[^{}]*?)?\}\}', re.IGNORECASE | re.DOTALL), ''),
            (re.compile(r'\{\{(?:[Cc]itation needed|[Cc]n|[Ff]act|[Rr]efnec|[Rr]éférence nécessaire|[Rr]éfnec|[Rr]éférence souhaitée|[Ss]ourcer|[Dd]emande de source|[Pp]assage non neutre|[Pp]assage évasif)(?:\|[^{}]*)?\}\}', re.IGNORECASE | re.DOTALL), ''),
            (re.compile(r'\{\{!--.*?--\}\}', re.DOTALL), ''),
        ]
        return patterns

    def _compile_content_extraction_patterns(self):
        patterns = [
            (re.compile(r'\{\{(?:lang(?:ue)?)\|[a-z]{2,3}(?:-[a-zA-Z0-9]+)*\|([^}]+)\}\}', re.IGNORECASE), r'\1'),
            (re.compile(r'\{\{(?:nobr|nowrap)\|([^}]+)\}\}', re.IGNORECASE), r'\1'),
            (re.compile(r"\{\{'\}\}", re.IGNORECASE), r"'"),
            (re.compile(r'\{\{(?:unité|nombre)\|([^|}]+)\|([^|}]+)(?:\|[^}]*)?\}\}', re.IGNORECASE), r'\1 \2'),
            (re.compile(r'\{\{etc\.\}\}', re.IGNORECASE), 'etc.'),
            (re.compile(r'\{\{(?:[Dd]ate|[Dd]ate de naissance|[Dd]ate de décès|[Dd]ate sport|dts)(?:\|[^}]*)*?\|([0-9]{1,4})\|([a-zA-Z]+|[0-9]{1,2})\|([0-9]{1,4})(?:\|[^}]*)*\}\}', re.IGNORECASE), r'\1 \2 \3'),
            (re.compile(r'\{\{(?:[Dd]ate|[Dd]ate de naissance|[Dd]ate de décès|[Dd]ate sport|dts)(?:\|[^}]*)*?\|([0-9]{1,4})\|([0-9]{1,2})(?:\|[^}]*)*\}\}', re.IGNORECASE), r'\1/\2'), # pour MM/AAAA ou JJ/MM
            (re.compile(r'\{\{(?:[Dd]ate|[Dd]ate de naissance|[Dd]ate de décès|[Dd]ate sport|dts)(?:\|[^}]*)*?\|([0-9]{4})(?:\|[^}]*)*\}\}', re.IGNORECASE), r'\1'), # pour AAAA
        ]
        return patterns

    def _compile_markup_simplification_patterns(self):
        patterns = [
            (re.compile(r'\[\[(?:[^|\]]*:)?(?:[^|\]]+\|)?([^\]]+)\]\]'), r'\1'),
            (re.compile(r"'{5}(.*?)'{5}"), r'\1'),
            (re.compile(r"'{2,3}(.*?)'{2,3}"), r'\1'),
            (re.compile(r'\[https?://\S+\s+([^\]]+?)\]'), r'\1'),
            (re.compile(r'\[https?://\S+\]'), ''),
            (re.compile(r'https?://\S+'), ''),
            (re.compile(r'<[a-zA-Z/][^>]*>'), ''),
        ]
        return patterns

    def _compile_section_removal_patterns(self):
        section_keywords = [
            "Notes et références", "Références", "Notes",
            "Voir aussi", "Articles connexes", "Liens internes",
            "Annexes", "Appendices",
            "Bibliographie", "Lectures complémentaires", "Ouvrages de référence",
            "Liens externes",
            "Sources", "Sources et références",
            "Palettes de navigation", "Palettes",
            "Portail", "Portails", "Discographie", "Filmographie", "Ludographie"
        ]
        patterns = []
        for kw in section_keywords:
            pattern_str = rf"(?is)(^\s*={'{2,6}'}\s*{re.escape(kw)}\s*={'{2,6}'}\s*$.*?(?=\n^\s*={'{2,6}'}|\Z))"
            patterns.append((re.compile(pattern_str, re.MULTILINE | re.DOTALL), ''))
        
        patterns.append((re.compile(r'^\s*\{\{(?:Portail|Portal)(?:[^{}]*|\{\{[^{}]*\}\})*\}\}\s*$', re.MULTILINE | re.IGNORECASE), ''))
        patterns.append((re.compile(r'^\s*Portail (?:de |du |des |de l\'|d\')[^\n]+', re.MULTILINE | re.IGNORECASE), ''))
        patterns.append((re.compile(r"Cet article est partiellement ou en totalité issu de l'article intitulé\s*«[^»]+»\s*\(voir la liste des auteurs(?: et mentionner la licence)?\)\.?", re.IGNORECASE), ''))
        return patterns

    def _compile_final_formatting_patterns(self):
        return [
            (re.compile(r'^[ \t]*={2,6}[ \t]*(.*?)[ \t]*={2,6}[ \t]*\n?', re.MULTILINE), r'\n\n== \1 ==\n'),
            (re.compile(r'^\s*[*#;:]+\s*(.*)', re.MULTILINE), r'\1'),
            (re.compile(r'\r\n|\r'), '\n'),
            (re.compile(r'[ \t]{2,}'), ' '),
            (re.compile(r'^\s+', re.MULTILINE), ''),
            (re.compile(r'\s+\n', re.MULTILINE), '\n'),
            (re.compile(r'\n{3,}'), '\n\n'),
            (re.compile(r'\s+([,.!?;:])'), r'\1'),
            (re.compile(r'([({\[])\s+'), r'\1'),
            (re.compile(r'\s+([)}\]])'), r'\1'),
            (re.compile(r'\(\s*\)'), ''),
            (re.compile(r'\[\s*\]'), ''),
            (re.compile(r'\{\s*\}'), ''),
        ]

    def clean(self, text):
        if not isinstance(text, str):
            logger.warning("Received non-string input for cleaning.")
            return ""

        cleaned_text = text
        try:
            cleaned_text = html.unescape(cleaned_text)
            cleaned_text = cleaned_text.replace('\u00A0', ' ')

            for pattern, replacement in self.initial_block_removal_patterns:
                cleaned_text = pattern.sub(replacement, cleaned_text)
            
            for pattern, replacement in self.content_extraction_patterns:
                 cleaned_text = pattern.sub(replacement, cleaned_text)

            iterations = self.config.get("template_removal_iterations", 5)
            for _ in range(iterations):
                 simple_template_pattern = re.compile(r'\{\{(?:[^{}]*?|\{[^{}]*?\})*?\}\}')
                 prev_text_len = len(cleaned_text)
                 cleaned_text = simple_template_pattern.sub('', cleaned_text)
                 if len(cleaned_text) == prev_text_len:
                      break
            
            for pattern, replacement in self.markup_simplification_patterns:
                cleaned_text = pattern.sub(replacement, cleaned_text)

            for pattern, replacement in self.section_removal_patterns:
                cleaned_text = pattern.sub(replacement, cleaned_text)

            for pattern, replacement in self.final_formatting_patterns:
                cleaned_text = pattern.sub(replacement, cleaned_text)
            
            cleaned_text = re.sub(r'(^\s*==\s*[^=]*?\s*==\s*\n)(?=\s*==|\s*$)', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
            cleaned_text = cleaned_text.strip()

        except Exception as e:
            logger.error(f"Error during text cleaning: {e}", exc_info=True)
            return None
        return cleaned_text

class WikipediaClient:
    def __init__(self, config):
        self.config = config
        self.session = self._create_session()
        self.cleaner = UltraEnhancedTextCleaner(config)
        self.downloaded_in_run = set()
        self.download_lock = threading.Lock()

        self.index_file_path = os.path.join(self.config['data_dir'], self.config['index_file_name'])
        self.downloaded_index_set = set()
        self.index_file_handle = None
        self.index_write_lock = threading.Lock()
        try:
            self._load_downloaded_index()
            self._open_index_for_append()
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


    def _load_downloaded_index(self):
        if os.path.exists(self.index_file_path):
            try:
                with open(self.index_file_path, 'r', encoding='utf-8') as f_index:
                    self.downloaded_index_set = {line.strip() for line in f_index if line.strip()}
                logger.info(f"Loaded {len(self.downloaded_index_set)} titles from index file: {self.index_file_path}")
            except IOError as e:
                logger.error(f"Error reading index file {self.index_file_path}: {e}. Proceeding without index.")
                self.downloaded_index_set = set()
        else:
            logger.info(f"Index file {self.index_file_path} not found. Assuming first run or index needs creation.")
            self.downloaded_index_set = set()

    def _open_index_for_append(self):
        try:
            os.makedirs(os.path.dirname(self.index_file_path), exist_ok=True)
            self.index_file_handle = open(self.index_file_path, 'a+', encoding='utf-8')
            logger.debug(f"Index file {self.index_file_path} opened for appending.")
        except IOError as e:
            logger.error(f"FATAL: Could not open index file {self.index_file_path} for appending: {e}. Indexing disabled.", exc_info=True)
            self.index_file_handle = None

    def close(self):
        if self.index_file_handle:
            try:
                self.index_file_handle.close()
                logger.info(f"Closed index file: {self.index_file_path}")
            except IOError as e:
                logger.error(f"Error closing index file {self.index_file_path}: {e}")
        self.index_file_handle = None


    def _create_session(self):
        session = requests.Session()
        session.headers.update({'User-Agent': self.config['user_agent']})
        adapter = HTTPAdapter(max_retries=self.config['retry_strategy'])
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _make_api_request(self, params, method='GET'):
        params['format'] = 'json'
        params['formatversion'] = 2
        params['errorformat'] = 'plaintext'
        params['uselang'] = 'fr'
        params['maxlag'] = '5'

        response = None
        try:
            response = self.session.request(
                method,
                self.config['api_url'],
                params=params if method == 'GET' else None,
                data=params if method == 'POST' else None,
                timeout=self.config['request_timeout']
            )

            if response.status_code == 429:
                 retry_after = response.headers.get("Retry-After")
                 sleep_time = int(retry_after) if retry_after and retry_after.isdigit() else self.config['error_sleep_interval']
                 logger.warning(f"Rate limited (429). Retrying after {sleep_time} seconds.")
                 time.sleep(sleep_time)
                 return None

            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                error_info = data['error']
                code = error_info.get('code', 'N/A')
                info = error_info.get('text', 'Unknown API error')
                logger.error(f"API Error: Code='{code}', Info='{info}' Params: {params.get('titles', params.get('srsearch', 'N/A'))}")
                return None
            if 'warnings' in data:
                warnings = data['warnings']
                if isinstance(warnings, dict):
                    for warning_type, warning_info in warnings.items():
                        warning_text = warning_info.get('text', 'Details unavailable') if isinstance(warning_info, dict) else str(warning_info)
                        logger.warning(f"API Warning ({warning_type}): {warning_text}")
                elif isinstance(warnings, list):
                    for warning_obj in warnings:
                        if isinstance(warning_obj, dict):
                            warning_type = warning_obj.get('module', 'unknown')
                            warning_text = warning_obj.get('text', str(warning_obj))
                        else:
                            warning_type = 'unknown'
                            warning_text = str(warning_obj)
                        logger.warning(f"API Warning ({warning_type}): {warning_text}")
                else:
                    logger.warning(f"API Warning: {warnings}")

            return data
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            url = e.response.url
            if status == 404:
                 logger.warning(f"HTTP 404 Not Found for URL: {url}")
            else:
                 logger.error(f"HTTP Error {status} for URL: {url}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection Error: {e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Request Timeout: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"General Request Error: {e}")
        except ValueError as e:
             logger.error(f"Failed to decode JSON response: {e} - Response text: {response.text[:200] if response else 'No response'}")
        return None

    def _safe_filename(self, title):
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', title)
        filename = filename.replace(' ', '_')
        filename = re.sub(r'[\._]+$', '', filename)
        filename = re.sub(r'^[._]+', '', filename)
        filename = re.sub(r'_+', '_', filename)
        if not filename:
            filename = "_invalid_title_"
        max_len = 220

        try:
            byte_len = len(filename.encode('utf-8'))
            while byte_len > max_len:
                original_len = len(filename)
                filename = filename[:-1]
                if len(filename) == original_len:
                     break
                byte_len = len(filename.encode('utf-8'))
        except Exception as e:
            logger.warning(f"Could not accurately shorten filename '{title}' due to encoding issue: {e}. Using simple truncation.")
            filename = filename[:max_len // 2]

        return filename.strip('_') or "_fallback_name_"


    def _fetch_paginated_list(self, base_params, list_key, result_processor, limit, item_name="items"):
        all_items = []
        retrieved_ids = set()
        continue_params = {}
        attempts = 0
        max_attempts_per_page = 3
        
        api_limit_param_map = {
            'search': 'srlimit',
            'allcategories': 'aclimit',
            'categorymembers': 'cmlimit'
        }
        
        config_batch_size_key_map = {
            'search': 'batch_size_search',
            'allcategories': 'batch_size_allcategories',
            'categorymembers': 'batch_size_category_members'
        }

        if list_key not in api_limit_param_map or list_key not in config_batch_size_key_map:
            logger.error(f"Unsupported list_key '{list_key}' in _fetch_paginated_list.")
            return []

        api_limit_param = api_limit_param_map[list_key]
        config_batch_size_key = config_batch_size_key_map[list_key]

        logger.info(f"Starting fetch for {item_name} (limit {limit}).")
        while len(retrieved_ids) < limit:
            current_page_attempts = 0
            
            batch_size = min(self.config[config_batch_size_key], limit - len(retrieved_ids), 500)

            if batch_size <= 0: break

            params = {**base_params, **continue_params}
            params[api_limit_param] = batch_size
            
            while current_page_attempts < max_attempts_per_page:
                attempts += 1
                logger.debug(f"Requesting {batch_size} {item_name} (attempt {attempts}, page attempt {current_page_attempts + 1})...")
                data = self._make_api_request(params)

                if data and 'query' in data and list_key in data['query']:
                    results = data['query'][list_key]
                    if not results:
                        logger.info(f"No more {item_name} found.")
                        return all_items[:limit]

                    new_count = 0
                    for item in results:
                        processed = result_processor(item)
                        if processed and 'id' in processed and processed['id'] is not None and processed['id'] not in retrieved_ids:
                            all_items.append(processed)
                            retrieved_ids.add(processed['id'])
                            new_count += 1
                        elif processed and 'id' not in processed:
                            logger.warning(f"Processed item for {item_name} lacks an 'id' field: {processed}")

                    logger.info(f"Retrieved {new_count} new unique {item_name}. Total unique: {len(retrieved_ids)}/{limit}")

                    if 'continue' in data:
                        continue_params = {k: v for k, v in data['continue'].items() if k.endswith('continue')}
                        if not continue_params:
                            logger.info(f"Empty 'continue' block found. Fetching complete for {item_name}.")
                            return all_items[:limit]
                    else:
                        logger.info(f"No 'continue' marker found. Fetching complete for {item_name}.")
                        return all_items[:limit]

                    time.sleep(self.config['api_sleep_interval'])
                    break
                else:
                    current_page_attempts += 1
                    logger.warning(f"Failed to retrieve {item_name} batch (page attempt {current_page_attempts}/{max_attempts_per_page}). Pausing.")
                    if current_page_attempts >= max_attempts_per_page:
                        logger.error(f"Max attempts reached for fetching a page of {item_name}. Aborting fetch.")
                        return all_items[:limit]
                    time.sleep(self.config['error_sleep_interval'])

        logger.info(f"Fetch completed for {item_name}. Found {len(retrieved_ids)} unique items.")
        return all_items[:limit]


    def get_random_articles(self, count):
        all_articles = []
        retrieved_ids = set()
        attempts = 0
        max_total_attempts = count * 3 + 10 

        logger.info(f"Attempting to retrieve {count} unique random articles.")
        while len(retrieved_ids) < count and attempts < max_total_attempts:
            attempts += 1
            needed = count - len(retrieved_ids)
            batch_size = min(self.config['batch_size_random'], needed, 500)

            logger.debug(f"Requesting {batch_size} random articles (attempt {attempts}).")
            params = {"action": "query", "list": "random", "rnlimit": batch_size, "rnnamespace": 0}
            data = self._make_api_request(params)

            if data and 'query' in data and 'random' in data['query']:
                found_articles = data['query']['random']
                new_count = 0
                for article in found_articles:
                    if 'id' in article and 'title' in article and article['id'] not in retrieved_ids:
                        all_articles.append({'id': article['id'], 'title': article['title']})
                        retrieved_ids.add(article['id'])
                        new_count += 1
                if new_count > 0:
                     logger.info(f"Retrieved {new_count} new unique articles. Total unique: {len(retrieved_ids)}/{count}")
                else:
                     logger.debug("No new unique articles in this random batch.")

                if len(retrieved_ids) < count:
                    time.sleep(self.config['api_sleep_interval'])
            else:
                logger.warning("Failed to retrieve random articles batch or empty response. Pausing.")
                time.sleep(self.config['error_sleep_interval'])

        if len(retrieved_ids) < count:
             logger.warning(f"Could only retrieve {len(retrieved_ids)} out of {count} requested random articles after {attempts} attempts.")

        return all_articles


    def get_article_content(self, title):
        logger.debug(f"Fetching wikitext for '{title}'")
        params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "redirects": True,
        }
        data = self._make_api_request(params)

        if data and 'parse' in data:
            parse = data['parse']
            final_title = parse.get('title', title)
            wikitext = parse.get('wikitext')
            if isinstance(wikitext, dict):
                wikitext = wikitext.get('*')
            if wikitext is None:
                logger.warning(f"No wikitext found for page '{final_title}'.")
                return None, final_title
            if title != final_title:
                logger.info(f"Title '{title}' redirected to '{final_title}'.")
            return wikitext, final_title

        logger.error(f"Failed to parse content response or find page data for '{title}'")
        return None, title


    def search_articles(self, query, limit):
        base_params = {"action": "query", "list": "search", "srsearch": query, "srnamespace": 0, "srprop": "pageid"}
        def process_search_result(item):
            if 'pageid' in item and 'title' in item:
                 return {'id': item['pageid'], 'title': item['title']}
            return None
        return self._fetch_paginated_list(base_params, 'search', process_search_result, limit, "search results")


    def get_categories(self, limit):
         base_params = {"action": "query", "list": "allcategories", "acprop": ""}
         def process_category_result(item):
             cat_name = item.get('category')
             if cat_name:
                 return {'id': hash(cat_name), 'title': cat_name}
             return None
         results = self._fetch_paginated_list(base_params, 'allcategories', process_category_result, limit, "categories")
         return [r['title'] for r in results]


    def get_articles_in_category(self, category, limit):
        category_title = f"Category:{category}"
        base_params = {"action": "query", "list": "categorymembers", "cmtitle": category_title, "cmtype": "page", "cmnamespace": 0, "cmprop": "ids|title"}
        def process_cm_result(item):
            if 'pageid' in item and 'title' in item:
                 return {'id': item['pageid'], 'title': item['title']}
            return None
        return self._fetch_paginated_list(base_params, 'categorymembers', process_cm_result, limit, f"articles in '{category}'")


    def download_single_article_task(self, original_title, data_dir):
        thread_name = threading.current_thread().name
        logger.debug(f"[{thread_name}] Starting download task for '{original_title}'")

        if original_title in self.downloaded_index_set:
            logger.info(f"[{thread_name}] Article '{original_title}' already in index file. Skipping.")
            return original_title, "skipped_indexed"

        with self.download_lock:
            if original_title in self.downloaded_in_run:
                logger.debug(f"[{thread_name}] Article '{original_title}' already processed/processing in this run. Skipping.")
                return original_title, "skipped_session"
            self.downloaded_in_run.add(original_title)

        logger.info(f"[{thread_name}] Downloading article '{original_title}'...")
        content, final_title = self.get_article_content(original_title)

        if original_title != final_title and final_title in self.downloaded_index_set:
             logger.info(f"[{thread_name}] Article '{original_title}' redirected to '{final_title}', which is already in index. Skipping.")
             with self.download_lock:
                  self.downloaded_in_run.add(final_title)
             return original_title, "skipped_indexed_redirect"

        if content is None:
            logger.warning(f"[{thread_name}] No content retrieved for article '{final_title}' (from '{original_title}').")
            return original_title, "failed_no_content"
        if not content.strip():
            logger.warning(f"[{thread_name}] Content retrieved for '{final_title}' is empty or whitespace only.")
            return original_title, "failed_empty_content"

        cleaned_text = self.cleaner.clean(content)

        if cleaned_text is None:
            logger.warning(f"[{thread_name}] Cleaning failed for '{final_title}'. Skipping save.")
            return original_title, "failed_clean_error"

        if len(cleaned_text) < self.config['min_content_length']:
            logger.warning(f"[{thread_name}] Cleaned content for '{final_title}' is too short ({len(cleaned_text)} chars). Skipping save.")
            return original_title, "failed_too_short"

        safe_title = self._safe_filename(final_title)
        if not safe_title:
             logger.error(f"[{thread_name}] Could not generate a safe filename for final title '{final_title}'. Skipping.")
             return original_title, "failed_bad_filename"

        output_file = os.path.join(data_dir, f'{safe_title}.txt')

        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            logger.info(f"[{thread_name}] Article '{final_title}' (from '{original_title}') downloaded successfully to {output_file}")

            if self.index_file_handle:
                with self.index_write_lock:
                    if final_title not in self.downloaded_index_set:
                         self.index_file_handle.write(final_title + '\n')
                         self.index_file_handle.flush()
                         self.downloaded_index_set.add(final_title)
                         logger.debug(f"[{thread_name}] Added '{final_title}' to index file.")
                    else:
                         logger.debug(f"[{thread_name}] '{final_title}' was already added to index set concurrently, skipping duplicate write.")

            if original_title != final_title:
                 with self.download_lock:
                      self.downloaded_in_run.add(final_title)

            return original_title, "success"

        except IOError as e:
            logger.error(f"[{thread_name}] Failed to write article '{final_title}' to disk ({output_file}): {e}", exc_info=False)
            if os.path.exists(output_file):
                try: os.remove(output_file)
                except OSError as remove_e: logger.error(f"[{thread_name}] Failed to remove incomplete file {output_file}: {remove_e}")
            return original_title, "failed_write_error"
        except Exception as e:
            logger.error(f"[{thread_name}] An unexpected error occurred during file writing for '{final_title}': {e}", exc_info=True)
            return original_title, "failed_unexpected_write"


    def download_article_list_parallel(self, articles_list, data_dir):
        total_articles = len(articles_list)
        if total_articles == 0:
            logger.info("No articles provided to download.")
            return Counter()

        valid_articles = [a for a in articles_list if isinstance(a, dict) and 'title' in a]
        num_valid = len(valid_articles)
        if num_valid != total_articles:
             logger.warning(f"Filtered out {total_articles - num_valid} invalid entries from article list.")

        if num_valid == 0:
            logger.info("No valid articles with titles found in the list to process.")
            return Counter()

        logger.info(f"Starting parallel download process for {num_valid} articles using up to {self.config['max_workers']} workers.")
        results_counter = Counter()
        processed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers'], thread_name_prefix='WikiDownloader') as executor:
            future_to_title = {executor.submit(self.download_single_article_task, article['title'], data_dir): article['title']
                               for article in valid_articles}

            logger.info(f"Submitted {len(future_to_title)} download tasks.")

            for future in concurrent.futures.as_completed(future_to_title):
                original_title = future_to_title[future]
                processed_count += 1
                try:
                    _, status = future.result()
                    results_counter[status] += 1
                    logger.debug(f"Completed task for: {original_title} -> Status: {status} ({processed_count}/{len(future_to_title)})")

                except Exception as exc:
                    logger.error(f"Article '{original_title}' generated an exception during threaded execution: {exc}", exc_info=True)
                    results_counter['failed_exception'] += 1
                finally:
                     if processed_count % 50 == 0 or processed_count == len(future_to_title):
                          print(f"Progrès: {processed_count}/{len(future_to_title)} traités.", end='\r')
        print()
        logger.info("Parallel download process finished.")
        logger.info(f"Summary: {dict(results_counter)}")
        return results_counter


def get_positive_integer_input(prompt):
    while True:
        try:
            value_str = input(prompt).strip()
            if not value_str: raise ValueError("Input cannot be empty.")
            value = int(value_str)
            if value > 0:
                return value
            else:
                print("Veuillez entrer un nombre entier positif.")
        except ValueError as e:
            print(f"Entrée invalide: {e}. Veuillez entrer un nombre entier.")
        except EOFError:
             logger.warning("EOF received, assuming non-interactive mode. Aborting input.")
             raise

def main():
    start_time = time.monotonic()
    logger.info(f"--- Starting Wikipedia Downloader (Enhanced Cleaner v3.2) ---")
    exit_code = 0

    try:
        data_dir = CONFIG['data_dir']
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
                logger.info(f"Created data directory: {data_dir}")
            except OSError as e:
                logger.critical(f"FATAL: Failed to create data directory '{data_dir}': {e}")
                return 1

        with WikipediaClient(CONFIG) as client:
            exit_code = _run_interactive(client, data_dir)

    except KeyboardInterrupt:
        logger.warning("Opération interrompue par l'utilisateur (Ctrl+C).")
        print("\nOpération interrompue.")
        exit_code = 1
    except EOFError:
        logger.error("Fin de fichier atteinte pendant la saisie utilisateur (peut-être exécuté non interactivement?).")
        print("\nErreur de saisie (EOF).")
        exit_code = 1
    except Exception as e:
        logger.critical(f"Une erreur critique et inattendue est survenue dans le thread principal: {e}", exc_info=True)
        print(f"\nUne erreur critique est survenue. Vérifiez le fichier log: {CONFIG['log_file']}")
        exit_code = 1
    finally:
        end_time = time.monotonic()
        logger.info(f"Temps d'exécution total: {end_time - start_time:.2f} secondes.")
        logger.info(f"--- Fin du script Wikipedia Downloader ---")
    return exit_code


def _run_interactive(client, data_dir):
    print("\n--- Options ---")
    print("1. Télécharger des articles aléatoires")
    print("2. Rechercher des articles par mot-clé")
    print("3. Télécharger des articles par catégorie")
    print("---------------")

    while True:
        choice = input("Choisissez une option (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Choix invalide. Veuillez entrer 1, 2 ou 3.")

    final_results = Counter()
    articles_to_process = []

    if choice == '1':
        num_articles = get_positive_integer_input("Combien d'articles aléatoires uniques voulez-vous télécharger? ")
        articles_to_process = client.get_random_articles(num_articles)

    elif choice == '2':
        query = input("Entrez un terme de recherche: ").strip()
        if not query:
            logger.error("Search query cannot be empty.")
            return 1
        num_results = get_positive_integer_input("Combien de résultats maximum voulez-vous télécharger? ")
        articles_to_process = client.search_articles(query, num_results)

    elif choice == '3':
        num_categories_show = get_positive_integer_input("Combien de catégories voulez-vous afficher pour choisir? (ex: 30) ")
        categories = client.get_categories(num_categories_show)
        if not categories:
            logger.warning("Aucune catégorie trouvée ou erreur lors de la récupération. Impossible de continuer.")
            return 1

        print("\n--- Catégories Disponibles ---")
        for idx, cat in enumerate(categories, 1):
            print(f"{idx}. {cat}")
        print("-----------------------------")

        selected_category = None
        while selected_category is None:
            try:
                cat_choice_input = input(f"Choisissez une catégorie par numéro (1-{len(categories)}) ou entrez son nom exact: ").strip()
                if not cat_choice_input:
                    print("Saisie vide. Réessayez.")
                    continue
                try:
                    cat_choice_idx = int(cat_choice_input) - 1
                    if 0 <= cat_choice_idx < len(categories):
                        selected_category = categories[cat_choice_idx]
                    else:
                        print(f"Numéro invalide. Entrez un nombre entre 1 et {len(categories)}.")
                except ValueError:
                    if cat_choice_input in categories:
                        selected_category = cat_choice_input
                    else:
                        matches = [c for c in categories if c.lower() == cat_choice_input.lower()]
                        if len(matches) == 1:
                            selected_category = matches[0]
                        elif len(matches) > 1:
                            print(f"Plusieurs catégories correspondent à '{cat_choice_input}'. Utilisez la casse exacte ou le numéro.")
                        else:
                            print("Nom de catégorie non trouvé. Essayez la casse exacte ou le numéro.")
            except EOFError:
                logger.warning("EOF received during category selection.")
                raise

        logger.info(f"Catégorie choisie: '{selected_category}'")
        max_articles_cat = get_positive_integer_input(f"Combien d'articles maximum de la catégorie '{selected_category}' voulez-vous télécharger? ")
        articles_to_process = client.get_articles_in_category(selected_category, max_articles_cat)

    if articles_to_process:
        final_results = client.download_article_list_parallel(articles_to_process, data_dir)
    else:
        logger.info("Aucun article trouvé à télécharger pour l'option et les paramètres choisis.")

    total_downloaded = final_results.get('success', 0)
    logger.info(f"Opération terminée. {total_downloaded} nouveaux articles téléchargés (selon l'index).")
    print(f"\nOpération terminée.")
    print("--- Résumé ---")
    for status, count in sorted(final_results.items()):
        print(f"{status.replace('_', ' ').capitalize():<25}: {count}")
    print("--------------")
    print(f"Consultez '{CONFIG['log_file']}' pour les détails.")
    return 0


if __name__ == "__main__":
    sys.exit(main())