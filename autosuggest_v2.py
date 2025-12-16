"""
BajajMall Autosuggest API v2 - Trie-based Implementation
Features:
- Fast prefix matching with Trie data structure
- Multiple tries: Keywords, Brands, Categories, Products
- Ranking based on popularity/frequency
- Fuzzy matching for typo tolerance
- Returns max 7 suggestions from actual dataset
- Attribute-level chips for category-specific filters
"""

import json
import re
import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set, Any
from functools import lru_cache
from dataclasses import dataclass, field
from elasticsearch import Elasticsearch
from rapidfuzz import fuzz, process as fuzz_process

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================== TRIE NODE ===================
@dataclass
class TrieNode:
    """Node in the Trie data structure"""
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    is_end: bool = False
    word: str = ""
    score: int = 0  # Popularity score for ranking
    metadata: Dict = field(default_factory=dict)  # type, category, etc.


class Trie:
    """
    Trie data structure for fast prefix matching
    Supports:
    - Insert with score (popularity)
    - Prefix search with ranking
    - Fuzzy matching for typos
    """
    
    def __init__(self, name: str = "default"):
        self.root = TrieNode()
        self.name = name
        self.word_count = 0
        self.all_words: Dict[str, Tuple[int, Dict]] = {}  # word -> (score, metadata)
    
    def insert(self, word: str, score: int = 1, metadata: Dict = None):
        """Insert word into trie with optional score and metadata"""
        if not word or not word.strip():
            return
        
        word_lower = word.lower().strip()
        node = self.root
        
        for char in word_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end:
            self.word_count += 1
        
        node.is_end = True
        node.word = word.strip()  # Store original case
        node.score = max(node.score, score)  # Keep highest score
        node.metadata = metadata or {}
        
        # Also store in all_words for fuzzy matching
        self.all_words[word_lower] = (score, metadata or {})
    
    def search_prefix(self, prefix: str, limit: int = 10) -> List[Tuple[str, int, Dict]]:
        """
        Search for words starting with prefix
        Returns: List of (word, score, metadata) sorted by score descending
        """
        if not prefix:
            return []
        
        prefix_lower = prefix.lower().strip()
        node = self.root
        
        # Navigate to prefix node
        for char in prefix_lower:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]
        
        # Collect all words under this prefix
        results = []
        self._collect_words(node, results)
        
        # Sort by score (descending), then alphabetically
        results.sort(key=lambda x: (-x[1], x[0].lower()))
        return results[:limit]
    
    def _collect_words(self, node: TrieNode, results: List[Tuple[str, int, Dict]]):
        """Recursively collect all words under a node"""
        if node.is_end:
            results.append((node.word, node.score, node.metadata))
        
        for child in node.children.values():
            self._collect_words(child, results)
    
    def search_contains(self, substring: str, limit: int = 10) -> List[Tuple[str, int, Dict]]:
        """
        Search for words containing substring (slower, use sparingly)
        Returns: List of (word, score, metadata)
        """
        if not substring:
            return []
        
        substring_lower = substring.lower().strip()
        results = []
        
        for word_lower, (score, metadata) in self.all_words.items():
            if substring_lower in word_lower:
                # Get original word from trie
                node = self.root
                for char in word_lower:
                    if char in node.children:
                        node = node.children[char]
                if node.is_end:
                    results.append((node.word, score, metadata))
        
        results.sort(key=lambda x: (-x[1], x[0].lower()))
        return results[:limit]
    
    def fuzzy_search(self, query: str, limit: int = 10, threshold: int = 70) -> List[Tuple[str, int, Dict]]:
        """
        Fuzzy search for words similar to query
        Uses rapidfuzz for fast fuzzy matching
        Returns: List of (word, score, metadata) sorted by similarity
        """
        if not query or len(query) < 2:
            return []
        
        query_lower = query.lower().strip()
        
        # Get all words for fuzzy matching
        all_word_list = list(self.all_words.keys())
        
        if not all_word_list:
            return []
        
        # Use rapidfuzz for fast fuzzy matching
        matches = fuzz_process.extract(
            query_lower, 
            all_word_list, 
            scorer=fuzz.WRatio,
            limit=limit * 2  # Get more for filtering
        )
        
        results = []
        for match_word, match_score, _ in matches:
            if match_score >= threshold:
                word_score, metadata = self.all_words[match_word]
                # Navigate to get original case
                node = self.root
                for char in match_word:
                    if char in node.children:
                        node = node.children[char]
                if node.is_end:
                    # Combine fuzzy score with popularity score
                    combined_score = (match_score * word_score) / 100
                    results.append((node.word, int(combined_score), metadata))
        
        results.sort(key=lambda x: (-x[1], x[0].lower()))
        return results[:limit]


# =================== AUTOSUGGEST DATA ===================
class AutosuggestData:
    """
    Manages all autosuggest data with multiple tries
    - Keywords: Common search terms
    - Brands: Product brands
    - Categories: Product categories
    - Products: Popular products (model names)
    - Attributes: Storage, RAM, Size, etc.
    """
    
    def __init__(self, es_client=None, product_index: str = None, autosuggest_index: str = None):
        self.es = es_client
        self.product_index = product_index
        self.autosuggest_index = autosuggest_index
        
        # Initialize tries
        self.keyword_trie = Trie("keywords")
        self.brand_trie = Trie("brands")
        self.category_trie = Trie("categories")
        self.product_trie = Trie("products")
        self.attribute_trie = Trie("attributes")
        
        # Category-specific attribute chips
        self.category_chips: Dict[str, Tuple[List[str], str]] = {}
        
        # Brand -> Category mapping for contextual suggestions
        self.brand_categories: Dict[str, Set[str]] = defaultdict(set)
        
        # Category -> Brands mapping
        self.category_brands: Dict[str, Set[str]] = defaultdict(set)
        
        # Popularity scores from ES
        self.brand_scores: Dict[str, int] = {}
        self.category_scores: Dict[str, int] = {}
        
        # Build data
        self._build_static_data()
        if es_client:
            self._build_from_elasticsearch()
    
    def _build_static_data(self):
        """Build tries from static data (BUSINESS_AUTOSUGGEST, etc.)"""
        from utils import (
            BUSINESS_AUTOSUGGEST, 
            BUSINESS_SYNONYMS, 
            CATEGORY_CANONICAL,
            CATEGORY_HARDCODED_CHIPS,
            TWOWHEELER_AUTOSUGGEST
        )
        
        # Category chips
        self.category_chips = CATEGORY_HARDCODED_CHIPS.copy()
        
        # Add canonical category mappings
        for cat_key, canonical in CATEGORY_CANONICAL.items():
            self.category_trie.insert(cat_key, score=100, metadata={"canonical": canonical, "type": "category"})
            self.category_trie.insert(canonical, score=100, metadata={"canonical": canonical, "type": "category"})
        
        # Add synonyms as keywords
        for cat_key, synonyms in BUSINESS_SYNONYMS.items():
            canonical = CATEGORY_CANONICAL.get(cat_key, cat_key)
            for syn in synonyms:
                self.keyword_trie.insert(syn, score=80, metadata={"category": canonical, "type": "keyword"})
        
        # Add two-wheeler autosuggest
        for term in TWOWHEELER_AUTOSUGGEST:
            self.keyword_trie.insert(term, score=70, metadata={"category": "two wheeler", "type": "keyword"})
        
        # Add business autosuggest keywords
        for key, suggestions in BUSINESS_AUTOSUGGEST.items():
            # Key itself is a keyword
            self.keyword_trie.insert(key.replace("_", " "), score=90, metadata={"type": "keyword"})
            
            # Suggestions are searchable
            for suggestion in suggestions:
                self.keyword_trie.insert(suggestion, score=60, metadata={"type": "suggestion", "parent_key": key})
        
        # Add common search patterns
        common_patterns = [
            # Price-based
            ("under 10000", 95), ("under 15000", 95), ("under 20000", 95), ("under 30000", 90),
            ("under 50000", 85), ("below 10000", 85), ("below 20000", 85),
            
            # Quality/Type
            ("best", 95), ("top", 90), ("latest", 90), ("new", 85),
            ("4g", 80), ("5g", 90), ("smart", 85),
            
            # Actions
            ("buy", 70), ("price", 75), ("offer", 80), ("discount", 80), ("emi", 85),
        ]
        
        for pattern, score in common_patterns:
            self.keyword_trie.insert(pattern, score=score, metadata={"type": "pattern"})
        
        # Add static brands (most popular)
        static_brands = [
            ("Samsung", 100), ("Apple", 100), ("OnePlus", 95), ("Vivo", 90), ("OPPO", 90),
            ("Realme", 90), ("Xiaomi", 85), ("Redmi", 85), ("MI", 85), ("Nothing", 80),
            ("Motorola", 80), ("Nokia", 75), ("Google", 85), ("iQOO", 80),
            ("LG", 95), ("Sony", 90), ("Panasonic", 85), ("Haier", 80), ("Whirlpool", 85),
            ("Godrej", 85), ("Voltas", 85), ("Daikin", 85), ("Blue Star", 80), ("Carrier", 80),
            ("HP", 90), ("Dell", 90), ("Lenovo", 90), ("Asus", 85), ("Acer", 85),
            ("MSI", 80), ("Apple MacBook", 85),
            ("Hero", 85), ("Honda", 90), ("TVS", 85), ("Bajaj", 85), ("Royal Enfield", 90),
            ("Yamaha", 85), ("Suzuki", 80), ("Ather", 80), ("Ola Electric", 75),
            ("Maruti", 90), ("Hyundai", 90), ("Tata", 90), ("Mahindra", 85), ("Kia", 85),
            ("Toyota", 85),
            ("Bosch", 80), ("IFB", 80), ("Kent", 80), ("Aquaguard", 75), ("Eureka Forbes", 75),
            ("Havells", 80), ("Orient", 75), ("Crompton", 75), ("Usha", 75),
            ("JBL", 85), ("Bose", 85), ("Sony", 85), ("Boat", 80), ("Sennheiser", 75),
        ]
        
        for brand, score in static_brands:
            self.brand_trie.insert(brand, score=score, metadata={"type": "brand"})
        
        # Add static categories with scores
        static_categories = [
            ("smartphone", 100), ("mobile", 100), ("phone", 100),
            ("laptop", 95), ("laptops", 95),
            ("television", 90), ("tv", 90), ("smart tv", 90),
            ("refrigerator", 90), ("fridge", 90),
            ("washing machine", 88), ("washer", 85),
            ("air conditioner", 88), ("ac", 90),
            ("air cooler", 80), ("cooler", 80),
            ("microwave", 80), ("oven", 80),
            ("water purifier", 80), ("ro", 75),
            ("two wheeler", 85), ("bike", 85), ("scooter", 85),
            ("car", 90), ("suv", 85), ("sedan", 80),
            ("watch", 80), ("smartwatch", 85),
            ("camera", 80), ("dslr", 75),
            ("speaker", 80), ("headphone", 80), ("earphone", 80), ("earbuds", 85),
            ("tablet", 80), ("ipad", 80),
            ("printer", 70), ("monitor", 75),
            ("fan", 75), ("ceiling fan", 75),
            ("inverter", 70), ("ups", 65),
            ("mattress", 75), ("bed", 75), ("sofa", 75), ("furniture", 80),
            ("tractor", 70),
            # Kitchen appliances
            ("chimney", 80), ("kitchen chimney", 80), ("gas stove", 75), ("induction cooktop", 75),
            ("induction", 75), ("mixer grinder", 75), ("juicer", 70), ("blender", 70),
            ("dishwasher", 75), ("water heater", 80), ("geyser", 80),
            # Home appliances
            ("vacuum cleaner", 75), ("air purifier", 75), ("dehumidifier", 70),
            ("iron", 70), ("steam iron", 70), ("heater", 70), ("room heater", 70),
        ]
        
        for cat, score in static_categories:
            self.category_trie.insert(cat, score=score, metadata={"type": "category"})
        
        # Add common attribute values
        storage_values = ["4GB", "8GB", "16GB", "32GB", "64GB", "128GB", "256GB", "512GB", "1TB"]
        ram_values = ["2GB RAM", "3GB RAM", "4GB RAM", "6GB RAM", "8GB RAM", "12GB RAM", "16GB RAM"]
        screen_sizes = ["32 inch", "43 inch", "50 inch", "55 inch", "65 inch", "75 inch"]
        ac_capacity = ["1 ton", "1.5 ton", "2 ton"]
        
        for val in storage_values:
            self.attribute_trie.insert(val, score=80, metadata={"type": "storage"})
        for val in ram_values:
            self.attribute_trie.insert(val, score=80, metadata={"type": "ram"})
        for val in screen_sizes:
            self.attribute_trie.insert(val, score=75, metadata={"type": "screen_size"})
        for val in ac_capacity:
            self.attribute_trie.insert(val, score=75, metadata={"type": "capacity"})
        
        logger.info(f"Static data loaded: {self.keyword_trie.word_count} keywords, "
                   f"{self.brand_trie.word_count} brands, {self.category_trie.word_count} categories")
    
    def _build_from_elasticsearch(self):
        """Build tries from Elasticsearch indices"""
        if not self.es:
            return
        
        try:
            # Get brands from product index with counts
            if self.product_index:
                self._load_brands_from_products()
                self._load_categories_from_products()
            
            # Get suggestions from autosuggest index
            if self.autosuggest_index:
                self._load_from_autosuggest_index()
            
            logger.info(f"ES data loaded: {self.brand_trie.word_count} brands, "
                       f"{self.category_trie.word_count} categories, "
                       f"{self.product_trie.word_count} products")
        except Exception as e:
            logger.error(f"Error loading from ES: {e}")
    
    def _load_brands_from_products(self):
        """Load brands from product index with popularity scores"""
        try:
            resp = self.es.search(
                index=self.product_index,
                body={
                    "size": 0,
                    "aggs": {
                        "brands": {
                            "terms": {"field": "manufacturer_desc", "size": 500}
                        }
                    }
                }
            )
            
            for bucket in resp['aggregations']['brands']['buckets']:
                brand_name = bucket['key']
                doc_count = bucket['doc_count']
                
                # Skip generic/invalid brands
                if brand_name.lower() in ['mattress', 'photography', 'generic brand', 'unknown']:
                    continue
                
                # Score based on product count (log scale)
                score = min(100, int(50 + 10 * (doc_count / 100)))
                
                self.brand_trie.insert(brand_name, score=score, metadata={"type": "brand", "count": doc_count})
                self.brand_scores[brand_name.lower()] = doc_count
                
        except Exception as e:
            logger.error(f"Error loading brands: {e}")
    
    def _load_categories_from_products(self):
        """Load categories from product index"""
        try:
            resp = self.es.search(
                index=self.product_index,
                body={
                    "size": 0,
                    "aggs": {
                        "categories": {
                            "terms": {"field": "actual_category", "size": 100}
                        }
                    }
                }
            )
            
            for bucket in resp['aggregations']['categories']['buckets']:
                cat_name = bucket['key']
                doc_count = bucket['doc_count']
                
                score = min(100, int(60 + 8 * (doc_count / 1000)))
                
                self.category_trie.insert(cat_name, score=score, metadata={"type": "category", "count": doc_count})
                self.category_scores[cat_name.lower()] = doc_count
                
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
    
    def _load_from_autosuggest_index(self):
        """Load data from autosuggest index"""
        try:
            # Load brands
            resp = self.es.search(
                index=self.autosuggest_index,
                body={
                    "size": 1000,
                    "query": {"term": {"type": "brand"}}
                }
            )
            
            for hit in resp['hits']['hits']:
                brand_name = hit['_source'].get('value', '')
                if brand_name:
                    existing_score = self.brand_scores.get(brand_name.lower(), 0)
                    score = max(70, min(100, int(70 + existing_score / 100)))
                    self.brand_trie.insert(brand_name, score=score, metadata={"type": "brand"})
            
            # Load categories
            resp = self.es.search(
                index=self.autosuggest_index,
                body={
                    "size": 200,
                    "query": {"term": {"type": "category"}}
                }
            )
            
            for hit in resp['hits']['hits']:
                cat_name = hit['_source'].get('value', '')
                if cat_name:
                    existing_score = self.category_scores.get(cat_name.lower(), 0)
                    score = max(70, min(100, int(70 + existing_score / 500)))
                    self.category_trie.insert(cat_name, score=score, metadata={"type": "category"})
            
            # Load popular products (sample)
            resp = self.es.search(
                index=self.autosuggest_index,
                body={
                    "size": 5000,
                    "query": {"term": {"type": "product"}}
                }
            )
            
            for hit in resp['hits']['hits']:
                product_name = hit['_source'].get('value', '')
                if product_name and len(product_name) < 100:
                    # Extract brand and model for better matching
                    self.product_trie.insert(product_name, score=50, metadata={"type": "product"})
                    
        except Exception as e:
            logger.error(f"Error loading from autosuggest index: {e}")
    
    def get_brand_products(self, brand: str, limit: int = 5) -> List[str]:
        """Get popular products for a brand from ES"""
        if not self.es or not self.product_index:
            return []
        
        try:
            resp = self.es.search(
                index=self.product_index,
                body={
                    "size": limit,
                    "query": {
                        "match": {"manufacturer_desc": brand}
                    },
                    "_source": ["product_name"],
                    "collapse": {"field": "modelid"}
                }
            )
            
            products = []
            for hit in resp['hits']['hits']:
                name = hit['_source'].get('product_name', '')
                if name:
                    # Truncate long names
                    if len(name) > 60:
                        name = name[:57] + "..."
                    products.append(name)
            
            return products
        except Exception as e:
            logger.error(f"Error getting brand products: {e}")
            return []


# =================== AUTOSUGGEST ENGINE ===================
class AutosuggestEngine:
    """
    Main autosuggest engine with intelligent suggestion ranking
    """
    
    MAX_SUGGESTIONS = 7
    
    def __init__(self, data: AutosuggestData):
        self.data = data
    
    def get_suggestions(self, query: str) -> Dict[str, Any]:
        """
        Get autosuggest results for a query
        Returns max 7 suggestions with proper ranking
        """
        start_time = time.time()
        
        if not query or not query.strip():
            return self._empty_response()
        
        query = query.strip()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Collect suggestions from different sources
        suggestions = []
        seen = set()
        
        # Special handling for brand + category queries (e.g., "samsung phone", "lg refrigerator")
        if len(query_words) >= 2:
            brand_category_suggestions = self._handle_brand_category_query(query_lower, query_words, seen)
            if brand_category_suggestions:
                suggestions.extend(brand_category_suggestions)
        
        # 1. Brand prefix match (highest priority for brand queries)
        if query_type in ['brand', 'brand_product', 'general']:
            brand_matches = self.data.brand_trie.search_prefix(query_lower, limit=5)
            for word, score, meta in brand_matches:
                if word.lower() not in seen:
                    suggestions.append({
                        "text": word,
                        "type": "brand",
                        "score": score + 50,  # Boost brands
                        "meta": meta
                    })
                    seen.add(word.lower())
        
        # 2. Category prefix match
        if query_type in ['category', 'general']:
            cat_matches = self.data.category_trie.search_prefix(query_lower, limit=5)
            for word, score, meta in cat_matches:
                if word.lower() not in seen:
                    suggestions.append({
                        "text": word.title(),
                        "type": "category",
                        "score": score + 40,
                        "meta": meta
                    })
                    seen.add(word.lower())
        
        # 3. Keyword prefix match
        keyword_matches = self.data.keyword_trie.search_prefix(query_lower, limit=7)
        for word, score, meta in keyword_matches:
            if word.lower() not in seen:
                suggestions.append({
                    "text": word,
                    "type": "keyword",
                    "score": score + 30,
                    "meta": meta
                })
                seen.add(word.lower())
        
        # 4. Product prefix match (for longer queries)
        if len(query) >= 3:
            product_matches = self.data.product_trie.search_prefix(query_lower, limit=5)
            for word, score, meta in product_matches:
                if word.lower() not in seen:
                    suggestions.append({
                        "text": word,
                        "type": "product",
                        "score": score + 20,
                        "meta": meta
                    })
                    seen.add(word.lower())
        
        # 5. If few results, try contains search
        if len(suggestions) < 3 and len(query) >= 3:
            contains_matches = self.data.keyword_trie.search_contains(query_lower, limit=5)
            for word, score, meta in contains_matches:
                if word.lower() not in seen:
                    suggestions.append({
                        "text": word,
                        "type": "keyword",
                        "score": score + 10,
                        "meta": meta
                    })
                    seen.add(word.lower())
        
        # 6. If still few results, try fuzzy match
        if len(suggestions) < 3 and len(query) >= 2:
            # Try fuzzy on keywords first
            fuzzy_matches = self.data.keyword_trie.fuzzy_search(query_lower, limit=5, threshold=75)
            for word, score, meta in fuzzy_matches:
                if word.lower() not in seen:
                    suggestions.append({
                        "text": word,
                        "type": "keyword_fuzzy",
                        "score": score,
                        "meta": meta
                    })
                    seen.add(word.lower())
            
            # Try fuzzy on brands
            if len(suggestions) < 3:
                fuzzy_brands = self.data.brand_trie.fuzzy_search(query_lower, limit=3, threshold=70)
                for word, score, meta in fuzzy_brands:
                    if word.lower() not in seen:
                        suggestions.append({
                            "text": word,
                            "type": "brand_fuzzy",
                            "score": score + 20,
                            "meta": meta
                        })
                        seen.add(word.lower())
        
        # 7. Generate contextual suggestions (brand + category combinations)
        contextual = self._generate_contextual_suggestions(query_lower, seen)
        suggestions.extend(contextual)
        
        # Sort by score and limit to MAX_SUGGESTIONS
        suggestions.sort(key=lambda x: -x['score'])
        suggestions = suggestions[:self.MAX_SUGGESTIONS]
        
        # Get chips and filter text for detected category
        chips, filter_text = self._get_chips_for_query(query_lower)
        
        # Format response
        keywords = [s['text'] for s in suggestions]
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Autosuggest for '{query}': {len(keywords)} suggestions in {elapsed:.2f}ms")
        
        return {
            "message": "success",
            "response": {
                "keywords": keywords,
                "chips": chips,
                "filter_text": filter_text,
                "synonym_suggestions": [],
                "language": "english",
                "query_type": query_type,
                "timing_ms": round(elapsed, 2)
            }
        }
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query for better suggestion strategy"""
        query_lower = query.lower()
        
        # Check if starts with a known brand
        for word, _, _ in self.data.brand_trie.search_prefix(query_lower, limit=1):
            if query_lower.startswith(word.lower()[:3]):
                # Check if there's more after brand (brand + product)
                if len(query_lower) > len(word) + 2:
                    return "brand_product"
                return "brand"
        
        # Check if it's a category-like query
        for word, _, _ in self.data.category_trie.search_prefix(query_lower, limit=1):
            if query_lower.startswith(word.lower()[:3]):
                return "category"
        
        # Check for price patterns
        if any(p in query_lower for p in ['under', 'below', 'price', 'budget', 'cheap']):
            return "price_intent"
        
        # Check for attribute patterns
        if any(p in query_lower for p in ['gb', 'inch', 'ton', 'kg', 'litre', 'watt']):
            return "attribute"
        
        return "general"
    
    def _handle_brand_category_query(self, query: str, query_words: List[str], seen: Set[str]) -> List[Dict]:
        """
        Handle multi-word queries like 'samsung phone', 'lg refrigerator', 'voltas ac'
        Returns suggestions prioritizing the brand + category combination
        """
        suggestions = []
        
        # Category keyword mapping to display text and ES suggestions
        category_mappings = {
            # Phone/mobile
            'phone': ('phone', 'mobile', 'smartphone', 'mobiles'),
            'mobile': ('mobile', 'phone', 'smartphone', 'mobiles'),
            'smartphone': ('smartphone', 'phone', 'mobile'),
            'phones': ('phones', 'mobile', 'smartphone'),
            # Refrigerator
            'refrigerator': ('refrigerator', 'fridge', 'double door refrigerator', 'single door refrigerator', 'side by side refrigerator'),
            'fridge': ('fridge', 'refrigerator', 'double door refrigerator'),
            'ref': ('refrigerator', 'fridge'),
            # Washing machine
            'washing': ('washing machine', 'front load washing machine', 'top load washing machine'),
            'washer': ('washing machine', 'washer'),
            # TV
            'tv': ('TV', 'smart tv', 'led tv', '4k tv', 'oled tv'),
            'television': ('television', 'tv', 'smart tv'),
            # AC
            'ac': ('AC', 'split ac', 'window ac', '1.5 ton ac', 'inverter ac'),
            'air': ('air conditioner', 'AC'),
            'airconditioner': ('air conditioner', 'AC'),
            # Laptop
            'laptop': ('laptop', 'gaming laptop', 'business laptop', 'laptop for students'),
            'laptops': ('laptops', 'laptop', 'gaming laptop'),
            # Cooler
            'cooler': ('air cooler', 'cooler', 'desert cooler'),
            # Microwave
            'microwave': ('microwave', 'microwave oven', 'convection microwave'),
            # Water purifier
            'purifier': ('water purifier', 'air purifier'),
            'water': ('water purifier', 'water heater'),
            # Dishwasher
            'dishwasher': ('dishwasher', 'dish washer'),
            # Kitchen Appliances
            'chimney': ('chimney', 'kitchen chimney', 'ducted chimney', 'ductless chimney'),
            'mixer': ('mixer grinder', 'mixer', 'blender'),
            'oven': ('oven', 'microwave oven', 'convection oven', 'otg'),
            'geyser': ('geyser', 'water heater', 'instant geyser'),
            'heater': ('water heater', 'geyser', 'room heater'),
            # Home Appliances
            'vacuum': ('vacuum cleaner', 'vacuum'),
            'iron': ('iron', 'steam iron'),
            'fan': ('fan', 'ceiling fan', 'table fan', 'pedestal fan'),
        }
        
        # Try to identify brand and category from query words
        detected_brand = None
        detected_category_word = None
        
        # Check first word as brand
        first_word = query_words[0]
        brand_matches = self.data.brand_trie.search_prefix(first_word, limit=1)
        if brand_matches:
            match_word = brand_matches[0][0].lower()
            # Check if first word is actually a brand
            if first_word.startswith(match_word[:min(3, len(match_word))]) or \
               match_word.startswith(first_word[:min(3, len(first_word))]):
                detected_brand = brand_matches[0][0]
        
        # Check remaining words for category
        if detected_brand:
            remaining = ' '.join(query_words[1:])
            for cat_key, variations in category_mappings.items():
                if remaining.startswith(cat_key) or cat_key.startswith(remaining):
                    detected_category_word = cat_key
                    break
        
        # Generate suggestions for brand + category
        if detected_brand and detected_category_word:
            variations = category_mappings.get(detected_category_word, (detected_category_word,))
            
            # Add the exact query as first suggestion (with proper casing)
            exact_match = f"{detected_brand} {variations[0]}"
            if exact_match.lower() not in seen:
                suggestions.append({
                    "text": exact_match,
                    "type": "brand_category",
                    "score": 200,  # Highest priority
                    "meta": {"brand": detected_brand, "category": variations[0]}
                })
                seen.add(exact_match.lower())
            
            # Add variations
            for var in variations[1:4]:  # Limit to 3 more variations
                combo = f"{detected_brand} {var}"
                if combo.lower() not in seen:
                    suggestions.append({
                        "text": combo,
                        "type": "brand_category",
                        "score": 180,
                        "meta": {"brand": detected_brand, "category": var}
                    })
                    seen.add(combo.lower())
            
            # Try to get actual products from ES for this brand + category
            try:
                brand_products = self._get_brand_category_products(detected_brand, detected_category_word)
                for prod in brand_products[:3]:
                    if prod.lower() not in seen:
                        suggestions.append({
                            "text": prod,
                            "type": "product",
                            "score": 150,
                            "meta": {"brand": detected_brand, "category": detected_category_word}
                        })
                        seen.add(prod.lower())
            except:
                pass
        
        return suggestions
    
    def _get_brand_category_products(self, brand: str, category_word: str) -> List[str]:
        """Get actual products from ES for a brand + category combination"""
        try:
            # Map category word to ES asset_category_name patterns
            category_es_mapping = {
                'phone': ['mobile'],
                'mobile': ['mobile'],
                'smartphone': ['mobile'],
                'refrigerator': ['refrigerator'],
                'fridge': ['refrigerator'],
                'washing': ['washing machine'],
                'washer': ['washing machine'],
                'tv': ['television', 'tv', 'led'],
                'television': ['television', 'tv'],
                'ac': ['air conditioner', 'split ac', 'window ac'],
                'laptop': ['laptop'],
                'cooler': ['cooler'],
                'microwave': ['microwave'],
                'purifier': ['purifier'],
                'dishwasher': ['dishwasher'],
                # Kitchen appliances
                'chimney': ['chimney', 'kitchen'],
                'mixer': ['mixer', 'grinder'],
                'oven': ['oven', 'microwave', 'otg'],
                'geyser': ['geyser', 'water heater'],
                'heater': ['heater', 'geyser'],
                # Home appliances
                'vacuum': ['vacuum', 'cleaner'],
                'iron': ['iron'],
                'fan': ['fan'],
            }
            
            search_patterns = category_es_mapping.get(category_word, [category_word])
            
            # Build query
            should_clauses = [{"match_phrase": {"asset_category_name": pat}} for pat in search_patterns]
            
            resp = self.data.es.search(
                index=self.data.product_index,
                body={
                    "size": 5,
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"manufacturer_desc": brand}}
                            ],
                            "should": should_clauses,
                            "minimum_should_match": 1
                        }
                    },
                    "_source": ["product_name"],
                    "collapse": {"field": "modelid"}
                }
            )
            
            products = []
            for hit in resp['hits']['hits']:
                name = hit['_source'].get('product_name', '')
                if name and len(name) <= 60:
                    products.append(name)
            
            return products
        except Exception as e:
            logger.error(f"Error getting brand category products: {e}")
            return []
    
    def _generate_contextual_suggestions(self, query: str, seen: Set[str]) -> List[Dict]:
        """Generate contextual suggestions like 'Samsung phone', 'Samsung TV'"""
        suggestions = []
        
        # Check if query is a brand
        brand_match = self.data.brand_trie.search_prefix(query, limit=1)
        if brand_match:
            brand_name = brand_match[0][0]
            brand_lower = brand_name.lower()
            
            # Get common categories for this brand
            common_combos = []
            
            # Phone brands
            if brand_lower in ['samsung', 'apple', 'oppo', 'vivo', 'realme', 'redmi', 'xiaomi', 'oneplus', 'nothing', 'motorola', 'nokia', 'iqoo']:
                common_combos.extend([
                    f"{brand_name} phone",
                    f"{brand_name} mobile",
                    f"{brand_name} smartphone"
                ])
            
            # TV brands
            if brand_lower in ['samsung', 'lg', 'sony', 'panasonic', 'mi', 'oneplus', 'tcl', 'vu']:
                common_combos.append(f"{brand_name} TV")
            
            # Appliance brands
            if brand_lower in ['samsung', 'lg', 'whirlpool', 'godrej', 'haier', 'bosch', 'ifb']:
                common_combos.extend([
                    f"{brand_name} refrigerator",
                    f"{brand_name} washing machine"
                ])
            
            # AC brands
            if brand_lower in ['voltas', 'daikin', 'lg', 'samsung', 'blue star', 'carrier', 'lloyd']:
                common_combos.append(f"{brand_name} AC")
            
            # Laptop brands
            if brand_lower in ['hp', 'dell', 'lenovo', 'asus', 'acer', 'msi', 'apple']:
                common_combos.append(f"{brand_name} laptop")
            
            for combo in common_combos[:3]:  # Limit to 3 contextual suggestions
                if combo.lower() not in seen:
                    suggestions.append({
                        "text": combo,
                        "type": "contextual",
                        "score": 85,
                        "meta": {"type": "brand_category"}
                    })
                    seen.add(combo.lower())
        
        return suggestions
    
    def _get_chips_for_query(self, query: str) -> Tuple[List[str], str]:
        """Get attribute chips for a query based on detected category"""
        from utils import CATEGORY_CANONICAL, BUSINESS_SYNONYMS
        
        query_lower = query.lower().strip()
        
        # Direct category match
        if query_lower in self.data.category_chips:
            return self.data.category_chips[query_lower]
        
        # Try canonical mapping
        canonical = CATEGORY_CANONICAL.get(query_lower)
        if canonical and canonical in self.data.category_chips:
            return self.data.category_chips[canonical]
        
        # Try synonym mapping
        for cat_key, synonyms in BUSINESS_SYNONYMS.items():
            norm_synonyms = [s.lower() for s in synonyms]
            if query_lower in norm_synonyms:
                if cat_key in self.data.category_chips:
                    return self.data.category_chips[cat_key]
                canonical = CATEGORY_CANONICAL.get(cat_key)
                if canonical and canonical in self.data.category_chips:
                    return self.data.category_chips[canonical]
        
        # Check for category keywords in query
        category_keywords = {
            "phone": "mobile phones",
            "mobile": "mobile phones",
            "smartphone": "mobile phones",
            "fridge": "refrigerators",
            "refrigerator": "refrigerators",
            "ac": "ac",
            "air conditioner": "ac",
            "tv": "tv and home entertainment",
            "television": "tv and home entertainment",
            "laptop": "laptops",
            "washing machine": "washing machines",
            "washer": "washing machines",
            "bike": "two-wheeler",
            "scooter": "two-wheeler",
            "car": "new cars",
            "cooler": "air coolers",
            "tractor": "tractor",
        }
        
        for keyword, chip_key in category_keywords.items():
            if keyword in query_lower:
                if chip_key in self.data.category_chips:
                    return self.data.category_chips[chip_key]
        
        return [], ""
    
    def _empty_response(self) -> Dict[str, Any]:
        """Return empty response"""
        return {
            "message": "success",
            "response": {
                "keywords": [],
                "chips": [],
                "filter_text": "",
                "synonym_suggestions": [],
                "language": "english",
                "query_type": "empty",
                "timing_ms": 0
            }
        }


# =================== SINGLETON INSTANCE ===================
_autosuggest_data: Optional[AutosuggestData] = None
_autosuggest_engine: Optional[AutosuggestEngine] = None


def init_autosuggest(es_client=None, product_index: str = None, autosuggest_index: str = None):
    """Initialize autosuggest engine (call once at startup)"""
    global _autosuggest_data, _autosuggest_engine
    
    logger.info("Initializing autosuggest engine...")
    start = time.time()
    
    _autosuggest_data = AutosuggestData(es_client, product_index, autosuggest_index)
    _autosuggest_engine = AutosuggestEngine(_autosuggest_data)
    
    logger.info(f"Autosuggest initialized in {(time.time() - start)*1000:.2f}ms")
    
    return _autosuggest_engine


def get_autosuggest_engine() -> AutosuggestEngine:
    """Get the autosuggest engine instance"""
    global _autosuggest_engine
    
    if _autosuggest_engine is None:
        # Initialize with defaults if not already done
        init_autosuggest()
    
    return _autosuggest_engine


# =================== TEST ===================
if __name__ == "__main__":
    # Test without ES
    print("Testing autosuggest without ES...")
    engine = init_autosuggest()
    
    test_queries = [
        "sam",
        "samsung",
        "samsung ph",
        "iph",
        "iphone",
        "lap",
        "laptop",
        "wash",
        "refriger",
        "ac",
        "tv",
        "onepl",
        "realme",
        "under 20000",
        "best phone",
        "5g",
        "fan",
        "tractor",
    ]
    
    print("\n=== AUTOSUGGEST TEST ===\n")
    for q in test_queries:
        result = engine.get_suggestions(q)
        keywords = result['response']['keywords']
        chips = result['response']['chips']
        timing = result['response']['timing_ms']
        print(f"'{q}' ({timing}ms): {keywords[:7]}")
        if chips:
            print(f"  Chips: {chips[:5]}...")
        print()
