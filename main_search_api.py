"""
BajajMall Search API - Fixed Version
Bug Fixes Applied:
1. Enhanced fuzzy matching when regex/pool doesn't match - uses ES autosuggest fallback
2. Proper pagination with 26 unique SKUs per page using ES collapse
"""

from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import traceback
import json
import logging
import re
from rapidfuzz import fuzz, process as rapidfuzz_process
from collections import defaultdict
import time
from functools import lru_cache, wraps
from typing import Dict, List, Tuple, Optional, Any, Set

from utils import (
    CATEGORY_CANONICAL,
    TWOWHEELER_AUTOSUGGEST,
    BUSINESS_SYNONYMS,
    CATEGORY_HARDCODED_CHIPS,
    APPLE_TERMS,
    FILTER_ATTRIBUTE_EXCLUSIONS,
    SCOOTER_SYNONYMS,
    BUSINESS_AUTOSUGGEST,
    KNOWN_BRANDS,
    COMMON_PRODUCT_TERMS,
)

# =================== CONFIGS ===========================
PRODUCT_INDEX_NAME = "bajajmall_products_s3_esidx3_15102025_chkpt"
CATEGORY_INDEX_NAME = "bajajmall_categories_s3_esidx3_15102025_chkpt"
AUTOSUGGEST_INDEX_NAME = "bajajmall_autosuggest_s3_esidx3_15102025_chkpt"
IMAGE_DOMAIN = "https://mc.bajajfinserv.in/media/catalog/product"

# Pagination config
DEFAULT_PAGE_SIZE = 26  # 26 unique SKUs per page
MAX_PAGE_SIZE = 100

logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
es = Elasticsearch("http://localhost:9200")


# =================== LOGGING DECORATOR ===================
def log_api_call(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        logger.info(f"API {f.__name__} completed in {elapsed:.2f}ms")
        return result
    return decorated


# =================== NORMALIZATION HELPERS ===================
def normalize(text: str) -> str:
    """Normalize text: lowercase, remove special chars, normalize spaces."""
    if not text:
        return ""
    t = re.sub(r"[^a-zA-Z0-9\s&-]", "", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =================== BUG FIX 1: ENHANCED CORRECTION POOL ===================
def build_enhanced_correction_pool() -> Set[str]:
    """
    Build an enhanced correction pool that includes:
    - Categories and their canonical forms
    - All synonyms
    - Known brand names
    - Common product terms (storage, RAM, etc.)
    """
    pool = set()

    # Add categories
    for k, v in CATEGORY_CANONICAL.items():
        pool.add(normalize(k))
        pool.add(normalize(v))
        pool.add(normalize(k.replace("-", "")))
        pool.add(normalize(k.replace(" ", "")))

    # Add all synonyms
    for syns in BUSINESS_SYNONYMS.values():
        for s in syns:
            pool.add(normalize(s))
            pool.add(normalize(s.replace("-", "")))
            pool.add(normalize(s.replace(" ", "")))

    # Add known brands (NEW)
    for brand in KNOWN_BRANDS:
        pool.add(normalize(brand))
        pool.add(brand.lower().replace(" ", ""))

    # Add common product terms (NEW)
    for term in COMMON_PRODUCT_TERMS:
        pool.add(normalize(term))
        pool.add(term.lower().replace(" ", ""))

    # Add two-wheeler specific terms
    for term in TWOWHEELER_AUTOSUGGEST:
        pool.add(normalize(term))

    # Add scooter synonyms
    for term in SCOOTER_SYNONYMS:
        pool.add(normalize(term))

    return {p for p in pool if p and len(p) > 1}

CORRECTION_POOL = build_enhanced_correction_pool()
CORRECTION_POOL_LIST = sorted(CORRECTION_POOL)  # For rapidfuzz


# =================== BUG FIX 1: ENHANCED FUZZY MATCHING ===================
@lru_cache(maxsize=1024)
def fuzzy_match_from_es(query: str) -> Optional[str]:
    """
    Fallback fuzzy matching using ES autosuggest index.
    Called when the static correction pool doesn't have a good match.
    """
    try:
        response = es.search(
            index=AUTOSUGGEST_INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"value": {"query": query, "fuzziness": "AUTO"}}},
                            {"match_phrase_prefix": {"value": query}},
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": 5,
                "_source": ["value", "type"]
            }
        )

        hits = response.get("hits", {}).get("hits", [])
        if hits:
            # Find best fuzzy match
            best_match = None
            best_score = 0
            query_lower = query.lower()

            for hit in hits:
                value = hit["_source"].get("value", "")
                value_lower = value.lower()

                # Calculate similarity score
                score = fuzz.ratio(query_lower, value_lower)

                # Boost if it starts with the query
                if value_lower.startswith(query_lower[:min(3, len(query_lower))]):
                    score += 20

                if score > best_score and score >= 65:
                    best_score = score
                    best_match = value

            if best_match:
                logger.info(f"ES fuzzy match: '{query}' -> '{best_match}' (score: {best_score})")
                return normalize(best_match)

    except Exception as e:
        logger.warning(f"ES fuzzy match failed for '{query}': {e}")

    return None


def correct_query(user_query: str) -> str:
    """
    Enhanced query correction with multiple fallbacks:
    1. Exact match in correction pool
    2. Fuzzy match against correction pool (threshold 78%)
    3. Word-by-word fuzzy correction (threshold 75%)
    4. NEW: ES autosuggest fuzzy fallback for unmatched queries
    """
    if not user_query:
        return ""

    user_query_n = normalize(user_query)

    # Exact match check
    if user_query_n in CORRECTION_POOL:
        return user_query_n

    # Try full phrase fuzzy match
    match_result = rapidfuzz_process.extractOne(
        user_query_n,
        CORRECTION_POOL_LIST,
        scorer=fuzz.ratio,
        score_cutoff=78
    )

    if match_result:
        match, score, _ = match_result
        logger.info(f"Pool fuzzy match: '{user_query}' -> '{match}' (score: {score})")
        return match

    # Word-by-word correction
    words = user_query_n.split()
    new_words = []
    any_corrected = False

    for w in words:
        if len(w) <= 2:
            new_words.append(w)
            continue

        word_match = rapidfuzz_process.extractOne(
            w,
            CORRECTION_POOL_LIST,
            scorer=fuzz.ratio,
            score_cutoff=75
        )

        if word_match:
            m, sc, _ = word_match
            if sc >= 75:
                new_words.append(m)
                if m != w:
                    any_corrected = True
            else:
                new_words.append(w)
        else:
            new_words.append(w)

    if any_corrected:
        corrected = " ".join(new_words)
        logger.info(f"Word-by-word correction: '{user_query}' -> '{corrected}'")
        return corrected

    # BUG FIX 1: NEW - ES autosuggest fallback for completely unmatched queries
    es_match = fuzzy_match_from_es(user_query_n)
    if es_match:
        return es_match

    # No correction found, return normalized original
    return user_query_n


# =================== PRICE/EMI PARSING ===================
def parse_price_from_query(query: str) -> Tuple[str, Optional[Dict]]:
    """
    Parse price and EMI filters from natural language query.
    Returns: (cleaned_query, filters_dict)
    """
    q = query.lower()
    cleaned_query = q
    filters = {}

    # EMI patterns
    emi_patterns = [
        (r'(lowest )?(emi|installment|monthly|per month)[^\d]{0,10}(under|below|less than|upto|<=)\s*([0-9]{3,8})',
         lambda m: ('lowest_emi', {'lte': int(m.group(4))})),
        (r'(lowest )?(emi|installment|monthly|per month)[^\d]{0,10}(above|over|greater than|more than|>=)\s*([0-9]{3,8})',
         lambda m: ('lowest_emi', {'gte': int(m.group(4))})),
        (r'(lowest )?(emi|installment|monthly|per month)[^\d]{0,10}(from|between|range)?\s*([0-9]{3,8})\s*(to|and|-)\s*([0-9]{3,8})',
         lambda m: ('lowest_emi', {'gte': min(int(m.group(4)), int(m.group(6))), 'lte': max(int(m.group(4)), int(m.group(6)))}))
    ]

    for pat, rng_fn in emi_patterns:
        for match in re.finditer(pat, cleaned_query):
            key, val = rng_fn(match)
            filters[key] = val
            cleaned_query = cleaned_query.replace(match.group(0), '').strip()

    # Price patterns
    price_patterns = [
        (r'(price|cost|offer price|rate|mop)[^\d]{0,10}(under|below|less than|upto|<=)\s*([0-9]{3,8})',
         lambda m: ('mop', {'lte': int(m.group(3))})),
        (r'(price|cost|offer price|rate|mop)[^\d]{0,10}(above|over|greater than|more than|>=)\s*([0-9]{3,8})',
         lambda m: ('mop', {'gte': int(m.group(3))})),
        (r'(price|cost|offer price|rate|mop)[^\d]{0,10}(from|between|range)?\s*([0-9]{3,8})\s*(to|and|-)\s*([0-9]{3,8})',
         lambda m: ('mop', {'gte': min(int(m.group(4)), int(m.group(6))), 'lte': max(int(m.group(4)), int(m.group(6)))}))
    ]

    for pat, rng_fn in price_patterns:
        for match in re.finditer(pat, cleaned_query):
            key, val = rng_fn(match)
            filters[key] = val
            cleaned_query = cleaned_query.replace(match.group(0), '').strip()

    # "Under 25000 emi"
    generic_emi = re.search(r'(under|below|less than|upto|<=)\s*([0-9]{3,8})\s*emi', cleaned_query)
    if generic_emi:
        filters['lowest_emi'] = {'lte': int(generic_emi.group(2))}
        cleaned_query = cleaned_query.replace(generic_emi.group(0), '').strip()

    # Generic "under X" (price)
    if 'emi' not in cleaned_query and 'installment' not in cleaned_query:
        general_patterns = [
            (r'(under|below|less than|upto|<=)\s*([0-9]{3,8})',
             lambda m: ('mop', {'lte': int(m.group(2))})),
            (r'(above|over|greater than|more than|>=)\s*([0-9]{3,8})',
             lambda m: ('mop', {'gte': int(m.group(2))})),
        ]
        for pat, rng_fn in general_patterns:
            for match in re.finditer(pat, cleaned_query):
                key, val = rng_fn(match)
                filters[key] = val
                cleaned_query = cleaned_query.replace(match.group(0), '').strip()

    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    if not cleaned_query:
        cleaned_query = query

    return cleaned_query, (filters if filters else None)


# =================== SEARCH TERM EXPANSION ===================
def expand_search_terms(user_query: str) -> List[str]:
    """Expand query into related search terms."""
    user_query_n = normalize(user_query)
    variants = {user_query_n}

    # Add canonical mapping
    canonical = CATEGORY_CANONICAL.get(user_query_n)
    if canonical:
        variants.add(canonical)

    # Add synonyms
    for canon, syns in BUSINESS_SYNONYMS.items():
        if user_query_n == canon or user_query_n in [s.lower() for s in syns]:
            variants.add(canon)
            variants.update(s.lower() for s in syns)

    # Add word splits
    variants.update(user_query_n.split())

    return sorted({t.strip() for t in variants if t.strip()})


# =================== IMAGE URL HELPER ===================
def update_image_url(product: Dict) -> Dict:
    """Fix relative image URLs to absolute."""
    for prod in product.get("products", []):
        image = prod.get("image", "")
        if image and not image.startswith("http"):
            if image.startswith("/"):
                prod["image"] = IMAGE_DOMAIN + image
            else:
                prod["image"] = IMAGE_DOMAIN + "/" + image
    return product


# =================== BUG FIX 2: PAGINATION WITH UNIQUE SKUs ===================
def build_search_query_with_collapse(
    user_query: str,
    filters: Optional[Dict] = None,
    city_id: Optional[str] = None,
    mapped_category: Optional[str] = None,
    price_filter: Optional[Dict] = None,
    emi_range: Optional[Dict] = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Dict:
    """
    Build ES query with collapse for unique SKU pagination.

    BUG FIX 2: Uses ES 'collapse' on modelid to ensure each page
    contains exactly page_size unique products (SKUs), not duplicates.
    """
    terms = expand_search_terms(user_query)
    should_clauses = []
    must_clauses = []
    must_not_clauses = []
    filter_clauses = []

    # Build should clauses for relevance
    for term in terms:
        should_clauses.extend([
            {"match": {"product_name": {"query": term, "boost": 5}}},
            {"match": {"search_field": {"query": term, "boost": 5}}},
            {"match": {"actual_category": {"query": term, "boost": 4}}},
            {"match_phrase_prefix": {"search_field": {"query": term, "boost": 3}}},
            {"match": {"search_field": {"query": term, "fuzziness": "AUTO", "boost": 2}}},
        ])

    # Category filter
    if mapped_category:
        should_clauses.extend([
            {"term": {"actual_category": {"value": mapped_category, "boost": 100}}},
            {"match": {"actual_category": {"query": mapped_category, "boost": 50}}},
        ])

    # City filter
    if city_id:
        city_key = f"citi_id_{city_id}" if not city_id.startswith("citi_id_") else city_id
        filter_clauses.append({
            "bool": {
                "should": [
                    {"term": {"cityid": city_key}},
                    {"term": {"cityid": "citi_id_0"}}
                ],
                "minimum_should_match": 1
            }
        })

    # Attribute filters
    if filters:
        for key, value in filters.items():
            if key in ["emi", "emi_min", "emi_max"]:
                continue
            values = [v.strip() for v in str(value).split(",") if v.strip()]
            if values:
                filter_clauses.append({"terms": {key: values}})

    # Price filter
    if price_filter:
        for field, rng in price_filter.items():
            filter_clauses.append({"range": {field: rng}})

    # EMI range filter
    if emi_range:
        filter_clauses.append({"range": {"lowest_emi": emi_range}})

    # Build final query
    es_query = {
        "bool": {
            "filter": filter_clauses,
            "must": must_clauses,
            "must_not": must_not_clauses,
            "should": should_clauses,
            "minimum_should_match": 1 if should_clauses else 0
        }
    }

    # BUG FIX 2: Calculate from_offset for pagination
    from_offset = (page - 1) * page_size

    query_body = {
        "query": es_query,
        "from": from_offset,
        "size": page_size,
        # BUG FIX 2: Collapse by modelid to ensure unique products
        "collapse": {
            "field": "modelid",
            "inner_hits": {
                "name": "variants",
                "size": 10,  # Get up to 10 variants per model
                "_source": ["products", "modelid"]
            }
        },
        "sort": [
            {"_score": {"order": "desc"}},
            {"lowest_emi": {"order": "asc"}}
        ],
        # Track total for accurate pagination info
        "track_total_hits": True
    }

    return query_body


def process_search_response(
    response: Dict,
    city_id: Optional[str] = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE
) -> Dict:
    """
    Process ES response with proper pagination info.

    BUG FIX 2: Returns accurate pagination metadata with unique SKU counts.
    """
    hits = response.get("hits", {}).get("hits", [])
    total_hits = response.get("hits", {}).get("total", {})

    if isinstance(total_hits, dict):
        total = total_hits.get("value", 0)
    else:
        total = total_hits

    products = []
    seen_model_ids = set()
    emi_values = []
    final_filters = {}

    input_city_id = str(city_id) if city_id else "0"
    city_id_to_check = f"citi_id_{input_city_id}"

    for hit in hits:
        source = hit.get("_source", {})
        model_id = source.get("modelid", "")

        # BUG FIX 2: Skip if we've already seen this model (shouldn't happen with collapse, but safety check)
        if model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)

        # Update image URLs
        source = update_image_url(source)

        # City filtering
        cityids = source.get("cityid", [])
        if isinstance(cityids, str):
            cityids = [cityids]

        if city_id_to_check in cityids:
            selected_city = city_id_to_check
        elif "citi_id_0" in cityids:
            selected_city = "citi_id_0"
        else:
            continue

        # Build product response
        emi_values.append(source.get("lowest_emi", 0))

        product_data = {
            "model_id": model_id,
            "model_launch_date": source.get("model_launch_date", "1970-01-01"),
            "mkp_active_flag": source.get("mkp_active_flag", 0),
            "avg_rating": source.get("avg_rating", 0),
            "asset_category_id": source.get("asset_category_id", 0),
            "asset_category_name": source.get("asset_category_name", "UNKNOWN"),
            "manufacturer_id": source.get("manufacturer_id", 999),
            "manufacturer_desc": source.get("manufacturer_desc", "UNKNOWN"),
            "category_type": source.get("category_type", "UNKNOWN"),
            "mop": source.get("mop", 0),
            "product_name": source.get("product_name", ""),
            "property": [{
                "cityid": [selected_city],
                "lowest_emi": source.get("lowest_emi", 0),
                "offer_price": source.get("offer_price", 0),
                "score": source.get("score", 0),
            }],
            "products": source.get("products", [])
        }

        products.append(product_data)

        # Collect filter attributes
        for key, value in source.items():
            if key.startswith("attribute_") and key.endswith("_value") and value:
                attr_key = key.replace("_value", "")
                if attr_key not in final_filters:
                    final_filters[attr_key] = set()
                attr_id = source.get(attr_key, "")
                if value and attr_id:
                    final_filters[attr_key].add((str(value), str(attr_id)))

    # Format filters
    formatted_filters = {}
    for key, items in final_filters.items():
        unique_items = []
        seen_ids = set()
        for val, fid in items:
            if fid not in seen_ids:
                unique_items.append({"name": val, "id": fid})
                seen_ids.add(fid)
        if unique_items:
            formatted_filters[key] = unique_items

    # BUG FIX 2: Calculate accurate pagination info
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    has_next = page < total_pages
    has_prev = page > 1

    return {
        "data": {
            "PostV1Productlist": {
                "status": True,
                "message": "Success",
                "data": {
                    "products": products,
                    "totalrecords": total,
                    "suggested_search_keyword": "*",
                    "filters": [{"attributes": formatted_filters}] if formatted_filters else [],
                    # BUG FIX 2: Add pagination metadata
                    "pagination": {
                        "current_page": page,
                        "page_size": page_size,
                        "total_pages": total_pages,
                        "total_products": total,
                        "products_on_page": len(products),
                        "has_next_page": has_next,
                        "has_previous_page": has_prev,
                    },
                    "emi_slider_range": {
                        "min": min(emi_values) if emi_values else 0,
                        "max": max(emi_values) if emi_values else 0
                    }
                }
            }
        }
    }


# =================== SEARCH API ENDPOINT ===================
@app.route("/api/mall_plp_search", methods=["POST"])
@log_api_call
def mall_plp_search():
    """
    Product Listing Page Search API.

    Request body:
    {
        "query": "samsung phone",
        "city_id": "1003",
        "filters": {"attribute_color": "black"},
        "page": 1,
        "page_size": 26,
        "sort_by": {"by": "emi", "order": "asc"}
    }

    BUG FIXES APPLIED:
    1. Enhanced fuzzy matching with ES autosuggest fallback
    2. Proper pagination returning 26 unique SKUs per page
    """
    try:
        data = request.get_json() or {}

        # Extract parameters
        query = data.get("query", "").strip()
        city_id = data.get("city_id", "0")
        filters = data.get("filters", {})

        # BUG FIX 2: Pagination parameters
        page = max(1, int(data.get("page", 1)))
        page_size = min(MAX_PAGE_SIZE, max(1, int(data.get("page_size", DEFAULT_PAGE_SIZE))))

        sort_by = data.get("sort_by", {})
        base_category = data.get("base_category", "")
        chip = data.get("chip", "")

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        original_query = query

        # Combine query with base_category and chip if present
        combined_query = query
        if base_category:
            combined_query = f"{base_category} {query}"
        if chip:
            combined_query = f"{combined_query} {chip}"

        # BUG FIX 1: Enhanced query correction
        corrected_query = correct_query(combined_query)

        # Parse price/EMI from query
        cleaned_query, price_filter = parse_price_from_query(corrected_query)

        # Resolve category
        query_n = normalize(cleaned_query)
        mapped_category = CATEGORY_CANONICAL.get(query_n)
        if not mapped_category:
            for cat_key, syns in BUSINESS_SYNONYMS.items():
                if query_n in [normalize(s) for s in syns]:
                    mapped_category = cat_key
                    break

        # Handle EMI filters from request
        emi_range = None
        if "emi" in filters:
            emi_filter = filters.pop("emi")
            if isinstance(emi_filter, dict):
                emi_range = {}
                if "min" in emi_filter:
                    emi_range["gte"] = emi_filter["min"]
                if "max" in emi_filter:
                    emi_range["lte"] = emi_filter["max"]

        # BUG FIX 2: Build query with collapse for unique SKUs per page
        query_body = build_search_query_with_collapse(
            user_query=cleaned_query,
            filters=filters,
            city_id=city_id,
            mapped_category=mapped_category,
            price_filter=price_filter,
            emi_range=emi_range,
            page=page,
            page_size=page_size,
        )

        # Apply custom sorting
        if sort_by and "by" in sort_by:
            order = sort_by.get("order", "asc")
            if sort_by["by"] == "emi":
                query_body["sort"] = [{"lowest_emi": {"order": order}}]
            elif sort_by["by"] == "price":
                query_body["sort"] = [{"mop": {"order": order}}]

        # Execute search
        response = es.search(
            index=PRODUCT_INDEX_NAME,
            body=query_body
        )

        # BUG FIX 2: Process response with pagination info
        result = process_search_response(
            response=response,
            city_id=city_id,
            page=page,
            page_size=page_size
        )

        # Add query correction info
        query_was_corrected = (normalize(combined_query) != normalize(corrected_query))
        result["data"]["PostV1Productlist"]["data"]["original_query"] = original_query
        result["data"]["PostV1Productlist"]["data"]["corrected_query"] = corrected_query
        result["data"]["PostV1Productlist"]["data"]["query_corrected"] = query_was_corrected

        if query_was_corrected:
            result["data"]["PostV1Productlist"]["data"]["suggested_search_keyword"] = corrected_query

        logger.info(f"Search: query='{query}', corrected='{corrected_query}', "
                   f"page={page}, results={len(result['data']['PostV1Productlist']['data']['products'])}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Search error: {traceback.format_exc()}")
        return jsonify({"message": "Internal server error", "error": str(e)}), 500


# =================== AUTOSUGGEST API ENDPOINT ===================
@app.route("/api/mall_autosuggest", methods=["POST"])
@log_api_call
def mall_autosuggest():
    """
    Autosuggest API for search autocomplete.

    BUG FIX 1: Uses enhanced fuzzy matching for typo tolerance.
    """
    try:
        data = request.get_json() or {}
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        # BUG FIX 1: Enhanced query correction
        corrected_query = correct_query(query)
        expanded_terms = expand_search_terms(corrected_query)

        # Get business suggestions
        n_query = normalize(corrected_query)
        business_suggestions = []

        if n_query in BUSINESS_AUTOSUGGEST:
            business_suggestions.extend(BUSINESS_AUTOSUGGEST[n_query])

        for key in BUSINESS_AUTOSUGGEST:
            if n_query and (n_query in key or key in n_query):
                business_suggestions.extend(BUSINESS_AUTOSUGGEST[key])

        # ES autosuggest query
        should_clauses = []
        for term in expanded_terms:
            if len(term) >= 2:
                should_clauses.extend([
                    {"match_phrase_prefix": {"value": {"query": term, "boost": 7}}},
                    {"match": {"value": {"query": term, "fuzziness": "AUTO", "boost": 4}}},
                ])

        es_response = es.search(
            index=AUTOSUGGEST_INDEX_NAME,
            body={
                "query": {"bool": {"should": should_clauses, "minimum_should_match": 1}},
                "size": 50,
                "_source": ["value", "type"]
            }
        )

        es_keywords = []
        seen = set()
        for hit in es_response.get("hits", {}).get("hits", []):
            value = hit["_source"].get("value", "").strip()
            if value and value.lower() not in seen:
                es_keywords.append(value)
                seen.add(value.lower())

        # Combine and deduplicate
        all_keywords = list(dict.fromkeys(business_suggestions + es_keywords))[:15]

        # Get chips for category
        chips, filter_text = [], ""
        mapped_category = CATEGORY_CANONICAL.get(n_query)
        if not mapped_category:
            for cat_key, syns in BUSINESS_SYNONYMS.items():
                if n_query in [normalize(s) for s in syns]:
                    mapped_category = cat_key
                    break

        if mapped_category and mapped_category in CATEGORY_HARDCODED_CHIPS:
            chips, filter_text = CATEGORY_HARDCODED_CHIPS[mapped_category]

        return jsonify({
            "message": "success",
            "response": {
                "keywords": all_keywords,
                "chips": chips,
                "filter_text": filter_text,
                "language": "english",
                "corrected_query": corrected_query if corrected_query != query else None
            }
        }), 200

    except Exception as e:
        logger.error(f"Autosuggest error: {traceback.format_exc()}")
        return jsonify({"message": "Internal server error", "error": str(e)}), 500


# =================== HEALTH CHECK ===================
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        es_health = es.ping()
        return jsonify({
            "status": "healthy",
            "elasticsearch": "connected" if es_health else "disconnected"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# =================== MAIN ===================
if __name__ == "__main__":
    logger.info("Starting BajajMall Search API...")
    logger.info(f"ES Index: {PRODUCT_INDEX_NAME}")
    logger.info(f"Correction pool size: {len(CORRECTION_POOL)}")
    app.run(host="0.0.0.0", port=8007, debug=True)
