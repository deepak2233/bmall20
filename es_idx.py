# elastic_search_indexing.py
# ES 8.17.4-ready indexing script for Bajaj Mall

import json
import math
import os
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
import subprocess
import requests

# ===================== CONFIG =====================
# DATA_FILE = "processed_products.json"
DATA_FILE = "/datadrive1/deepak/new_mall_pipeline/data/processed_data/processed_products_latest.json"

PRODUCT_INDEX = "bajajmall_products_s3_esidx3_15102025_chkpt"
CATEGORY_INDEX = "bajajmall_categories_s3_esidx3_15102025_chkpt"
AUTOSUGGEST_INDEX = "bajajmall_autosuggest_s3_esidx3_15102025_chkpt"
BRAND_INDEX = "bajajmall_brands_s3_esidx3_30092025_chkpt"  # NEW

# Where to save attribute id->name mapping (your app reads this)
ATTRIBUTE_ID_NAME_MAP_FILE = "attribute_id_name_mapping_esidx.json"

# IMPORTANT: This path is relative to the ES **config** dir on the node/container,
# not this Python script’s folder. Mount your synonyms file there.
SYNONYM_PATH_IN_ES_CONFIG = "analysis/synonym-set.txt"

ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")
MIN_RANK_FEATURE = 1e-3

BOOL_KEYS = [
    "zero_dp_flag",
    "new_launch_flag",
    "most_viewed_flag",
    "top_seller_flag",
    "model_city_flag",
    "phone_setup",
    "exchange_flag",
    "installation_flag",
]

# ========== LOGGING ==========
def setup_logging():
    Path("logs").mkdir(parents=True, exist_ok=True)
    log_file = f"logs/es_indexing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Console too
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)
    logging.info("==== Starting Elasticsearch Indexing ====")
    logging.info(f"ES_HOST={ES_HOST}")
    logging.info(f"DATA_FILE={DATA_FILE}")
    return log_file

log_file = setup_logging()

# ===================== ES CLIENT =====================
# If your ES uses security (default in 8.x), you may need basic_auth or API key:
# es = Elasticsearch(ES_HOST, basic_auth=("elastic", "password"), verify_certs=False)
# ===================== ES CLIENT =====================
es = Elasticsearch(
    hosts=[ES_HOST],
    request_timeout=120,      
    max_retries=5,
    retry_on_timeout=True
)


try:
    if not es.ping():
        raise ConnectionError("Could not connect to Elasticsearch")
    logging.info("Successfully connected to Elasticsearch")
except Exception as e:
    logging.error(f"Elasticsearch connection error: {str(e)}")
    raise

# ===================== MAPPINGS =====================

PRODUCT_MAPPING: Dict[str, Any] = {
    "settings": {
        # Safe defaults (tune shards/replicas for prod)
        "number_of_shards": 1,
        "number_of_replicas": 0,

        "index.mapping.total_fields.limit": 2000,
        # needed because edge_ngram min=2, max=15  -> diff 13
        "index.max_ngram_diff": 13,

        "analysis": {
            "filter": {
                # Use synonym (index) + synonym_graph (search)
                "bfl_synonyms_index": {
                    "type": "synonym",
                    "synonyms_path": SYNONYM_PATH_IN_ES_CONFIG,
                },
                "bfl_synonyms_search": {
                    "type": "synonym_graph",
                    "synonyms_path": SYNONYM_PATH_IN_ES_CONFIG,
                },
                "english_stemmer": {"type": "stemmer", "language": "english"},
                "edge_ngram_filter": {"type": "edge_ngram", "min_gram": 2, "max_gram": 15},
            },
            "analyzer": {
                "custom_indexing_analyzer": {
                    "type": "custom",
                    "tokenizer": "whitespace",
                    "filter": [
                        "lowercase",
                        "bfl_synonyms_index",
                        "english_stemmer",
                        "edge_ngram_filter",
                    ],
                },
                "custom_search_analyzer": {
                    "type": "custom",
                    "tokenizer": "whitespace",
                    "filter": ["lowercase", "bfl_synonyms_search", "english_stemmer"],
                },
            },
        },
    },
    "mappings": {
        "dynamic_templates": [
            {
                "attribute_keywords": {
                    "match": "attribute_*_value",
                    "mapping": {
                        "type": "keyword",
                        "ignore_above": 256,
                        "fields": {
                            "analyzed": {
                                "type": "text",
                                "analyzer": "custom_indexing_analyzer",
                                "search_analyzer": "custom_search_analyzer",
                            }
                        },
                    },
                }
            },
            {
                "attribute_ids": {
                    "match": "attribute_*",
                    "unmatch": "*_value",
                    "mapping": {"type": "keyword", "ignore_above": 256},
                }
            },
            {
                "strings_as_keywords": {
                    "match_mapping_type": "string",
                    "mapping": {
                        "type": "keyword",
                        "ignore_above": 256,
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "custom_indexing_analyzer",
                                "search_analyzer": "custom_search_analyzer",
                            }
                        },
                    },
                }
            },
        ],
        "properties": {
            "search_field": {
                "type": "text",
                "analyzer": "custom_indexing_analyzer",
                "search_analyzer": "custom_search_analyzer",
            },
            "product_name": {
                "type": "text",
                "analyzer": "custom_indexing_analyzer",
                "search_analyzer": "custom_search_analyzer",
            },

            "actual_category": {"type": "keyword"},
            "top_level_category_name": {"type": "keyword"},
            "asset_category_name": {"type": "keyword"},

            "mrp": {"type": "float"},
            "mop": {"type": "float"},
            "modelid": {"type": "keyword"},
            "manufacturer_desc": {"type": "keyword"},

            "created_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},
            "updated_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},

            "city_offers": {
                "type": "nested",
                "properties": {
                    "cityid": {"type": "keyword"},
                    "transaction_count": {"type": "integer"},
                    "lowest_emi": {"type": "float"},
                    "offer_price": {"type": "float"},
                    "score": {"type": "float"},
                    "promotion_score": {"type": "float"},
                    "ty_page_count": {"type": "integer"},
                    "one_emi_off": {"type": "integer"},
                    "24_month_emi": {"type": "integer"},
                    "pdp_view_count": {"type": "integer"},
                    "off_percentage": {"type": "float"},
                    "zero_dp_flag": {"type": "boolean"},
                    "new_launch_flag": {"type": "boolean"},
                    "most_viewed_flag": {"type": "boolean"},
                    "top_seller_flag": {"type": "boolean"},
                    "highest_tenure": {"type": "integer"},
                    "model_city_flag": {"type": "boolean"},
                    "phone_setup": {"type": "boolean"},
                    "exchange_flag": {"type": "boolean"},
                    "installation_flag": {"type": "boolean"},
                },
            },

            "products": {
                "type": "nested",
                "properties": {
                    "sku": {"type": "keyword"},
                    "name": {
                        "type": "text",
                        "analyzer": "custom_indexing_analyzer",
                        "search_analyzer": "custom_search_analyzer",
                    },
                    "image": {"type": "keyword"},
                    "attribute_set_id": {"type": "keyword"},
                    "brand_id": {"type": "keyword"},
                    "color_hex_code": {"type": "keyword"},
                    "color_label": {"type": "keyword"},
                    "keyword": {"type": "text"},
                    "product_url": {"type": "keyword"},
                    "attribute_swatch_color": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "keyword"},
                            "name": {"type": "keyword"},
                            "value": {"type": "keyword"},
                        },
                    },
                },
            },

            # rank_feature (for query scoring) + numeric clones for sorting
            "popularity_score": {"type": "rank_feature"},
            "popularity_score_num": {"type": "scaled_float", "scaling_factor": 1000},

            "recency_boost": {"type": "rank_feature"},
            "recency_boost_num": {"type": "scaled_float", "scaling_factor": 1000000},
        },
    },
}

CATEGORY_MAPPING: Dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "value": {"type": "keyword"},
            "keyword": {"type": "text"},
        }
    },
}

AUTOSUGGEST_MAPPING: Dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        # edge_ngram 1..20 -> diff 19
        "index.max_ngram_diff": 19,
        "analysis": {
            "analyzer": {
                "autosuggest_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "edge_ngram_filter"],
                }
            },
            "filter": {
                "edge_ngram_filter": {
                    "type": "edge_ngram",
                    "min_gram": 1,
                    "max_gram": 20,
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "type": {"type": "keyword"},
            "value": {"type": "text", "analyzer": "autosuggest_analyzer"},
        }
    },
}

BRAND_MAPPING: Dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        # include the same analysis block used by the product index
        "analysis": {
            "filter": {
                "bfl_synonyms_index": {
                    "type": "synonym",
                    "synonyms_path": SYNONYM_PATH_IN_ES_CONFIG,
                },
                "bfl_synonyms_search": {
                    "type": "synonym_graph",
                    "synonyms_path": SYNONYM_PATH_IN_ES_CONFIG,
                },
                "english_stemmer": {"type": "stemmer", "language": "english"},
                "edge_ngram_filter": {"type": "edge_ngram", "min_gram": 2, "max_gram": 15},
            },
            "analyzer": {
                "custom_indexing_analyzer": {
                    "type": "custom",
                    "tokenizer": "whitespace",
                    "filter": [
                        "lowercase",
                        "bfl_synonyms_index",
                        "english_stemmer",
                        "edge_ngram_filter",
                    ],
                },
                "custom_search_analyzer": {
                    "type": "custom",
                    "tokenizer": "whitespace",
                    "filter": ["lowercase", "bfl_synonyms_search", "english_stemmer"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "brand": {
                "type": "keyword",
                "fields": {
                    "text": {
                        "type": "text",
                        "analyzer": "custom_indexing_analyzer",
                        "search_analyzer": "custom_search_analyzer"
                    }
                }
            },
            "product_count": {"type": "integer"},
            "categories": {"type": "keyword"}
        }
    }
}


# ===================== HELPERS =====================

def parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ["1", "true", "yes"]
    return False


def city_offers_dict_to_list(city_offers_dict) -> List[Dict[str, Any]]:
    if not isinstance(city_offers_dict, dict):
        return []
    return [dict(offer, cityid=cityid) for cityid, offer in city_offers_dict.items()]


def transform_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize, coerce types, compute scores, and make nested lists."""
    if not doc:
        return {}

    offers_list = city_offers_dict_to_list(doc.get("city_offers", {}))
    total_txn, total_views = 0, 0

    for offer in offers_list:
        # booleans
        for key in BOOL_KEYS:
            if key in offer:
                offer[key] = parse_bool(offer[key])

        # numeric coercion
        numeric_fields = [
            "transaction_count", "lowest_emi", "offer_price", "score",
            "promotion_score", "ty_page_count", "one_emi_off", "24_month_emi",
            "pdp_view_count", "off_percentage", "highest_tenure"
        ]
        for f in numeric_fields:
            if f in offer and offer[f] not in (None, '') and not isinstance(offer[f], (int, float)):
                try:
                    # keep ints as ints if possible
                    s = str(offer[f])
                    offer[f] = float(s) if "." in s else int(s)
                except (ValueError, TypeError):
                    offer[f] = 0

        total_txn += int(offer.get("transaction_count", 0) or 0)
        total_views += int(offer.get("pdp_view_count", 0) or 0)

    doc["city_offers"] = offers_list

    # products as list
    if isinstance(doc.get("products"), dict):
        doc["products"] = list(doc["products"].values())

    # popularity score (rank_feature + numeric clone)
    score = total_views + (total_txn * 5)
    if score <= 0:
        score = MIN_RANK_FEATURE
    doc["popularity_score"] = float(score)
    doc["popularity_score_num"] = float(score)

    # recency boost (rank_feature + numeric clone)
    try:
        updated_at_str = doc.get("updated_at", "2000-01-01 00:00:00")
        if isinstance(updated_at_str, datetime):
            updated_at = updated_at_str
        else:
            updated_at = datetime.strptime(str(updated_at_str), "%Y-%m-%d %H:%M:%S")

        days = max((datetime.now() - updated_at).days, 0)
        recency = 1 / (1 + math.log1p(days))
        if recency <= 0:
            recency = MIN_RANK_FEATURE
        recency = round(float(recency), 6)
        doc["recency_boost"] = recency
        doc["recency_boost_num"] = recency
    except (ValueError, TypeError) as e:
        logging.warning(f"Error calculating recency boost: {str(e)}")
        doc["recency_boost"] = MIN_RANK_FEATURE
        doc["recency_boost_num"] = MIN_RANK_FEATURE

    # copy a product_name for display/search if missing
    if doc.get("products"):
        doc.setdefault("product_name", doc["products"][0].get("name", ""))

    # coerce root numeric fields
    for root_f in ["mrp", "mop"]:
        if root_f in doc and doc[root_f] not in (None, '') and not isinstance(doc[root_f], (int, float)):
            try:
                doc[root_f] = float(doc[root_f])
            except (ValueError, TypeError):
                doc[root_f] = 0.0
    
    # ======== ENHANCED SEARCH_FIELD POPULATION ========
    # Build a comprehensive search_field that includes all relevant product attributes
    search_field_parts = []
    
    # Add the product name
    product_name = doc.get("product_name", "")
    if product_name:
        search_field_parts.append(product_name)
    
    # Add manufacturer info
    manufacturer = doc.get("manufacturer_desc", "")
    if manufacturer:
        search_field_parts.append(manufacturer)
    
    # Add category info
    categories = [
        doc.get("actual_category", ""),
        doc.get("top_level_category_name", ""),
        doc.get("asset_category_name", "")
    ]
    search_field_parts.extend([c for c in categories if c])
    
    # Extract and add all attribute values (especially for iPhones)
    for key, value in doc.items():
        if key.startswith("attribute_") and key.endswith("_value") and value:
            search_field_parts.append(str(value))
    
    # Add all product variations and their attributes
    products = doc.get("products", [])
    for prod in products:
        if isinstance(prod, dict):
            # Add product name
            if "name" in prod:
                search_field_parts.append(prod["name"])
            
            # Add all attributes of each product variant
            for k, v in prod.items():
                if k not in ["name", "sku", "image", "product_url"] and v:
                    if isinstance(v, dict):
                        # Handle nested attributes like color
                        for sub_k, sub_v in v.items():
                            if sub_v and isinstance(sub_v, str):
                                search_field_parts.append(str(sub_v))
                    elif isinstance(v, list):
                        # Handle list attributes
                        for item in v:
                            if isinstance(item, dict):
                                for sub_k, sub_v in item.items():
                                    if sub_v and isinstance(sub_v, str):
                                        search_field_parts.append(str(sub_v))
                            elif item:
                                search_field_parts.append(str(item))
                    else:
                        search_field_parts.append(str(v))
    
    # Create the combined search_field with all parts
    if search_field_parts:
        # Filter out empty strings and join with spaces
        search_field_parts = [p for p in search_field_parts if p and p.strip()]
        doc["search_field"] = " ".join(search_field_parts)
    
    return doc

def recreate_index(index_name: str, mapping: Dict[str, Any]) -> bool:
    try:
        if es.indices.exists(index=index_name, request_timeout=60):
            es.indices.delete(index=index_name, ignore=[404], request_timeout=60)
            logging.info(f"Deleted existing index: {index_name}")

        es.indices.create(index=index_name, body=mapping, request_timeout=120)
        logging.info(f"Created index: {index_name}")
        return True
    except Exception as e:
        logging.error(f"Error creating index {index_name}: {str(e)}")
        return False



def bulk_index(index_name: str, data: List[Dict[str, Any]]) -> int:
    if not data:
        logging.warning(f"No data to index for {index_name}")
        return 0

    actions = ({"_index": index_name, "_source": d} for d in data)
    try:
        success, _ = bulk(es, actions, refresh="wait_for", request_timeout=120)
        logging.info(f"Indexed {success} docs into {index_name}")
        print(f"✅ Indexed {success} docs into {index_name}")
        return success
    except BulkIndexError as e:
        logging.error("BulkIndexError during indexing!")
        for err in e.errors[:10]:
            logging.error(json.dumps(err, indent=2))
        raise
    except Exception as ex:
        logging.error(f"Unexpected exception: {str(ex)}")
        raise


def verify_index_with_curl(index_name: str) -> bool:
    """Optional: Verify index using curl commands (handy for quick sanity)."""
    try:
        # Check exists
        result = subprocess.run(
            ["curl", "-s", "-X", "GET", f"{ES_HOST}/{index_name}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logging.error(f"Failed to check index {index_name} with curl: {result.stderr}")
            return False
        index_info = json.loads(result.stdout or "{}")
        if index_name not in index_info:
            logging.error(f"Index {index_name} not found in curl response")
            return False

        # Mapping
        result = subprocess.run(
            ["curl", "-s", "-X", "GET", f"{ES_HOST}/{index_name}/_mapping"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logging.error(f"Failed to get mapping for {index_name}: {result.stderr}")
            return False
        _ = json.loads(result.stdout or "{}")

        logging.info(f"Successfully verified index {index_name} with curl")
        return True
    except Exception as e:
        logging.error(f"Error verifying index with curl: {str(e)}")
        return False


def verify_index_with_requests(index_name: str) -> bool:
    """Optional: Verify index using requests library."""
    try:
        resp = requests.get(f"{ES_HOST}/{index_name}")
        if resp.status_code != 200:
            logging.error(f"Failed to check index {index_name}: {resp.text}")
            return False

        info = resp.json()
        if index_name not in info:
            logging.error(f"Index {index_name} not found in response")
            return False

        resp = requests.get(f"{ES_HOST}/{index_name}/_mapping")
        if resp.status_code != 200:
            logging.error(f"Failed to get mapping for {index_name}: {resp.text}")
            return False

        logging.info(f"Successfully verified index {index_name} with requests")
        return True
    except Exception as e:
        logging.error(f"Error verifying index with requests: {str(e)}")
        return False


def build_and_save_attribute_id_name_mapping(
    transformed_data: List[Dict[str, Any]],
    output_path: str = ATTRIBUTE_ID_NAME_MAP_FILE,
):
    """
    Build a mapping like:
      'attribute_brand_new': {'4978': 'Sony', ...}
    from pairs: attribute_* and attribute_*_value
    """
    attribute_id_name_mapping: Dict[str, Dict[str, str]] = defaultdict(dict)

    for product in transformed_data:
        for key, value in product.items():
            if key.startswith("attribute_") and not key.endswith("_value"):
                value_key = f"{key}_value"
                attr_id = str(value) if value is not None else ""
                attr_name = product.get(value_key)
                if attr_id and attr_name:
                    attribute_id_name_mapping[key][attr_id] = str(attr_name)

    # Write file (avoid makedirs('') when no dir)
    dirn = os.path.dirname(output_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(dict(attribute_id_name_mapping), fout, indent=2, ensure_ascii=False)
    logging.info(f"Attribute ID-name mapping saved to {output_path}")


def reload_search_analyzers(index_name: str):
    """Reload analyzers to pick up synonym file changes (ES 7.3+)."""
    try:
        es.indices.reload_search_analyzers(index=index_name)
        logging.info(f"Reloaded search analyzers for {index_name}")
    except Exception as e:
        logging.warning(f"Could not reload analyzers for {index_name}: {e}")


# ======== BRAND DOCS (NEW) ========
def build_brand_docs(transformed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build brand docs from attribute_brand_new_value or manufacturer_desc,
    along with product_count and set of categories.
    """
    stats: Dict[str, Tuple[int, Set[str]]] = {}
    for d in transformed_data:
        brand = (d.get("attribute_brand_new_value") or d.get("manufacturer_desc") or "").strip()
        if not brand:
            continue
        cat = d.get("actual_category") or ""
        cnt, cats = stats.get(brand, (0, set()))
        cnt += 1
        if cat:
            cats.add(cat)
        stats[brand] = (cnt, cats)

    brand_docs = []
    for b, (cnt, cats) in stats.items():
        brand_docs.append({
            "brand": b,
            "product_count": int(cnt),
            "categories": sorted(list(cats)) if cats else []
        })
    logging.info(f"Prepared {len(brand_docs)} brand docs")
    return brand_docs


# ===================== MAIN =====================

if __name__ == "__main__":
    try:
        # --- Load data ---
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not raw_data:
            logging.error("No data found in the input file")
            raise ValueError("Empty input data")

        logging.info(f"Loaded {len(raw_data)} records from {DATA_FILE}")

        transformed_data = [transform_document(doc) for doc in raw_data]
        logging.info(f"Transformed {len(transformed_data)} records")

        # --- 1) Products ---
        if recreate_index(PRODUCT_INDEX, PRODUCT_MAPPING):
            bulk_index(PRODUCT_INDEX, transformed_data)
            verify_index_with_curl(PRODUCT_INDEX)
            verify_index_with_requests(PRODUCT_INDEX)
            # If you updated the synonyms file, reload analyzers:
            reload_search_analyzers(PRODUCT_INDEX)

        # --- 2) Categories ---
        unique_categories = [
            {"keyword": c.replace(" ", "_"), "value": c}
            for c in {d.get("actual_category", "") for d in transformed_data if d.get("actual_category")}
        ]
        logging.info(f"Prepared {len(unique_categories)} category docs")
        if recreate_index(CATEGORY_INDEX, CATEGORY_MAPPING):
            bulk_index(CATEGORY_INDEX, unique_categories)
            verify_index_with_curl(CATEGORY_INDEX)
            verify_index_with_requests(CATEGORY_INDEX)

        # --- 3) Autosuggest ---
        autosuggest_data: List[Dict[str, str]] = []
        seen = set()
        for d in transformed_data:
            for p in d.get("products", []) or []:
                pname = p.get("name")
                if pname and pname not in seen:
                    autosuggest_data.append({"type": "product", "value": pname})
                    seen.add(pname)

        for c in {d.get("actual_category", "") for d in transformed_data if d.get("actual_category")}:
            autosuggest_data.append({"type": "category", "value": c})

        for b in {d.get("attribute_brand_new_value", "") for d in transformed_data if d.get("attribute_brand_new_value")}:
            autosuggest_data.append({"type": "brand", "value": b})

        logging.info(f"Prepared {len(autosuggest_data)} autosuggest docs")
        if recreate_index(AUTOSUGGEST_INDEX, AUTOSUGGEST_MAPPING):
            bulk_index(AUTOSUGGEST_INDEX, autosuggest_data)
            verify_index_with_curl(AUTOSUGGEST_INDEX)
            verify_index_with_requests(AUTOSUGGEST_INDEX)

        # --- 4) Brands (NEW) ---
        brand_docs = build_brand_docs(transformed_data)
        if recreate_index(BRAND_INDEX, BRAND_MAPPING):
            bulk_index(BRAND_INDEX, brand_docs)
            verify_index_with_curl(BRAND_INDEX)
            verify_index_with_requests(BRAND_INDEX)

        # --- 5) Attribute ID-name mapping file ---
        build_and_save_attribute_id_name_mapping(transformed_data, ATTRIBUTE_ID_NAME_MAP_FILE)

        logging.info("==== Indexing Completed Successfully ====")
        print(f"🎉 Indexing Completed! Logs saved at {log_file}")

    except Exception as e:
        logging.error(f"Fatal error in indexing process: {str(e)}")
        raise
