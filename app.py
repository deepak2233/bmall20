import sys
import os

# Add required paths for imports - allows running directly: python3 app_v2_corrected_query.py
#_paths_to_add = [
#    "/datadrive1/deepak",
#    "/datadrive1/deepak/new_mall_pipeline",
#    "/datadrive1/deepak/new_mall_pipeline/autosuggest",
#    "/datadrive1/deepak/new_mall_pipeline/utils",
#]
#for _p in _paths_to_add:
#    if _p not in sys.path:
#        sys.path.insert(0, _p)

from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import traceback
import json
import logging
import re
from rapidfuzz import process as fuzzproc, fuzz
from collections import defaultdict
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

from utils import (
    CATEGORY_CANONICAL,
    BUSINESS_SYNONYMS,
    ALL_ATTRIBUTE_FILTERS,
    ALL_ATTRIBUTE_FILTERS_Name,
    CATEGORY_FILTER_EXCLUSIONS
)
from query_enhancer import get_enhanced_parser  # NEW: Enhanced query parser for complex queries
from used_car_handler import check_used_car_query, build_used_car_response  # Used car search handler

# =================== CONFIGURATION ===================
# PRODUCT_INDEX_NAME = "bajajmall_products_s3_esidx3_15102025_chkpt_up26"
# CATEGORY_INDEX_NAME = "bajajmall_categories_s3_esidx3_159102025_chkpt_up26"
# AUTOSUGGEST_INDEX_NAME = "bajajmall_autosuggest_s3_esidx3_15102025_chkpt_up26"
# IMAGE_DOMAIN = "https://mc.bajajfinserv.in/media/catalog/product_up26"


# PRODUCT_INDEX_NAME = "bajajmall_products_s3_esidx3_22012026"
# CATEGORY_INDEX_NAME = "bajajmall_categories_s3_esidx3_22012026"
# AUTOSUGGEST_INDEX_NAME = "bajajmall_autosuggest_s3_esidx3_22012026"

# PRODUCT_INDEX_NAME = "bajajmall_products_s3_esidx3_04032026_chkpt_up26"
# CATEGORY_INDEX_NAME = "bajajmall_categories_s3_esidx3_04032026_chkpt_up26"
# AUTOSUGGEST_INDEX_NAME = "bajajmall_autosuggest_s3_esidx3_04032026_chkpt_up26"
# BRAND_INDEX_NAME = "bajajmall_brands_s3_esidx3_04032026_chkpt_up26"  # NEW

PRODUCT_INDEX_NAME = os.getenv("PRODUCT_INDEX_NAME", "bajajmall_products_read")
CATEGORY_INDEX_NAME = os.getenv("CATEGORY_INDEX_NAME", "bajajmall_categories_read")
AUTOSUGGEST_INDEX_NAME = os.getenv("AUTOSUGGEST_INDEX_NAME", "bajajmall_autosuggest_read")
BRAND_INDEX_NAME = os.getenv("BRAND_INDEX_NAME", "bajajmall_brands_read")


# PRODUCT_INDEX_NAME = "bajajmall_products_s3_esidx3_30032026"
# CATEGORY_INDEX_NAME = "bajajmall_categories_s3_esidx3_30032026"
# AUTOSUGGEST_INDEX_NAME = "bajajmall_autosuggest_s3_esidx3_30032026"
IMAGE_DOMAIN = "https://mc.bajajfinserv.in/media/catalog/product_up26"


# =================== USED CAR SEARCH FEATURE FLAG ===================
# Set to True to enable used car search, False to disable
USED_CAR_SEARCH_ENABLED = True

# =================== PRODUCTION TIMEOUT PROTECTION ===================
# Maximum allowed time for API request (in seconds)
# If request exceeds this, skip fallbacks and return partial results
MAX_REQUEST_TIME_SECONDS = 5.0  # Production limit is 6s, we use 5s for safety margin
ES_TIMEOUT_SECONDS = 3  # Elasticsearch query timeout
FALLBACK_TIMEOUT_SECONDS = 2  # Fallback queries timeout (shorter)

# =================== DEALER SEARCH FEATURE FLAG ===================
# Set to True to enable dealer search, False to disable
DEALER_SEARCH_ENABLED = True
DEALER_INDEX_NAME = "bajajmall_dealers_index"

# =================== COMPARE SEARCH FEATURE FLAG ===================
# Set to True to enable compare search, False to disable
COMPARE_SEARCH_ENABLED = True

# Setup logging
logging.basicConfig(
    filename="app_v2.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
app.json.sort_keys = False
es = Elasticsearch("http://localhost:9200")

# =================== DEALER SEARCH INITIALIZATION ===================
# Initialize dealer search using centralized handler
from dealer_handler import (
    init_dealer_search,
    is_dealer_search_ready,
    execute_dealer_search,
    get_dealer_results_for_hybrid,
    classify_dealer_intent,
    get_search_intent_class,
    get_dealer_autosuggest,
    clean_dealer_response
)

if DEALER_SEARCH_ENABLED:
    if init_dealer_search(es, DEALER_INDEX_NAME):
        logger.info("✅ Dealer search initialized via handler")
    else:
        logger.error("⚠️ Failed to initialize dealer search")
        DEALER_SEARCH_ENABLED = False


def is_dealer_search_enabled():
    """Check if dealer search is enabled and initialized"""
    return DEALER_SEARCH_ENABLED and is_dealer_search_ready()


# =================== COMPARE SEARCH INITIALIZATION ===================
# Initialize compare search using centralized handler
from compare_search_handler import (
    init_compare_search,
    is_compare_search_ready,
    execute_compare_search
)

if COMPARE_SEARCH_ENABLED:
    if init_compare_search(es, PRODUCT_INDEX_NAME):
        logger.info("✅ Compare search initialized via handler")
    else:
        logger.error("⚠️ Failed to initialize compare search")
        COMPARE_SEARCH_ENABLED = False


def is_compare_search_enabled():
    """Check if compare search is enabled and initialized"""
    return COMPARE_SEARCH_ENABLED and is_compare_search_ready()


# Load attribute ID to name mapping. Override ATTRIBUTE_ID_NAME_MAP_PATH during
# indexer rollouts to point at the latest attribute_id_name_mapping_<run_id>.json.
ATTRIBUTE_ID_NAME_MAP_PATH = os.getenv(
    "ATTRIBUTE_ID_NAME_MAP_PATH",
    "/data/deepak/new_mall_pipeline_prod/search_codebase_dealesearch_v2/idx/attribute_id_name_mapping_05052026.json",
)
ATTRIBUTE_ID_NAME_MAP_LOADED = False
ATTRIBUTE_ID_NAME_MAP_ERROR = None

try:
    with open(ATTRIBUTE_ID_NAME_MAP_PATH, 'r', encoding='utf-8') as f:
        ATTRIBUTE_ID_NAME_MAP = json.load(f)
    ATTRIBUTE_ID_NAME_MAP_LOADED = True
except Exception as e:
    ATTRIBUTE_ID_NAME_MAP_ERROR = str(e)
    logger.error(f"Failed to load attribute mapping from {ATTRIBUTE_ID_NAME_MAP_PATH}: {e}")
    ATTRIBUTE_ID_NAME_MAP = {}

# Build ALL_ATTRIBUTE_OPTIONS from the two lists
EXCLUDED_ATTRIBUTES = {"attribute_fuelType", "attribute_fuelType_value", "attribute_capacity_new", "attribute_capacity_new_value","attribute_battery_new","attribute_battery_new_value",
                        "attribute_hd_type", "attribute_hd_type_value",
                        "attribute_mount_type_value", "attribute_mount_type"}

ALL_ATTRIBUTE_OPTIONS = [
    {"es_key": es_key, "display_key": display_key}
    for es_key, display_key in zip(ALL_ATTRIBUTE_FILTERS, ALL_ATTRIBUTE_FILTERS_Name)
    if es_key not in EXCLUDED_ATTRIBUTES
]

# Brand names for protection during typo correction
BRAND_NAMES = [
    # Smartphone brands
    "samsung", "samsung mobiles", "apple", "oppo", "vivo", "realme", "redmi", "mi", "xiaomi", 
    "oneplus", "nothing", "motorola", "iqoo", "infinix", "tecno", "nokia", "poco",
    "lava", "google", "honor", "itel", "micromax", "blackberry", "evok",
    # Laptop/PC brands  
    "lenovo", "hp", "dell", "acer", "asus", "lg", "lg mobiles", "panasonic", "panasonic mobiles", "sony",
    # Vehicle brands
    "honda", "hero", "tvs", "bajaj", "royal enfield", "yamaha", "hyundai", "maruti", "tata", "mahindra", "kia", "toyota", "suzuki", "kawasaki", "bmw",
    # Appliance brands
    "ifb", "bosch", "whirlpool", "godrej", "voltas", "voltas beko", "haier", "lloyd", "carrier", "midea",
    "onida", "videocon", "croma", "intex", "daikin", "hitachi", "blue star", "o general", "mitsubishi",
    # TV brands
    "tcl", "vu", "hisense", "thomson", "kodak", "iffalcon", "akai", "bpl",
    # Audio brands
    "philips", "jbl", "bose", "boat", "zebronics", "harman kardon", "marshall", "sennheiser", "mivi",
    # Camera/Gaming brands
    "canon", "nikon", "fujifilm", "gopro", "dji", "insta360", "logitech", "razer", "corsair",
    # Home appliance brands
    "havells", "crompton", "orient", "usha", "prestige", "pigeon", "butterfly", "preethi", "maharaja", "sujata", "vidiem", "atomberg",
    "kent", "livpure", "aquaguard", "pureit", "eureka forbes", "ao smith", "racold",
    # Air cooler brands
    "symphony", "bajaj cooler", "kenstar", "orient cooler",
    # Furniture brands
    "sleepwell", "kurlon", "duroflex", "centuary", "nilkamal", "godrej interio", "urban ladder",
]

# Brand alias mapping: When user searches for a sub-brand, also search in parent brand
# Key = user's search brand, Value = list of manufacturer_desc values to search
BRAND_ALIAS_MAP = {
    "iqoo": ["vivo", "iqoo"],  # iQOO is a Vivo sub-brand
    "redmi": ["xiaomi", "redmi", "mi"],  # Redmi is under Xiaomi
    "poco": ["xiaomi", "poco", "mi"],  # Poco is under Xiaomi
    "mi": ["xiaomi", "mi", "redmi"],  # Mi is Xiaomi
    "xiaomi": ["xiaomi", "mi", "redmi"],  # Xiaomi brand family
}

# =================== PHONE BRAND PRIMARY CATEGORY ===================
# When user searches ONLY a phone brand name (e.g., "oppo", "vivo", "samsung"), 
# prioritize smartphones over other product categories (watches, tablets, etc.)
PHONE_BRAND_PRIMARY_CATEGORY = {
    "oppo", "vivo", "realme", "xiaomi", "redmi", "poco", "mi", "iqoo", "oneplus", "one plus",
    "samsung", "samsung mobiles", "motorola", "nokia", "infinix", "tecno", "lava",
    "micromax", "honor", "itel", "nothing", "google"
}

# =================== BLOCKED BRANDS ===================
# Brands that should be blocked/redirected to other brands in the same category
# Format: {brand_name: {"variations": [...], "default_category": "category_name"}}
# default_category is used when user searches just the brand name (e.g., "vestar" → show other AC brands)
BLOCKED_BRANDS = {
    "vestar": {
        "variations": ["vestar", "vester", "vastar", "vaster", "vestr", "vestaar", "vesttar", 
                       "veestar", "vestor", "vesstar", "vestaar", "vestarr"],
        "default_category": "air conditioner"  # Vestar is primarily an AC brand
    }
}

# Flatten all blocked brand keywords for easy lookup
BLOCKED_BRAND_KEYWORDS = set()
BLOCKED_BRAND_DEFAULT_CATEGORY = {}
for brand, config in BLOCKED_BRANDS.items():
    variations = config.get("variations", [])
    default_cat = config.get("default_category")
    BLOCKED_BRAND_KEYWORDS.add(brand)
    BLOCKED_BRAND_KEYWORDS.update(variations)
    # Map each variation to default category
    if default_cat:
        BLOCKED_BRAND_DEFAULT_CATEGORY[brand] = default_cat
        for v in variations:
            BLOCKED_BRAND_DEFAULT_CATEGORY[v] = default_cat

# =================== AUDIO/EARPHONE KEYWORDS ===================
# Keywords that should map to audio video category
AUDIO_VIDEO_KEYWORDS = {
    # Earphones and earbuds - with comprehensive typo coverage
    "earphone", "earphones", "earphn", "earfon", "ear phone", "ear phones",
    # Common earphone typos: missing letters, wrong letters, phonetic spellings
    "earphon", "earphn", "earfone", "earfon", "earpone", "erphn", "erpone",
    "raphon", "raphne", "raphone", "raphn",  # missing 'e' at start
    "airphon", "airphone", "airphn", "airfon", "airfone",  # 'air' instead of 'ear'
    "erphone", "erphones", "erfone", "erfon",  # 'er' instead of 'ear'
    "earphne", "earphons", "earphnes", "eaphone", "earpho",  # various misspellings
    "earbud", "earbuds", "ear bud", "ear buds", "earbudz", "earbudss",
    # Earbud typos
    "earbud", "erbud", "erbuds", "airbud", "airbuds", "earbds", "earbd",
    "tws", "true wireless", "truly wireless", "wireless earbuds", "wireless earphones",
    "bluetooth earphones", "bluetooth earbuds", "bt earphones",
    # Headphones
    "headphone", "headphones", "headfon", "headset", "headsets",
    "over ear", "on ear", "over-ear", "on-ear",
    # Speakers
    "speaker", "speakers", "bluetooth speaker", "portable speaker",
    "soundbar", "sound bar", "home theater", "home theatre",
    # General audio
    "audio", "sound system", "music system", "stereo",
    # Neckband
    "neckband", "neck band", "neckbands"
}

# =================== AIR FRYER / KITCHEN APPLIANCE KEYWORDS ===================
# Specific keywords for air fryer to avoid matching Dyson Airwrap
AIR_FRYER_KEYWORDS = {
    "air fryer", "airfryer", "air frier", "airfrier", "air friar",
    "fryer", "frier", "deep fryer", "oil free fryer", "oilless fryer"
}

# =================== WATER HEATER / GEYSER KEYWORDS ===================
# Including common typos like "gijar", "gizer", etc.
GEYSER_KEYWORDS = {
    "geyser", "geysers", "geysar", "gayser", "giser", "gizer", "gijar", "gisar",
    "geaser", "geasar", "geizer", "geezer", "gyser", "gaiser", "gaisar",
    "water heater", "waterheater", "water heaters", "hot water heater",
    "instant water heater", "storage water heater", "immersion heater",
    "heater water", "heating water"
}

# =================== SOLAR PANEL KEYWORDS ===================
# Keywords specifically for solar panels/inverters - NOT for solar heaters
# Include "solar" alone and common typos
SOLAR_KEYWORDS = {
    # Main keyword - "solar" alone should show solar panels, not bikes
    "solar",
    # Solar panel/inverter products
    "solar panel", "solar panels", "solar system", "solar power",
    "solar inverter", "solar battery", "solar charger", "solar light", "power backup", "powerbackup",
    "backup inverter", " solar inverter","power back up",
    "solar street light",
    "photovoltaic", "pv panel", "rooftop solar",
    # Solar typos
    "solr", "soler", "solor", "sollar", "solaar", "soalr", "solra",
    "solar pnel", "solar panl", "solarpanel", "solar panal"
}

# =================== SOLAR HEATER KEYWORDS ===================
# Solar heater/geyser queries - should return water heaters, not solar inverters
# These are water heaters powered by solar energy
SOLAR_HEATER_KEYWORDS = {
    "solar heater", "solar water heater", "solar geyser", "solar geysers",
    "solarheater", "solarwaterheater", "solargeyser",
    # Typos
    "soler heater", "solor heater", "solar heter", "solar hiter", "solar hetar",
    "soler water heater", "solor water heater", "solar weter heater",
    "soler geyser", "solor geyser", "solar gayser", "solar geysar"
}

# =================== MIXER GRINDER KEYWORDS ===================
# Keywords for mixer grinder detection (including typos)
MIXER_GRINDER_KEYWORDS = {
    "mixer grinder", "mixergrinder", "mixer-grinder", "mixer griner", "mixer grander",
    "mixi", "mixie", "mixy", "mixey", "mixer", "grinder", "juicer mixer",
    "mixer juicer", "wet grinder", "dry grinder", "blender", "food processor",
    "hand blender", "immersion blender", "mixar", "mixcer", "grindr", "grander"
}

# =================== TWO-WHEELER MODEL NAMES ===================
# These are bike/scooter MODEL names that should force two-wheeler category
# Even if brand+model combo doesn't exist, show two-wheelers
# Include common typos for popular models like Pulsar
# NOTE: These models trigger category detection, but search text handling is in QueryProcessor.process()
TWO_WHEELER_MODELS = {
    # Bajaj models + Pulsar typos (chetak is scooter, handled separately)
    "pulsar", "pulser", "plsar", "plser", "pulsr", "pulsaar", "pulsur", "pulzar",
    "pulsarr", "pulser", "pulsser", "pluser", "plusar",
    "dominar", "platina", "avenger",
    # Bajaj scooter - chetak (removed from here, it's a scooter not bike)
    "chetak",
    # TVS models
    "apache", "apachi", "apche", "ntorq", "jupiter", "raider", "ronin",
    # Hero models
    "splendor", "splendour", "splendr", "passion", "glamour", "xtreme", "xpulse", 
    "destini", "maestro", "pleasure",
    # Honda models
    "activa", "aktivia", "actva", "dio", "shine", "unicorn", "hornet", "sp125", "livo", "cb350",
    # Yamaha models
    "fz", "r15", "mt15", "fascino", "ray", "aerox",
    # Royal Enfield models
    "classic", "bullet", "bullat", "bullt", "hunter", "himalayan", "interceptor", 
    "meteor", "continental", "scram",
    # Suzuki models
    "access", "burgman", "avenis", "gixxer", "hayabusa", "intruder",
    # KTM models
    "duke", "adventure",
    # Ather/Electric - removed "ola" to avoid matching "solar"
    "ather", "revolt", "okinawa",
    # General two-wheeler keywords
    "bike", "bikes", "motorcycle", "motorbike", "scooter", "scooty", "moped",
    "two wheeler", "twowheeler", "2 wheeler", "2wheeler"
}

# =================== BIKE-ONLY MODELS ===================
# These are specifically BIKE models (not scooters) - used to show only bikes
# When user searches for these, show only bikes/motorcycles, not scooters
# NOTE: Scooter models like chetak, activa, jupiter etc. should NOT be here
BIKE_ONLY_MODELS = {
    # Bajaj bikes + all Pulsar variations
    "pulsar", "pulser", "plsar", "plser", "pulsr", "pulsaar", "pulsur", "pulzar",
    "pulsarr", "pulsser", "pluser", "plusar",
    "dominar", "platina", "avenger",
    # TVS bikes (NOT scooters like jupiter, ntorq)
    "apache", "apachi", "apche", "raider", "ronin",
    # Hero bikes (NOT scooters like destini, pleasure, maestro)
    "splendor", "splendour", "splendr", "passion", "glamour", "xtreme", "xpulse",
    # Honda bikes (NOT scooters like activa, dio)
    "shine", "unicorn", "hornet", "sp125", "livo", "cb350",
    # Yamaha bikes (NOT scooters like fascino, ray, aerox)
    "fz", "r15", "mt15",
    # Royal Enfield - all are bikes
    "classic", "bullet", "bullat", "bullt", "hunter", "himalayan", "interceptor",
    "meteor", "continental", "scram", "enfield",
    # Suzuki bikes (NOT scooters like access, burgman, avenis)
    "gixxer", "hayabusa", "intruder",
    # KTM bikes
    "duke", "adventure",
    # General bike keywords
    "bike", "bikes", "motorcycle", "motorbike", "motor cycle", "motor bike"
    # NOTE: chetak is a SCOOTER (Bajaj electric) - NOT included here
}

# =================== SCOOTER-ONLY MODELS ===================
# These are specifically SCOOTER models - used to show only scooters
# When user searches for these, apply asset_category_name = "Scooters" filter
SCOOTER_ONLY_MODELS = {
    # Honda scooters
    "activa", "aktivia", "actva", "actva", "aktiva", "acitva",
    "dio", "deo", "diyo", "deeo",
    "grazia",
    # TVS scooters  
    "jupiter", "jupitar", "jupitr", "jupter", "jupitor",
    "ntorq", "ntork", "n-torq", "ntorc",
    "iqube", "i-qube", "i qube",
    "scooty", "scootr",
    # Bajaj scooters
    "chetak", "chetk", "chetek", "chatak",
    # Hero scooters
    "destini", "destiny", "destny",
    "pleasure", "pleasur", "plesure",
    "maestro", "mastro", "mestro",
    # Yamaha scooters
    "fascino", "fascno", "fasino", "fassino",
    "ray", "rayzr", "rayzor",
    "aerox", "arox",
    # Suzuki scooters
    "access", "acces", "accss", "acess",
    "burgman", "burgmn", "burgemann",
    "avenis", "avnis", "avenis",
    # Vespa/Aprilia scooters
    "vespa", "vepsa", "vspa",
    "aprilia", "aprila", "aprillia",
    # Electric scooters
    "ather", "athr", "atheer",
    "ola", "olla",  # Only for "ola scooter", not standalone to avoid "solar" match
    "okinawa", "oknawa", "okinava",
    "ampere", "amper",
    "bgauss", "b-gauss",
    "revolt", "revolt",
    "pure ev", "pure-ev", "pureev",
    "battre", "batrey",
    "ivoomi", "i-voomi",
    # General scooter keywords
    "scooter", "scooters", "scooty", "scootr", "scuter", "scootar", "scooti",
    "moped", "mopd",
    "gearless", "gear less", "gear-less",
    "electric scooter", "e-scooter", "e scooter", "ev scooter", "electric scooty"
}

# =================== TWO-WHEELER SUBCATEGORY MAPPING ===================
# Maps two-wheeler models/keywords to their ES asset_category_name subcategory
# This enables precise filtering: scooter queries → "Scooters", bike queries → exclude "Scooters"
TWO_WHEELER_SUBCATEGORY_MAP = {
    "scooter": {
        # Keywords that should filter to asset_category_name = "Scooters"
        "models": SCOOTER_ONLY_MODELS,
        "asset_category_filter": "Scooters",  # ES asset_category_name value
        "description": "Gearless two-wheelers like Activa, Jupiter, Chetak"
    },
    "motorcycle": {
        # Keywords that should filter to asset_category_name != "Scooters" (i.e., bikes)
        "models": BIKE_ONLY_MODELS,
        "asset_category_filter": "NOT_Scooters",  # Special flag: exclude Scooters
        "description": "Geared motorcycles like Pulsar, Bullet, Splendor"
    }
}

# =================== MODEL TO BRAND MAPPING (CANONICAL) ===================
# Maps model names (with variations/typos) to their brand for accurate filtering
# This is the single source of truth for model-brand relationships
CANONICAL_MODEL_BRAND_MAP = {
    # ===== HONDA =====
    "activa": "honda", "aktivia": "honda", "actva": "honda", "aktiva": "honda", "acitva": "honda",
    "dio": "honda", "deo": "honda", "diyo": "honda",
    "shine": "honda", "shne": "honda",
    "unicorn": "honda", "unicrn": "honda",
    "hornet": "honda", "hornt": "honda",
    "sp125": "honda", "sp 125": "honda",
    "livo": "honda", "lvo": "honda",
    "cb350": "honda", "cb 350": "honda", "hness": "honda", "highness": "honda",
    "grazia": "honda",
    # ===== HERO =====
    "splendor": "hero", "splendour": "hero", "splendr": "hero", "splndr": "hero", "splendur": "hero",
    "passion": "hero", "pashion": "hero", "passn": "hero",
    "glamour": "hero", "glamor": "hero", "glmour": "hero",
    "xtreme": "hero", "extreme": "hero", "xtrem": "hero",
    "xpulse": "hero", "x-pulse": "hero", "x pulse": "hero",
    "destini": "hero", "destiny": "hero", "destny": "hero",
    "maestro": "hero", "mastro": "hero", "mestro": "hero",
    "pleasure": "hero", "pleasur": "hero", "plesure": "hero",
    # ===== TVS =====
    "apache": "tvs", "apachi": "tvs", "apche": "tvs", "apachi": "tvs",
    "ntorq": "tvs", "ntork": "tvs", "n-torq": "tvs", "ntorc": "tvs",
    "jupiter": "tvs", "jupitar": "tvs", "jupitr": "tvs", "jupter": "tvs", "jupitor": "tvs",
    "raider": "tvs", "raidr": "tvs",
    "ronin": "tvs", "ronn": "tvs",
    "iqube": "tvs", "i-qube": "tvs", "i qube": "tvs",
    # ===== BAJAJ =====
    "pulsar": "bajaj", "pulser": "bajaj", "plsar": "bajaj", "plser": "bajaj", "pulsr": "bajaj",
    "pulsaar": "bajaj", "pulsur": "bajaj", "pulzar": "bajaj", "pulsarr": "bajaj", "pulsser": "bajaj",
    "dominar": "bajaj", "domnar": "bajaj",
    "platina": "bajaj", "platna": "bajaj",
    "avenger": "bajaj", "avengr": "bajaj", "avngr": "bajaj",
    "chetak": "bajaj", "chetk": "bajaj", "chetek": "bajaj", "chatak": "bajaj",
    # ===== YAMAHA =====
    "fz": "yamaha", "fzs": "yamaha", "fz-s": "yamaha",
    "r15": "yamaha", "r 15": "yamaha",
    "mt15": "yamaha", "mt 15": "yamaha", "mt-15": "yamaha",
    "fascino": "yamaha", "fascno": "yamaha", "fasino": "yamaha", "fassino": "yamaha",
    "ray": "yamaha", "rayzr": "yamaha", "rayzor": "yamaha",
    "aerox": "yamaha", "arox": "yamaha",
    # ===== ROYAL ENFIELD =====
    "bullet": "royal enfield", "bullat": "royal enfield", "bullt": "royal enfield", "bulet": "royal enfield",
    "classic": "royal enfield", "clasic": "royal enfield", "klassic": "royal enfield",
    "hunter": "royal enfield", "huntr": "royal enfield",
    "himalayan": "royal enfield", "himlayan": "royal enfield", "himalyan": "royal enfield",
    "interceptor": "royal enfield", "intercptr": "royal enfield",
    "meteor": "royal enfield", "metor": "royal enfield", "meteror": "royal enfield",
    "continental": "royal enfield", "contintl": "royal enfield",
    "scram": "royal enfield", "scrm": "royal enfield",
    # ===== SUZUKI =====
    "access": "suzuki", "acces": "suzuki", "acess": "suzuki",
    "burgman": "suzuki", "burgmn": "suzuki", "burgemann": "suzuki",
    "avenis": "suzuki", "avnis": "suzuki",
    "gixxer": "suzuki", "gixr": "suzuki", "gixxr": "suzuki",
    "hayabusa": "suzuki", "hayabuza": "suzuki", "hayabsa": "suzuki",
    "intruder": "suzuki", "intrudr": "suzuki",
    # ===== ELECTRIC BRANDS =====
    "ather": "ather", "athr": "ather",
    "ola s1": "ola", "ola electric": "ola",
    "okinawa": "okinawa", "oknawa": "okinawa",
    "ampere": "ampere", "amper": "ampere",
    "bgauss": "bgauss", "b-gauss": "bgauss",
    "revolt": "revolt",
    "pure ev": "pure ev",
    "battre": "battre",
    "ivoomi": "ivoomi", "i-voomi": "ivoomi",
    # ===== CAR MODELS =====
    # Hyundai
    "creta": "hyundai", "creata": "hyundai", "creat": "hyundai",
    "venue": "hyundai", "venu": "hyundai",
    "i20": "hyundai", "i 20": "hyundai", "i10": "hyundai", "i 10": "hyundai",
    "verna": "hyundai", "verna": "hyundai",
    "tucson": "hyundai", "tucsn": "hyundai",
    "alcazar": "hyundai", "alcazer": "hyundai",
    "exter": "hyundai", "extr": "hyundai",
    # Tata
    "nexon": "tata", "nexn": "tata", "nexxon": "tata",
    "punch": "tata", "puch": "tata",
    "tiago": "tata", "tago": "tata",
    "tigor": "tata", "tigr": "tata",
    "harrier": "tata", "harier": "tata",
    "safari": "tata", "safri": "tata",
    "altroz": "tata", "altrz": "tata",
    # Maruti
    "swift": "maruti", "swft": "maruti", "swfit": "maruti",
    "baleno": "maruti", "beleno": "maruti",
    "brezza": "maruti", "breza": "maruti", "breeza": "maruti",
    "dzire": "maruti", "dezire": "maruti", "dzir": "maruti",
    "ertiga": "maruti", "ertga": "maruti",
    "ciaz": "maruti", "ceaz": "maruti",
    "wagon r": "maruti", "wagonr": "maruti", "wagn r": "maruti",
    "alto": "maruti", "altoo": "maruti",
    "fronx": "maruti", "frox": "maruti",
    "jimny": "maruti", "jimni": "maruti",
    # Mahindra
    "thar": "mahindra", "tharr": "mahindra",
    "xuv700": "mahindra", "xuv 700": "mahindra", "xuv7oo": "mahindra",
    "xuv300": "mahindra", "xuv 300": "mahindra",
    "xuv400": "mahindra", "xuv 400": "mahindra",
    "scorpio": "mahindra", "scorpion": "mahindra", "scorpeo": "mahindra",
    "bolero": "mahindra", "bolro": "mahindra",
    # Toyota
    "fortuner": "toyota", "fortunar": "toyota", "fortnr": "toyota",
    "innova": "toyota", "inova": "toyota", "innva": "toyota",
    "glanza": "toyota", "glansa": "toyota",
    "urban cruiser": "toyota", "urban crusier": "toyota",
    "hyryder": "toyota", "hyrider": "toyota",
    # Honda Cars
    "city": "honda", "citty": "honda",
    "amaze": "honda", "amaz": "honda",
    "elevate": "honda", "elevat": "honda",
    # Kia
    "seltos": "kia", "sletos": "kia", "seltoss": "kia",
    "sonet": "kia", "sonett": "kia",
    "carens": "kia", "carns": "kia",
    "ev6": "kia", "ev 6": "kia",
    # ===== GOOGLE PIXEL (Smartphones) =====
    "pixel": "google", "pixle": "google", "pixxel": "google", "pixal": "google",
    "pixel 6": "google", "pixel 7": "google", "pixel 8": "google", "pixel 9": "google",
    "pixel pro": "google", "pixel fold": "google", "pixel a": "google",
}

# Product model names that should NOT be treated as brands (used for product search, not brand filter)
# These are model/product line names, not manufacturer names
PRODUCT_MODEL_NAMES = {
    # Two-wheeler models (brand is Bajaj, TVS, Hero, Royal Enfield, etc.)
    "pulsar", "apache", "activa", "splendor", "passion", "classic", "bullet", "hunter",
    "dominar", "ntorq", "access", "dio", "jupiter", "fascino", "gixxer", "r15", "fz",
    "hornet", "unicorn", "shine", "glamour", "xtreme", "xpulse", "destini", "maestro",
    "aerox", "mt15", "himalayan", "interceptor", "meteor", "continental", "scram",
    "burgman", "avenis", "chetak", "ather",
    # Phone model series (brand is Samsung, Apple, etc.)
    "galaxy", "note", "fold", "flip", "s23", "s24", "s25", "a54", "a34", "m34", "f15", "iphone",
    "pixel", "reno", "nord", "neo", "narzo", "gt",
}

# Mobile phone specific keywords that indicate smartphone category
MOBILE_PHONE_KEYWORDS = {
    # Product series - longer, more specific terms only
    "galaxy", "note", "fold", "flip", "s23", "s24", "s25", "a54", "a34", "a55", "a35",
    "m34", "f15", "f54", "f55", "f34", "iphone", "ipad", "pixel", "reno", "find x",
    "nord", "pro max", "ultra", "oneplus ce", "narzo", "infinix hot", "spark", "oneplus nord",
    # General phone terms (including typos and variations)
    "smartphone", "smartphones", "smartfone", "smartfon",
    "mobilephone", "cellphone", "cell phone", "cellular",
    "featurephone", "feature phone", "fetur phon", "feture phone",
    "touch phone", "touchphone", "camera phone", "best camera phone",
    "dual sim", "dual sim mobile", "dualsim",'mob','mobaile','mobail'
    "5g mobile", "4g phone", "4g mobile", "lte phone",
    "android mobile",
    # Phone typos - common misspellings
    "fone", "fon", "phon", "fonr", "phne", "phn",
    "phin", "phine", "pohne", "phonr", "phoen", "pone", "phonne",
    "phine", "phon", "phonee", "phoner", "phons", "phonse"
}

# Laptop/Computer specific keywords - checked before phone keywords
LAPTOP_KEYWORDS = {
    "laptop", "laptops", "notebook", "ultrabook", "chromebook",
    "gaming laptop", "student laptop", "business laptop",
    "macbook", "thinkpad", "ideapad", "vivobook", "zenbook",
    "inspiron", "pavilion", "predator", "nitro", "rog",
    "computer", "pc", "desktop"
}

# Kitchen appliance keywords
KITCHEN_KEYWORDS = {
    "mixer", "grinder", "mixer grinder", "mixergrinder", "blender",
    "juicer", "food processor", "wet grinder", "dry grinder",
    "hand blender", "hand mixer", "chopper", "mincer"
}

# Dishwasher keywords (must be checked BEFORE washing machine to avoid "washer" substring match)
DISHWASHER_KEYWORDS = {
    "dishwasher", "dishwashers", "dish washer", "dish washers",
    "dishwash", "dish wash", "dishwashing machine", "dish washing machine",
    # Common typos
    "dishwashr", "dishwasheer", "dishwashar", "diswasher", "dishwaher",
    "dishwsher", "dishwashe", "dishwashers", "dishwahser"
}

# Washing machine keywords (including typos and variations)
WASHING_MACHINE_KEYWORDS = {
    # Standard terms
    "washing machine", "washingmachine", "washing machines", "washingmachines",
    "washer", "washers", "clothes washer", "laundry machine",
    
    # Types
    "front load", "frontload", "front loader", "frontloader",
    "top load", "topload", "top loader", "toploader",
    "semi automatic", "semiautomatic", "semi-automatic",
    "fully automatic", "fullyautomatic", "fully-automatic",
    
    # Common typos and variations
    "washing machin", "washingmachin", "washing machne", "washng machine",
    "wasing machine", "wahing machine", "washig machine", "washin machine",
    "washing mashine", "washingmashine", "washing masheen", "washing macine",
    "washing machien", "washingmachien", "waching machine", "wachingmachine",
    "washing mechine", "washingmechine", "washing machinge", "washinng machine",
    "wahsing machine", "washign machine", "washnig machine",
    
    # Short forms and slang
    "wm", "w/m", "w m",
    
    # Hindi/regional transliterations
    "kapde dhone ki machine", "dhulai machine", "dhobi machine"
}

# OnePlus brand keywords (including typos and variations)
ONEPLUS_KEYWORDS = {
    # Standard terms
    "oneplus", "one plus", "one-plus", "1plus", "1+", "1 plus",
    
    # Common typos
    "onplus", "oneplus", "onepls", "oneplus", "oneplue", "oneplsu",
    "oneplus", "wan plus", "wanplus", "on plus", "onplus",
    "one plue", "onepus", "oneplus", "1 +", "one+",
    
    # Model series
    "oneplus nord", "oneplusnord", "one plus nord", "1+ nord", "1plus nord",
    "oneplus ce", "oneplusce", "one plus ce", "nord ce", "nordce",
    "oneplus open", "oneplusopen", "one plus open",
    "oneplus 12", "oneplus12", "one plus 12", "1+ 12",
    "oneplus 11", "oneplus11", "one plus 11", "1+ 11",
    "oneplus 10", "oneplus10", "one plus 10", "1+ 10",
    "oneplus pro", "onepluspro", "one plus pro",
    "oneplus r", "oneplusr", "one plus r",
    "nord 3", "nord3", "nord 2", "nord2", "nord ce 3", "nord ce3"
}

# Realme brand keywords (including typos and variations)
REALME_KEYWORDS = {
    # Standard terms
    "realme", "real me", "real-me",
    
    # Common typos
    "relme", "ralme", "relame", "ralame", "reelme", "releme",
    "realmi", "relmi", "relmae", "realmee", "reame", "reme",
    "realeme", "reallme", "realmr", "realem", "raelme", "reaelme",
    
    # Model series
    "realme narzo", "realmenarzo", "narzo",
    "realme gt", "realmegt", "gt neo",  # NOTE: Removed standalone "neo" - it conflicts with Samsung Neo QLED TV
    "realme c", "realmec", "realme c55", "realme c53", "realme c35",
    "realme 12", "realme12", "realme 11", "realme11",
    "realme 10", "realme10", "realme 9", "realme9",
}

# =================== REDMI KEYWORDS ===================
# Must be checked BEFORE Realme to avoid "redme" being misidentified as "realme"
REDMI_KEYWORDS = {
    # Standard terms
    "redmi", "redmi note", "redmi a", "redmi k",
    
    # Common typos (IMPORTANT: "redme" is Redmi, not Realme!)
    "redme", "radmi", "ridmi", "redmii", "redmmi", "redmie",
    "redmi note", "redmi notte", "redme note", "radmi note",
    
    # Model series
    "redmi note 13", "redmi note 12", "redmi note 11",
    "redmi 13", "redmi 12", "redmi 11",
    "redmi a1", "redmi a2", "redmi a3",
    "redmi k60", "redmi k70",
}

# =================== IQOO KEYWORDS ===================
# Must be checked BEFORE Realme to avoid "neo" being misidentified as "realme GT NEO"
# iQOO Neo series is different from realme GT NEO
IQOO_KEYWORDS = {
    # Standard terms
    "iqoo", "iqoo neo", "iqoo z", "i qoo",
    
    # Common typos
    "iqo", "iqooo", "iqoq", "iqqo",
    
    # Model series
    "iqoo neo 10", "iqoo neo 9", "iqoo neo 7",
    "iqoo z9", "iqoo z7", "iqoo 12", "iqoo 13",
}

# =================== GOOGLE PIXEL KEYWORDS ===================
# Must be checked to identify "pixel" as Google brand
GOOGLE_PIXEL_KEYWORDS = {
    # Standard terms
    "google pixel", "pixel phone", "google phone",
    
    # Standalone pixel (implies Google)
    "pixel", "pixel 6", "pixel 7", "pixel 8", "pixel 9", "pixel 10",
    "pixel pro", "pixel fold", "pixel a",
    
    # Common typos
    "pixle", "pixxel", "pixal", "pxel", "piksel",
    "pixle 8", "pixle 9", "pixxel 8", "pixxel 9",
    "googel pixel", "gogle pixel",
}

# =================== VEHICLE KEYWORDS ===================
# Two-wheeler specific keywords (including all typos and variations)
TWO_WHEELER_KEYWORDS = {
    # Category terms and typos
    "two wheeler", "twowheeler", "2 wheeler", "2wheeler", "2-wheeler",
    "two weeler", "two wheler", "two wheele", "twowhelar",
    "2 weeler", "2 wheler",  # additional typos
    "to wheeler", "to wheelers", "to wheler", "towheeler",  # "to" as typo for "two"
    
    # Bike terms and typos
    "bike", "bikes", "bik", "byke", "byk", "motorbike", "motor bike",
    "motorcycle", "motor cycle", "motorcycl", "motercycle", "motorbyk",
    
    # Scooter terms and typos  
    "scooter", "scooters", "scootr", "scouter", "scuter", "scootar",
    "scootee", "scootr", "scuter", "skutar", "skutr", "scoter", "scotr",
    "scooty", "scootie", "scooti", "skooty", "skooti", "scoty", "scoti",
    "scootey", "scutty", "scuty",
    
    # Electric vehicle terms
    "electric scooter", "electric bike", "e-bike", "ebike", "e bike",
    "ev scooter", "ev bike", "electric scooty",
    
    # Brand-specific model names (popular two-wheelers)
    # Hero models
    "splendor", "passion", "glamour", "shine", "xtreme", "xpulse", "destini",
    "pleasure", "maestro", "hf deluxe", "hf100",
    # Honda models  
    "activa", "shine", "unicorn", "hornet", "dio", "aviator", "livo", "sp125",
    "cb350", "highness", "hness",
    # TVS models
    "apache", "ntorq", "jupiter", "raider", "radeon", "sport", "xl100",
    "ronin", "iqube", "zest", "scooty pep", "scootypep",
    # Bajaj models
    "pulsar", "dominar", "avenger", "platina", "ct", "chetak",
    # Yamaha models
    "fz", "fzs", "r15", "mt15", "mt-15", "aerox", "fascino", "ray zr", "rayzr",
    # Royal Enfield models
    "bullet", "classic", "meteor", "himalayan", "interceptor", "continental",
    "hunter", "scram", "enfield", "royal enfield", "royalenfield",
    # Suzuki models
    "gixxer", "access", "burgman", "avenis", "intruder", "vstrom", "v-strom",
    # Electric brands/models
    "ather", "ola", "revolt", "simple one", "okinawa", "ampere", "pure ev",
    
    # Generic terms
    "moped", "vespa", "gearless", "gear less"
}

# Four-wheeler / Car specific keywords (including typos and variations)
FOUR_WHEELER_KEYWORDS = {
    # Category terms and typos
    "car", "cars", "kar", "kaar", "caar", "carr",
    "four wheeler", "fourwheeler", "4 wheeler", "4wheeler", "4-wheeler",
    "four weeler", "four wheler",
    
    # Vehicle type terms
    "suv", "suvs", "sedan", "sedans", "hatchback", "hatchbacks",
    "crossover", "mpv", "muv", "compact suv", "mid suv", "full size suv",
    
    # Electric car terms
    "electric car", "ev car", "electric vehicle", "e-car", "ecar",
    "electric suv", "ev suv",
    
    # Brand-specific popular models
    # Maruti/Suzuki models
    "swift", "dzire", "baleno", "brezza", "ertiga", "xl6", "celerio",
    "alto", "wagon r", "wagonr", "ignis", "ciaz", "s-presso", "spresso",
    "fronx", "jimny", "invicto", "grand vitara", "e vitara", "evitara",
    # Hyundai models
    "creta", "venue", "i10", "i20", "verna", "alcazar", "tucson", "exter",
    "aura", "xcent", "santro", "ioniq", "kona", "creta electric",
    # Tata models
    "nexon", "punch", "harrier", "safari", "altroz", "tiago", "tigor",
    "curvv", "ev", "nexon ev", "punch ev", "tiago ev", "tigor ev",
    # Mahindra models
    "xuv700", "xuv 700", "xuv500", "xuv 500", "xuv400", "xuv 400",
    "xuv300", "xuv 300", "thar", "scorpio", "scorpio n", "bolero",
    "xuv 3xo", "xuv3xo", "be 6e", "be6e", "xev 9e", "xev9e",
    # Kia models
    "seltos", "sonet", "carens", "carnival", "ev6", "ev9",
    # MG models
    "hector", "astor", "gloster", "zs", "zs ev", "comet", "windsor",
    # Volkswagen models
    "polo", "vento", "taigun", "virtus", "tiguan",
    # Other popular models
    "fortuner", "innova", "crysta", "hycross", "legender", "urban cruiser"
}

# Watch/Wearable specific keywords (including typos)
WATCH_KEYWORDS = {
    "watch", "watches", "watc", "wach", "wtach", "wtch", "wacth",
    "analog watch", "digital watch", "smart watch", "smartwatch",
    "fitness band", "fitness watch", "sports watch", "wrist watch",
    "wristwatch", "analog", "chronograph", "wearable", "wearables",
    "fitband", "fitness tracker", "tracker"
}

# Furniture specific product keywords for boosting
FURNITURE_PRODUCT_KEYWORDS = {
    "bed": ["bed", "beds", "double bed", "single bed", "king bed", "queen bed", "king size", "queen size"],
    "sofa": ["sofa", "sofas", "couch", "settee", "loveseat"],
    "table": ["table", "tables", "dining table", "dinner table","diner table", "study table", "center table", "coffee table", 
              "dining", "side table", "work table", "computer table", "office table"],
    "chair": ["chair", "chairs", "office chair", "gaming chair", "recliner", "dining chair"],
    "wardrobe": ["wardrobe", "wardrobes", "almirah", "cupboard", "closet"],
    "mattress": ["mattress", "mattresses", "foam mattress", "spring mattress"],
    "cabinet": ["cabinet", "cabinets", "tv cabinet", "kitchen cabinet", "storage cabinet", 
                "shoe cabinet", "display cabinet", "tv unit", "tv stand"],
    "bookshelf": ["bookshelf", "bookshelves", "shelf", "shelves", "rack", "book rack", "shoe rack", "shoe stand"],
}

# Home appliance specific product keywords for boosting
HOME_APPLIANCE_KEYWORDS = {
    "fan": ["fan", "fans", "ceiling fan", "table fan", "pedestal fan", "exhaust fan", "wall fan", "tower fan"],
    "cooler": ["cooler", "air cooler", "desert cooler", "room cooler"],
    "heater": ["heater", "room heater", "water heater", "geyser"],
    "vacuum cleaner": ["vacuum", "vacuum cleaner", "vacuumcleaner", "vaccum", "vaccum cleaner", 
                       "robot vacuum", "robotic vacuum", "mop", "floor mop", "robot mop", 
                       "vacuum mop", "mopping", "floor cleaner"]
}

# =================== APPLE PRODUCT DETECTION ===================
# UPDATED: Now we have ALL Apple products in our data:
# - iPhone (smartphone)
# - iPad (tablets)
# - MacBook (laptops)
# - AirPods (audio video)
# - Apple Watch (watch and wearable)
#
# BUSINESS RULE: When ANY Apple product is detected, show ALL Apple products
# No other brand should appear when Apple is detected

# Comprehensive mapping of Apple product variations to their categories
# Used for: 1) Detecting Apple queries  2) Category hints for specific products
APPLE_PRODUCTS_MAP = {
    # ===== MacBook (all variations) =====
    "macbook": "laptops",
    "mac book": "laptops",
    "macbok": "laptops",
    "macbbok": "laptops",
    "mackbook": "laptops",
    "makbook": "laptops",
    "macbuk": "laptops",
    "mcbook": "laptops",
    "mcbk": "laptops",
    "macbk": "laptops",
    "macboook": "laptops",
    "macboo": "laptops",
    "macbookk": "laptops",
    # MacBook Pro
    "macbook pro": "laptops",
    "macbookpro": "laptops",
    "mac book pro": "laptops",
    "macbook proo": "laptops",
    "macbookp": "laptops",
    "mbp": "laptops",  # Short form
    "mac pro": "laptops",
    "macpro": "laptops",
    # MacBook Air
    "macbook air": "laptops",
    "macbookair": "laptops",
    "mac book air": "laptops",
    "macbook aire": "laptops",
    "mba": "laptops",  # Short form
    "mac air": "laptops",
    "macair": "laptops",
    # MacBook with chips
    "macbook m1": "laptops",
    "macbook m2": "laptops",
    "macbook m3": "laptops",
    "macbook m4": "laptops",
    "m1 macbook": "laptops",
    "m2 macbook": "laptops",
    "m3 macbook": "laptops",
    # MacBook Neo
    "macbook neo": "laptops",
    "macbookneo": "laptops",
    "mac book neo": "laptops",
    "macbok neo": "laptops",
    "mackbook neo": "laptops",
    "neo macbook": "laptops",
    "neo mac book": "laptops",
    "neo macbok": "laptops",
    # NOTE: "neo" alone should show realme GT NEO phones, not laptops
    "neo laptop": "laptops",
    "neo laptops": "laptops",
    "neo book": "laptops",
    "neobook": "laptops",
    "neo air": "laptops",
    "neo pro": "laptops",
    "neo pro max": "laptops",
    "apple neo": "laptops",
    "appleneo": "laptops",
    # Apple + laptop
    "apple macbook": "laptops",
    "apple laptop": "laptops",
    "applemacbook": "laptops",
    "apple mac book": "laptops",
    
    # ===== iMac (all variations) =====
    # iMac products have actual_category="desktop" in ES
    "imac": "desktop",
    "i mac": "desktop",
    "imak": "desktop",
    "i-mac": "desktop",
    "imacc": "desktop",
    "imac 24": "desktop",
    "imac 27": "desktop",
    "imac pro": "desktop",
    "imacpro": "desktop",
    "apple imac": "desktop",
    "appleimac": "desktop",
    "apple desktop": "desktop",
    "apple computer": "desktop",
    "apple pc": "desktop",
    
    # ===== Mac (general) - no category filter, show all Mac products =====
    "mac": None,  # Show both MacBook (laptops) and Mac mini/iMac (desktop)
    
    # ===== Mac Mini (desktop) - all variations and typos =====
    "mac mini": "desktop",  # Mac mini is a desktop computer
    "macmini": "desktop",
    "mac-mini": "desktop",
    # Reversed order variations
    "mini mac": "desktop",
    "minimac": "desktop",
    "mini-mac": "desktop",
    # Common typos - missing letters
    "mac min": "desktop",
    "macmin": "desktop",
    "mac mni": "desktop",
    "macmni": "desktop",
    "mac miin": "desktop",
    "mac mnii": "desktop",
    # Common typos - extra/wrong letters
    "mac minni": "desktop",
    "mac minii": "desktop",
    "macminni": "desktop",
    "mac minie": "desktop",
    "mack mini": "desktop",
    "mack minni": "desktop",
    "mackmini": "desktop",
    # Reversed order typos
    "mini mack": "desktop",
    "minni mac": "desktop",
    "mni mac": "desktop",
    "mini mc": "desktop",
    "minii mac": "desktop",
    # Apple prefix variations
    "apple mac mini": "desktop",
    "apple macmini": "desktop",
    "applemacmini": "desktop",
    "apple mini mac": "desktop",
    
    # ===== Mac Studio (desktop) =====
    "mac studio": "desktop",  # Mac Studio is a desktop computer
    "macstudio": "desktop",
    "mac-studio": "desktop",
    "mack studio": "desktop",
    "mac studeo": "desktop",
    "mac stdio": "desktop",
    "apple mac studio": "desktop",
    
    "mac desktop": "desktop",
    "apple mac": None,  # Show all Mac products
    "applemac": None,
    
    # ===== iPad (all variations) =====
    "ipad": "tablets",
    "i pad": "tablets",
    "ipd": "tablets",
    "ipda": "tablets",
    "ipaad": "tablets",
    "ipadd": "tablets",
    "ippad": "tablets",
    "i-pad": "tablets",
    # iPad Pro
    "ipad pro": "tablets",
    "ipadpro": "tablets",
    "i pad pro": "tablets",
    "ipad proo": "tablets",
    # iPad Air
    "ipad air": "tablets",
    "ipadair": "tablets",
    "i pad air": "tablets",
    "ipad aire": "tablets",
    # iPad Mini
    "ipad mini": "tablets",
    "ipadmini": "tablets",
    "i pad mini": "tablets",
    # iPad with numbers
    "ipad 10": "tablets",
    "ipad 11": "tablets",
    "ipad 9": "tablets",
    "ipad 8": "tablets",
    "ipad pro 12": "tablets",
    "ipad pro 11": "tablets",
    # Apple + tablet
    "apple ipad": "tablets",
    "appleipad": "tablets",
    "apple tablet": "tablets",
    "appletablet": "tablets",
    
    # ===== AirPods (all variations) =====
    "airpods": "audio video",
    "airpod": "audio video",
    "air pods": "audio video",
    "air pod": "audio video",
    "airpodz": "audio video",
    "airpodss": "audio video",
    "arpods": "audio video",
    "arpod": "audio video",
    "aiepods": "audio video",
    "airpds": "audio video",
    "aripods": "audio video",
    "aripod": "audio video",
    "airpods ": "audio video",
    "air-pods": "audio video",
    "air-pod": "audio video",
    "airppods": "audio video",
    "airpodds": "audio video",
    "airpos": "audio video",
    "airpod s": "audio video",
    # AirPods Pro
    "airpods pro": "audio video",
    "airpod pro": "audio video",
    "airpodspro": "audio video",
    "airpodpro": "audio video",
    "air pods pro": "audio video",
    "airpods proo": "audio video",
    "airpod proo": "audio video",
    # AirPods Max
    "airpods max": "audio video",
    "airpod max": "audio video",
    "airpodsmax": "audio video",
    "airpodmax": "audio video",
    "air pods max": "audio video",
    # AirPods generations
    "airpods 2": "audio video",
    "airpods 3": "audio video",
    "airpods 4": "audio video",
    "airpods gen 2": "audio video",
    "airpods gen 3": "audio video",
    "airpods gen2": "audio video",
    "airpods gen3": "audio video",
    # EarPods (wired Apple earphones)
    "earpods": "audio video",
    "earpod": "audio video",
    "ear pods": "audio video",
    "ear pod": "audio video",
    # Apple + audio
    "apple airpods": "audio video",
    "appleairpods": "audio video",
    "apple earbuds": "audio video",
    "apple earphones": "audio video",
    "apple buds": "audio video",
    "apple pods": "audio video",
    "apods": "audio video",
    "apple headphones": "audio video",
    "apple earphone": "audio video",
    "apple earbud": "audio video",
    
    # ===== Apple Watch (all variations) =====
    "apple watch": "watch and wearable",
    "applewatch": "watch and wearable",
    "apple wach": "watch and wearable",
    "apple wtch": "watch and wearable",
    "apple wathc": "watch and wearable",
    "apple watvh": "watch and wearable",
    "apple wactch": "watch and wearable",
    "apple watchh": "watch and wearable",
    "aple watch": "watch and wearable",
    "appel watch": "watch and wearable",
    "apple wath": "watch and wearable",
    "appple watch": "watch and wearable",
    "apple-watch": "watch and wearable",
    # iWatch (common misnomer)
    "iwatch": "watch and wearable",
    "i watch": "watch and wearable",
    "i-watch": "watch and wearable",
    "iwach": "watch and wearable",
    "iwatc": "watch and wearable",
    "i wach": "watch and wearable",
    # Apple Watch variants
    "apple watch ultra": "watch and wearable",
    "apple watch se": "watch and wearable",
    "apple watch series": "watch and wearable",
    "apple watch series 9": "watch and wearable",
    "apple watch series 10": "watch and wearable",
    "apple watch 9": "watch and wearable",
    "apple watch 10": "watch and wearable",
    "watchos": "watch and wearable",
    # Apple + smartwatch
    "apple smart watch": "watch and wearable",
    "apple smartwatch": "watch and wearable",
    "applesmartwatch": "watch and wearable",
    
    # ===== AirTag (all variations) =====
    "airtag": "watch and wearable",
    "air tag": "watch and wearable",
    "airtags": "watch and wearable",
    "air tags": "watch and wearable",
    "air-tag": "watch and wearable",
    "airetag": "watch and wearable",
    "airtg": "watch and wearable",
    "airtagg": "watch and wearable",
    "airtaag": "watch and wearable",
    "apple airtag": "watch and wearable",
    "appleairtag": "watch and wearable",
    "apple tag": "watch and wearable",
    "apple tracker": "watch and wearable",
    "apple air tag": "watch and wearable",
    
    # ===== HomePod (all variations) =====
    "homepod": "audio video",
    "home pod": "audio video",
    "homepods": "audio video",
    "home pods": "audio video",
    "homepod mini": "audio video",
    "homepodmini": "audio video",
    "home pod mini": "audio video",
    "hompod": "audio video",
    "homepd": "audio video",
    "homepodd": "audio video",
    "apple homepod": "audio video",
    "applehomepod": "audio video",
    "apple speaker": "audio video",
    "apple smart speaker": "audio video",
    "apple home speaker": "audio video",
    
    # ===== Apple TV (all variations) =====
    "apple tv": "television",
    "appletv": "television",
    "apple tv 4k": "television",
    "appletv4k": "television",
    "apple tv hd": "television",
    "aple tv": "television",
    "appel tv": "television",
    "apple tvv": "television",
    "apple-tv": "television",
    "apple television": "television",
    "apple streaming": "television",
    
    # ===== Vision Pro / Apple VR (all variations) =====
    "vision pro": "watch and wearable",
    "visionpro": "watch and wearable",
    "apple vision": "watch and wearable",
    "applevision": "watch and wearable",
    "apple vision pro": "watch and wearable",
    "applevisionpro": "watch and wearable",
    "apple vr": "watch and wearable",
    "applevr": "watch and wearable",
    "apple headset": "watch and wearable",
    "apple glasses": "watch and wearable",
    "apple ar": "watch and wearable",
    "apple mixed reality": "watch and wearable",
    
    # ===== Magic Keyboard/Mouse/Trackpad =====
    "magic keyboard": "laptops",
    "magic mouse": "laptops",
    "magic trackpad": "laptops",
    "apple keyboard": "laptops",
    "apple mouse": "laptops",
    "apple trackpad": "laptops",
    "apple pencil": "tablets",
    "applepencil": "tablets",
    "apple pen": "tablets",
    
    # ===== Beats (Apple-owned) =====
    "beats": "audio video",
    "beats headphones": "audio video",
    "beats earbuds": "audio video",
    "beats solo": "audio video",
    "beats studio": "audio video",
    "beats fit": "audio video",
    "beats pro": "audio video",
    "beats pill": "audio video",
    "beatspro": "audio video",
    "beatssolo": "audio video",
    "beatsstudio": "audio video",
    
    # ===== iPhone (all variations) =====
    "iphone": "smartphone",
    "i phone": "smartphone",
    "iphones": "smartphone",
    "i phones": "smartphone",
    "i-phone": "smartphone",
    "ifone": "smartphone",
    "ipone": "smartphone",
    "iphon": "smartphone",
    "iphne": "smartphone",
    "iphn": "smartphone",
    "aifon": "smartphone",
    "aifone": "smartphone",
    "aiphone": "smartphone",
    "eyephone": "smartphone",
    "eyefone": "smartphone",
    "iphome": "smartphone",
    "iphobe": "smartphone",
    "iphonw": "smartphone",
    "iphonr": "smartphone",
    "iphond": "smartphone",
    "iphons": "smartphone",
    "iphine": "smartphone",
    "iphoje": "smartphone",
    "iphohe": "smartphone",
    "iohone": "smartphone",
    "ipbone": "smartphone",
    "iphkne": "smartphone",
    "ipgone": "smartphone",
    "ipjone": "smartphone",
    "iphpne": "smartphone",
    "iphoen": "smartphone",
    "ipohne": "smartphone",
    "1phone": "smartphone",
    "iph0ne": "smartphone",
    # iPhone models
    "iphone 11": "smartphone",
    "iphone 12": "smartphone",
    "iphone 13": "smartphone",
    "iphone 14": "smartphone",
    "iphone 15": "smartphone",
    "iphone 16": "smartphone",
    "iphone 17": "smartphone",
    "iphone11": "smartphone",
    "iphone12": "smartphone",
    "iphone13": "smartphone",
    "iphone14": "smartphone",
    "iphone15": "smartphone",
    "iphone16": "smartphone",
    "iphone17": "smartphone",
    "iphone se": "smartphone",
    "iphonese": "smartphone",
    "iphone x": "smartphone",
    "iphonex": "smartphone",
    "iphone xr": "smartphone",
    "iphonexr": "smartphone",
    "iphone xs": "smartphone",
    "iphonexs": "smartphone",
    "iphone xs max": "smartphone",
    "iphonexsmax": "smartphone",
    "iphone pro": "smartphone",
    "iphone pro max": "smartphone",
    "iphone plus": "smartphone",
    "iphone mini": "smartphone",
    # iPhone typo combinations
    "ifone 15": "smartphone",
    "ipone 15": "smartphone",
    "iphon 15": "smartphone",
    "ifone 16": "smartphone",
    "ipone 16": "smartphone",
    "iphon 16": "smartphone",
    "ifone 14": "smartphone",
    "ipone 14": "smartphone",
    "iphon 14": "smartphone",
    "i phone 15": "smartphone",
    "i phone 16": "smartphone",
    "i phone 14": "smartphone",
    "i phone 13": "smartphone",
    "apple 15": "smartphone",
    "apple 16": "smartphone",
    "apple 14": "smartphone",
    "apple 13": "smartphone",
    "apple 15 pro": "smartphone",
    "apple 16 pro": "smartphone",
    "apple 15 pro max": "smartphone",
    "apple 16 pro max": "smartphone",
    
    # ===== Generic Apple (shows all Apple products) =====
    "apple": None,  # None means show all categories
    "aple": None,
    "appel": None,
    "appl": None,
    "applle": None,
    "aplle": None,
    "aapple": None,
    "appple": None,
    "apple products": None,
    "apple mobile": "smartphone",
    "applemobile": "smartphone",
    "apple phone": "smartphone",
    "applephone": "smartphone",
    "apple smartphone": "smartphone",
    "apple fone": "smartphone",
    "apple phn": "smartphone",
    "apple cell": "smartphone",
    "apple cellphone": "smartphone",
}

# Backward compatibility alias
APPLE_NON_PHONE_PRODUCTS = APPLE_PRODUCTS_MAP

# =================== COMPREHENSIVE IPHONE QUERY VARIATIONS ===================
# 50+ real user query variations including:
# - Typos (iphon, ifone, ipone)
# - Noise (i$phone, i-phn, i_phone)
# - Short forms (ip, iph, appl)
# - Spaced variations (i phone, i  phone, ap ple phone)
# - Keyboard errors (iphome, iphobe, iphine)
# - Phonetic spelling (aifon, aiphone)
# - Combined variations (phone iphone, iphone phone)

IPHONE_NOISY_VARIATIONS = {
    # ===== Standard iPhone variations =====
    "iphone", "iphones", "i phone", "i phones", "i-phone", "i_phone",
    
    # ===== Common typos =====
    "iphon", "iphne", "ipone", "ifone", "ihone", "iphoen", "iphohe",
    "iphobe", "iphome", "iphonw", "iphonr", "iphond", "iphons",
    "ipjone", "iphkne", "ipgone", "iphonee", "iphonne", "iphpne",
    "iphone", "iohone", "ipbone", "iphine", "iphoje", "iphonf",
    
    # ===== Very noisy/short =====
    "iphn", "ipn", "ifn", "iph", "ipho", "iphon", "ifone", "ipone",
    "i phn", "i pn", "i ph", "i-phn", "i$phn", "i$phone", "i@phone",
    
    # ===== Spaced variations =====
    "i phon", "i pone", "i fone", "i fon", "i phne", "i phn",
    "ip hone", "iph one", "ipho ne", "i p h o n e",
    
    # ===== Apple + phone combinations =====
    "apple phone", "apple phones", "applephone", "apple mobile", "applemobile",
    "appl phone", "appl phn", "appl fone", "aple phone", "aple mobile",
    "appel phone", "appel mobile", "applle phone", "aplle phone",
    "apple fone", "apple phn", "apple phon", "apple mbl",
    "ap ple phone", "app le phone", "appl e phone",
    
    # ===== Phonetic/Sound-based =====
    "aifon", "aiphone", "aifone", "eyephone", "eyefone", "eyephn",
    "ifon", "eyphone", "aphone", "ephone", "eiphone", "eifone",
    
    # ===== Keyboard adjacent errors =====
    "iphoje", "iphomd", "iphonw", "iphinr", "iphpne", "ipyone",
    "ipgone", "ipjone", "iphobe", "iphome", "iphonr", "iphonf",
    "iohone", "ipbone", "iphine", "iphkne", "ophine", "uphone",
    
    # ===== Number pad/symbol noise =====
    "1phone", "!phone", "iph0ne", "iph0n3", "1ph0ne", "iphon3",
    "i-ph0ne", "i_ph0ne", "i$ph0ne",
    
    # ===== Word order variations =====
    "phone iphone", "phone apple", "mobile apple", "mobile iphone",
    "fone apple", "phn apple", "phone i", "fone iphone",
    
    # ===== Hindi/Regional influenced =====
    "aifone", "aiefone", "aiphn", "aipon", "eifon", "eifone",
    "apel fon", "apel phone", "apal phone", "apal fone",
    
    # ===== Extra characters =====
    "iphonee", "iphonne", "iphoone", "iiphon", "iipone", "iphonee",
    "iphoness", "iphonez", "iphonex", "iphoney",
    
    # ===== Missing vowels (ONLY iPhone-related, not generic phone typos) =====
    # REMOVED: "phn", "phne", "phon", "fon" - these are generic phone typos, not iPhone
    "iphn", "iphne", "ifn", "ipn",
    "appl phn", "apl phn", "aple phn", "apel phn",
    
    # ===== Doubled letters =====
    "iipphone", "ipphone", "iphonne", "iphoone", "ippone", "iffone",
    
    # ===== Mixed case in lowercase =====
    "Iphone", "IPhone", "IPHONE", "iPHONE", "Iphone",
    
    # ===== Suffix variations =====
    "iphonemobile", "iphonephone", "appleiphon", "iphoneapple",
    "iphonesmartphone", "appleiphone",
}

# iPhone model-specific noisy patterns
IPHONE_MODEL_NOISY = {
    # iPhone 17 variations (future-proofing + user queries)
    "iphone 17", "iphone17", "i phone 17", "iphone17pro", "iphone 17 pro",
    "ifone 17", "ipone 17", "iphon 17", "iph 17", "ip 17",
    "apple 17", "appl 17", "aple 17", "17 pro max", "17 promax", "17pro max",
    "iphone 17 pro max", "iphone17promax", "17promax", "i17", "ip17",
    "17pro", "17 pro", "17max", "17 max", "17plus", "17 plus",
    "17mini", "17 mini", "iphone 17 mini", "iphone17mini",
    
    # iPhone 16 variations  
    "iphone 16", "iphone16", "i phone 16", "iphone16pro", "iphone 16 pro",
    "ifone 16", "ipone 16", "iphon 16", "iph 16", "ip 16",
    "apple 16", "appl 16", "aple 16", "16 pro max", "16 promax", "16pro max",
    "iphone 16 pro max", "iphone16promax", "16promax", "i16", "ip16",
    "16pro", "16 pro", "16max", "16 max", "16plus", "16 plus",
    "16mini", "16 mini", "iphone 16 mini", "iphone16mini",
    
    # iPhone 15 variations
    "iphone 15", "iphone15", "i phone 15", "iphone15pro", "iphone 15 pro",
    "ifone 15", "ipone 15", "iphon 15", "iph 15", "ip 15",
    "apple 15", "appl 15", "aple 15", "15 pro max", "15 promax", "15pro max",
    "iphone 15 pro max", "iphone15promax", "15promax", "i15", "ip15",
    "15pro", "15 pro", "15max", "15 max", "15plus", "15 plus",
    "15mini", "15 mini", "iphone 15 mini", "iphone15mini",
    
    # iPhone 14 variations
    "iphone 14", "iphone14", "i phone 14", "iphone14pro", "iphone 14 pro",
    "ifone 14", "ipone 14", "iphon 14", "iph 14", "ip 14",
    "apple 14", "appl 14", "14 pro max", "14 promax", "14pro max",
    "iphone 14 pro max", "iphone14promax", "14promax", "i14", "ip14",
    "14pro", "14 pro", "14max", "14 max", "14plus", "14 plus",
    "14mini", "14 mini", "iphone 14 mini", "iphone14mini",
    
    # iPhone 13 variations
    "iphone 13", "iphone13", "i phone 13", "iphone13pro", "iphone 13 pro",
    "ifone 13", "ipone 13", "iphon 13", "13 pro max", "13 promax", "13pro max",
    "iphone 13 pro max", "iphone13promax", "13promax", "i13", "ip13",
    "13pro", "13 pro", "13max", "13 max", "13mini", "13 mini",
    "iphone 13 mini", "iphone13mini",
    
    # iPhone SE variations
    "iphone se", "iphonese", "i phone se", "ifone se", "ipone se",
    "apple se", "ise", "ip se", "iph se",
    
    # iPhone X/XR/XS variations
    "iphone x", "iphonex", "iphone xs", "iphonexs", "iphone xr", "iphonexr",
    "ifone x", "ipone x", "apple x", "ix", "ipx",
    
    # Pro/Max combinations (ONLY for iPhone context, not standalone)
    "pro max", "promax", "pro mx", "promx", "prmax",
    # NOTE: Removed "plus", "mini", "ultra" - these are too generic
    # They match non-Apple queries like "one plus 15", "samsung s25 ultra"
    # These suffixes should only be detected when combined with iPhone/Apple context
}

# Comprehensive list of Apple/iPhone related terms including common typos
# ONLY includes iPhone terms - other Apple products handled separately
APPLE_IPHONE_EXACT_TERMS = {
    # Exact Apple/iPhone terms ONLY
    "apple", "iphone",
    # iPhone model names
    "iphone se", "iphone x", "iphone xr", "iphone xs", "iphone xsmax",
    "iphone 11", "iphone 12", "iphone 13", "iphone 14", "iphone 15", "iphone 16", "iphone 17",
    "iphone11", "iphone12", "iphone13", "iphone14", "iphone15", "iphone16", "iphone17",
}

# Common iPhone/Apple typos and variations (using word boundaries)
APPLE_IPHONE_TYPO_PATTERNS = [
    # iPhone typos - must be standalone words or at word boundary
    r"\bi\s*phone\b",           # i phone, i  phone
    r"\biph?o?ne?\d*\b",        # iphon, ifone, ipone, iphone15
    r"\biphn\d*\b",             # iphn, iphn15
    r"\bifon[e]?\d*\b",         # ifon, ifone, ifone15
    r"\bipon[e]?\d*\b",         # ipon, ipone
    r"\baphone\d*\b",           # aphone
    r"\bi\-?phone\d*\b",        # i-phone, i-phone15
    r"\beyephone\d*\b",         # eyephone
    r"\biphne\d*\b",            # iphne
    r"\bipohne\d*\b",           # ipohne
    r"\biphoen\d*\b",           # iphoen
    r"\bifoneh?\d*\b",          # ifoneh
    # Apple typos - must be standalone
    r"\bappel\b",               # appel
    r"\baple\b",                # aple
    r"\bapplle\b",              # applle
    # iPad typos
    r"\bi\s*pad\b",             # i pad
    r"\bipd\b",                 # ipd
    # Phonetic variations
    r"\baifon[e]?\d*\b",        # aifon, aifone
    r"\beiphone?\d*\b",         # eiphone, eiphon
    r"\beifon[e]?\d*\b",        # eifon, eifone
    # Keyboard errors
    r"\biphome\d*\b",           # iphome (m near n)
    r"\biphobe\d*\b",           # iphobe (b near n)
    r"\biphomd\d*\b",           # iphomd
    r"\biphonw\d*\b",           # iphonw (w near e)
    r"\biphonr\d*\b",           # iphonr (r near e)
    r"\biphinr?\d*\b",          # iphinr, iphin
    # Apple phone combinations
    r"\bappl[e]?\s*ph?o?n[e]?\b",  # appl phone, apple phn, appl phn
    r"\bappl[e]?\s*mob(ile)?\b",   # appl mobile, apple mbl
    r"\bappl[e]?\s*fon[e]?\b",     # apple fone, appl fon
    # Number pad errors
    r"\b1phone\d*\b",           # 1phone
    r"\biph0ne\d*\b",           # iph0ne
    # AirPods typos - REMOVED from iPhone detection
    # AirPods should now search audio category from all brands
    
    # === NEW: Number-prefixed iPhone typos (17iphon, 16iphn, 14iphonpro, etc.) ===
    r"\b\d{1,2}iph?o?n[e]?(pro|max|plus|mini)?\b",   # 17iphon, 16iphon, 14iphonpro, 15iphonmax
    r"\b\d{1,2}iphn(pro|max|plus|mini)?\b",          # 17iphn, 16iphn, 14iphnpro
    r"\b\d{1,2}ifon[e]?(pro|max|plus|mini)?\b",      # 17ifone, 16ifon, 14ifonpro
    r"\b\d{1,2}ipon[e]?(pro|max|plus|mini)?\b",      # 17ipone, 16ipon
    r"\b\d{1,2}iphne(pro|max|plus|mini)?\b",         # 17iphne, 16iphne
    r"\b\d{1,2}ipohne(pro|max|plus|mini)?\b",        # 17ipohne
    r"\b\d{1,2}aifon[e]?(pro|max|plus|mini)?\b",     # 17aifon, 16aifone
    r"\b\d{1,2}eifon[e]?(pro|max|plus|mini)?\b",     # 17eifon
    r"\b\d{1,2}iphome(pro|max|plus|mini)?\b",        # 17iphome
    r"\b\d{1,2}i\s*phone?(pro|max|plus|mini)?\b",    # 17 i phone, 16 i phon
]

# Apple + number patterns (apple 15, apple 16 pro, etc.)
APPLE_NUMBER_PATTERN = r"apple\s*\d+\s*(pro|max|plus|mini|ultra)?"

# =================== OUT-OF-SCOPE QUERY GUARDRAILS ===================
# These queries are outside our business scope (e-commerce mall for electronics, appliances, vehicles)
# We do NOT sell financial products, services, real estate, etc.
# Returns early with appropriate "no results" response for these queries

# Finance/Investment keywords (NOT part of e-commerce business)
OUT_OF_SCOPE_FINANCE = {
    # Mutual Funds & SIP
    "mutual fund", "mutual funds","lawde", "lode", "bc" "mutualfund", "mutualfunds", "mf", "sip", "systematic investment",
    "mutual fnd", "mutal fund", "mutua fund", "mutul fund", "mutuelfund", "elss",
    # Stocks & Trading
    "stock", "stocks", "share", "shares", "equity", "equities", "nifty", "sensex",
    "intraday", "trading", "trade", "trader", "investment portfolio", "portfolio",
    "nse", "bse", "stock market", "share market", "sharemarket", "stockmarket",
    # ETF & Index Funds
    "etf", "index fund", "index funds", "indexfund", "exchange traded fund",
    # Insurance
    "insurance", "insurence", "insurnce", "life insurance", "health insurance", "term insurance",
    "term plan", "lic", "policy", "premium", "claim settlement", "mediclaim", "ulip",
    "motor insurance", "car insurance", "bike insurance", "vehicle insurance",
    # Fixed Deposits & Savings
    "fixed deposit", "fd", "recurring deposit", "rd", "ppf", "public provident fund",
    "epf", "provident fund", "nps", "national pension", "pension scheme", "pension fund",
    # Loans (general - not EMI financing for products)
    "home loan", "homeloan", "personal loan", "personalloan", "loan","Loan","car loan", "carloan",
    "education loan", "gold loan", "loan apply", "loan application", "loan eligibility",
    "loan calculator", "emi calculator", "interest rate loan",
    # Credit Cards & Banking
    "credit card", "creditcard", "debit card", "debitcard", "credit limit", "card apply",
    "bank account", "savings account", "current account", "demat", "demat account",
    # Cryptocurrency
    "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency", "dogecoin", "doge",
    "blockchain", "nft", "binance", "coinbase", "wazirx", "coindcx",
    # Bonds & Other Investments
    "bond", "bonds", "gold bond", "sovereign bond", "government bond", "g-sec",
    "treasury", "gilt fund", "debt fund", "liquid fund", "arbitrage fund",
    # IPO
    "ipo", "initial public offering", "ipo apply", "ipo allotment",
}

# Services (NOT products we sell)
OUT_OF_SCOPE_SERVICES = {
    # Travel & Booking
    "flight", "flights", "air ticket", "airticket", "flight booking", "airline",
    "train ticket", "railway ticket", "irctc", "train booking", "rail ticket",
    "bus ticket", "bus booking", "redbus", "abhibus",
    "hotel", "hotels", "hotel booking", "oyo", "trivago", "makemytrip", "goibibo",
    "cab", "cab booking", "ola cab", "uber", "rapido", "taxi",
    # Entertainment Booking
    "movie ticket", "movie tickets", "bookmyshow", "paytm movies", "movie booking",
    "event ticket", "concert ticket", "match ticket",
    # Recharge & Bill Payment
    "recharge", "mobile recharge", "dth recharge", "prepaid recharge", "postpaid bill",
    "electricity bill", "gas bill", "water bill", "broadband bill", "bill payment",
    # Food Delivery
    "food delivery", "zomato", "swiggy", "food order", "restaurant",
    # Other Services
    "courier", "delivery service", "packers movers", "pest control", "cleaning service",
}

# Real Estate (NOT our business)
OUT_OF_SCOPE_REAL_ESTATE = {
    "flat", "flats", "apartment", "apartments", "house", "houses", "villa", "villas",
    "plot", "plots", "land", "property", "properties", "real estate", "realestate",
    "rent house", "rent flat", "pg accommodation", "hostel", "paying guest",
    "buy flat", "buy house", "buy property", "sell property", "home buy",
    "2bhk", "3bhk", "1bhk", "4bhk", "builder", "builders", "construction",
    "99acres", "magicbricks", "housing.com", "nobroker",
}

# Jobs & Education (NOT our business)
OUT_OF_SCOPE_JOBS_EDUCATION = {
    # Jobs
    "job", "jobs", "vacancy", "vacancies", "hiring", "recruitment", "career", "careers",
    "job opening", "job apply", "resume", "cv", "interview", "naukri", "indeed",
    "linkedin job", "internship", "fresher job", "experienced job", "work from home",
    # Education
    "course", "courses", "tutorial", "tutorials", "coaching", "classes", "class",
    "admission", "admissions", "college", "university", "school", "institute",
    "certification", "certificate", "degree", "diploma", "online course", "udemy",
    "coursera", "byjus", "unacademy", "vedantu", "exam", "examination",
}

# Healthcare/Pharmacy (unless we sell health devices)
OUT_OF_SCOPE_HEALTHCARE = {
    # Medicines (we don't sell)
    "medicine", "medicines", "tablet", "tablets", "capsule", "capsules", "syrup",
    "pharmacy", "medical store", "drug", "drugs", "prescription",
    "1mg", "pharmeasy", "netmeds", "apollo pharmacy", "medplus",
    # Medical Services
    "doctor", "hospital", "clinic", "appointment", "consultation", "lab test",
    "blood test", "health checkup", "diagnostic", "pathology",
}

# Groceries/Food (NOT our business)
OUT_OF_SCOPE_GROCERIES = {
    "grocery", "groceries", "vegetables", "fruits", "rice", "wheat", "dal", "flour",
    "milk", "bread", "egg", "eggs", "meat", "chicken", "fish", "mutton",
    "bigbasket", "blinkit", "zepto", "instamart", "jiomart", "dmart", "grofers",
}

# Combine all out-of-scope categories
ALL_OUT_OF_SCOPE_KEYWORDS = (
    OUT_OF_SCOPE_FINANCE | 
    OUT_OF_SCOPE_SERVICES | 
    OUT_OF_SCOPE_REAL_ESTATE | 
    OUT_OF_SCOPE_JOBS_EDUCATION |
    OUT_OF_SCOPE_HEALTHCARE |
    OUT_OF_SCOPE_GROCERIES
)

def is_out_of_scope_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a query is outside the business scope (e-commerce mall for electronics/appliances/vehicles).
    
    Args:
        query: The user's search query
        
    Returns:
        Tuple of (is_out_of_scope, category) where:
        - is_out_of_scope: True if query should return no results
        - category: The out-of-scope category name for logging (None if in-scope)
    
    IMPORTANT: This function is conservative - only blocks queries that are CLEARLY
    out of scope. Ambiguous queries pass through to normal search.
    """
    if not query:
        return False, None
    
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())
    
    # Normalize: remove extra spaces
    query_normalized = ' '.join(query_lower.split())
    
    # =================== STRICT BLOCK KEYWORDS ===================
    # These keywords are SO clearly out-of-scope that they should ALWAYS block
    # regardless of any context words like "price", "buy", etc.
    STRICT_BLOCK_KEYWORDS = {
        # Cryptocurrency - never sold on e-commerce
        "bitcoin", "btc", "ethereum", "eth", "dogecoin", "doge", "crypto", "cryptocurrency",
        "blockchain", "nft", "binance", "coinbase", "wazirx", "coindcx",
        # Financial platforms - never products
        "mutual fund", "mutualfund", "sip", "nifty", "sensex", "demat",
        "ipo", "stock market", "share market", "trading",
        # Service platforms - never products
        "zomato", "swiggy", "uber", "ola cab", "rapido", "bookmyshow",
        "makemytrip", "goibibo", "oyo", "trivago", "irctc", "redbus",
        "naukri", "indeed", "linkedin job",
        "1mg", "pharmeasy", "netmeds", "apollo pharmacy",
        "bigbasket", "blinkit", "zepto", "jiomart", "grofers",
        "99acres", "magicbricks", "nobroker", "housing.com",
        "udemy", "coursera", "byjus", "unacademy", "vedantu",
        # =================== BAJAJ FINANCE SERVICES (NOT PRODUCTS) ===================
        "bajaj emi card", "emi card", "emi card apply", "bajaj card", "bajaj finance card",
        "bajaj finserv card", "bajaj finserv", "bajaj finance", "finserv",
        "extended warranty", "extendedwarranty", "warranty extension", "amc", "annual maintenance",
        # Loan/Account related
        "loan account", "loan status", "emi status", "emi payment", "pay emi",
        "emi balance", "emi due", "overdue emi", "emi bounce",
        # Gift cards/Vouchers (not physical products)
        "gift card", "giftcard", "gift voucher", "voucher", "e-voucher", "evoucher",
        "amazon voucher", "flipkart voucher",
        # Customer support queries (not product searches)
        "customer care", "customercare", "helpline", "toll free", "tollfree",
        "contact number", "support number", "complaint", "grievance",
        # Order/Delivery tracking (not product searches)
        "track order", "order status", "delivery status", "order tracking",
        "where is my order", "order id", "shipment status",
        # Return/Refund queries (not product searches)
        "return policy", "refund status", "refund policy", "cancel order",
        "exchange policy", "replacement policy",
        # Coupons/Offers (promotional, not products)
        "coupon code", "promo code", "promocode", "discount code", "offer code",
    }
    
    # Check strict block keywords first
    for keyword in STRICT_BLOCK_KEYWORDS:
        if ' ' in keyword:
            if keyword in query_normalized:
                # Categorize for logging
                if keyword in {"bitcoin", "btc", "ethereum", "eth", "dogecoin", "doge", "crypto", "cryptocurrency", "blockchain", "nft", "binance", "coinbase", "wazirx", "coindcx"}:
                    return True, "finance"
                elif keyword in {"bajaj emi card", "emi card", "emi card apply", "bajaj card", "bajaj finance card", "bajaj finserv card", "bajaj finserv", "bajaj finance", "extended warranty", "extendedwarranty", "warranty extension", "amc", "annual maintenance", "loan account", "loan status", "emi status", "emi payment", "pay emi", "emi balance", "emi due", "overdue emi", "emi bounce"}:
                    return True, "finance_services"
                elif keyword in {"gift card", "giftcard", "gift voucher", "voucher", "e-voucher", "evoucher", "amazon voucher", "flipkart voucher", "coupon code", "promo code", "promocode", "discount code", "offer code"}:
                    return True, "non_product"
                elif keyword in {"customer care", "customercare", "helpline", "toll free", "tollfree", "contact number", "support number", "complaint", "grievance", "track order", "order status", "delivery status", "order tracking", "where is my order", "order id", "shipment status", "return policy", "refund status", "refund policy", "cancel order", "exchange policy", "replacement policy"}:
                    return True, "support_query"
                elif keyword in {"zomato", "swiggy", "uber", "ola cab", "rapido", "bookmyshow", "makemytrip", "goibibo", "oyo", "trivago", "irctc", "redbus"}:
                    return True, "services"
                elif keyword in {"naukri", "indeed", "linkedin job", "udemy", "coursera", "byjus", "unacademy", "vedantu"}:
                    return True, "jobs_education"
                elif keyword in {"1mg", "pharmeasy", "netmeds", "apollo pharmacy"}:
                    return True, "healthcare"
                elif keyword in {"bigbasket", "blinkit", "zepto", "jiomart", "grofers"}:
                    return True, "groceries"
                elif keyword in {"99acres", "magicbricks", "nobroker", "housing.com"}:
                    return True, "real_estate"
                return True, "finance"
        else:
            if keyword in query_words:
                if keyword in {"bitcoin", "btc", "ethereum", "eth", "dogecoin", "doge", "crypto", "cryptocurrency", "blockchain", "nft", "binance", "coinbase", "wazirx", "coindcx", "mutual", "mutualfund", "sip", "nifty", "sensex", "demat", "ipo", "trading"}:
                    return True, "finance"
                elif keyword in {"finserv", "amc", "voucher", "evoucher", "giftcard", "promocode", "helpline", "tollfree", "grievance", "complaint"}:
                    return True, "finance_services"
                elif keyword in {"zomato", "swiggy", "uber", "rapido", "bookmyshow", "makemytrip", "goibibo", "oyo", "trivago", "irctc", "redbus"}:
                    return True, "services"
                elif keyword in {"naukri", "indeed", "udemy", "coursera", "byjus", "unacademy", "vedantu"}:
                    return True, "jobs_education"
                elif keyword in {"1mg", "pharmeasy", "netmeds"}:
                    return True, "healthcare"
                elif keyword in {"bigbasket", "blinkit", "zepto", "jiomart", "grofers"}:
                    return True, "groceries"
                elif keyword in {"99acres", "magicbricks", "nobroker"}:
                    return True, "real_estate"
                return True, "finance"
    
    # =================== WHITELIST CHECK ===================
    # These are product categories we DO sell - never block these
    # Even if they contain out-of-scope keywords
    WHITELIST_PATTERNS = {
        # Health devices we sell
        "bp monitor", "blood pressure", "glucometer", "oximeter", "pulse oximeter",
        "thermometer", "weighing scale", "weighing machine", "body fat analyzer",
        "nebulizer", "massager", "massage chair", "massage gun", "health band",
        "fitness band", "fitness tracker", "smart watch", "smartwatch",
        # Solar products we sell
        "solar panel", "solar inverter", "solar battery", "solar light",
        # Product-related EMI terms (buying products on EMI - NOT emi card services)
        "emi mobile", "emi phone", "emi tv", "emi laptop", "emi ac",
        "no cost emi", "emi offer",  # Removed "bajaj emi", "emi card" - those are services
        # Tablet devices (not medicine tablets)
        "tablet samsung", "tablet ipad", "tablet lenovo", "android tablet",
        "ipad", "galaxy tab", "tab s", "drawing tablet", "graphics tablet",
        # Kitchen appliances (contain "rice" but are not groceries)
        "rice cooker", "cooker", "pressure cooker", "electric cooker", "induction cooker",
        "bread maker", "breadmaker", "bread toaster", "egg boiler", "egg cooker",
    }
    
    for pattern in WHITELIST_PATTERNS:
        if pattern in query_normalized:
            return False, None
    
    # =================== CONTEXT-AWARE CHECKS ===================
    # Don't block if query contains product-related context
    PRODUCT_CONTEXT_WORDS = {
        "buy", "cost", "compare", "best", "top", "review", "reviews",
        "specification", "specs", "features", "color", "colour", "size",
        "gb", "tb", "inch", "kg", "litre", "liter", "watt", "ton",
        "samsung", "apple", "lg", "sony", "mi", "redmi", "vivo", "oppo",
        "bajaj", "honda", "hero", "tvs", "yamaha", "royal enfield",
        "haier", "voltas", "daikin", "lloyd", "godrej", "whirlpool", "ifb",
        # Kitchen appliance context words
        "cooker", "mixer", "grinder", "blender", "toaster", "oven", "microwave",
        "induction", "kettle", "fryer", "juicer", "choppers",
    }
    # NOTE: Removed "price" from context words to avoid false positives like "bitcoin price"
    
    has_product_context = bool(query_words & PRODUCT_CONTEXT_WORDS)
    
    # =================== OUT-OF-SCOPE DETECTION ===================
    # Check for exact phrase matches first (more reliable)
    for keyword in ALL_OUT_OF_SCOPE_KEYWORDS:
        # For multi-word phrases, check exact phrase match
        if ' ' in keyword:
            if keyword in query_normalized:
                # But if there's product context, don't block
                if has_product_context:
                    continue
                # Determine category for logging
                if keyword in OUT_OF_SCOPE_FINANCE:
                    return True, "finance"
                elif keyword in OUT_OF_SCOPE_SERVICES:
                    return True, "services"
                elif keyword in OUT_OF_SCOPE_REAL_ESTATE:
                    return True, "real_estate"
                elif keyword in OUT_OF_SCOPE_JOBS_EDUCATION:
                    return True, "jobs_education"
                elif keyword in OUT_OF_SCOPE_HEALTHCARE:
                    return True, "healthcare"
                elif keyword in OUT_OF_SCOPE_GROCERIES:
                    return True, "groceries"
                return True, "other"
        else:
            # For single words, check word boundary match
            if keyword in query_words:
                # Skip common false positives
                # "tablet" could be medicine or device - need context
                # IMPORTANT: On e-commerce sites, "tablet" almost always means electronic device
                # Only block if there's clear medicine context
                if keyword == "tablet" or keyword == "tablets":
                    # Medicine indicators - only block if these are present
                    medicine_indicators = {"medicine", "mg", "capsule", "syrup", "dose", "prescription", "paracetamol", "crocin", "dolo"}
                    if query_words & medicine_indicators:
                        return True, "healthcare"  # Clearly medicine
                    continue  # Assume it's a tablet device (most common e-commerce search)
                
                # "insurance" skip if buying a product with insurance
                if keyword == "insurance" and any(w in query_normalized for w in ["phone insurance", "mobile insurance", "laptop insurance", "extended warranty"]):
                    continue
                
                # "policy" skip if it's return policy etc
                if keyword == "policy" and any(w in query_normalized for w in ["return policy", "exchange policy", "warranty policy"]):
                    continue
                
                # "premium" skip if it's a product tier
                if keyword == "premium" and any(w in query_normalized for w in ["premium phone", "premium tv", "premium laptop", "premium model"]):
                    continue
                
                # If product context exists, don't block ambiguous single words
                if has_product_context:
                    continue
                
                # Block if it's a clear out-of-scope intent
                if keyword in OUT_OF_SCOPE_FINANCE:
                    return True, "finance"
                elif keyword in OUT_OF_SCOPE_SERVICES:
                    return True, "services"
                elif keyword in OUT_OF_SCOPE_REAL_ESTATE:
                    return True, "real_estate"
                elif keyword in OUT_OF_SCOPE_JOBS_EDUCATION:
                    return True, "jobs_education"
                elif keyword in OUT_OF_SCOPE_HEALTHCARE:
                    return True, "healthcare"
                elif keyword in OUT_OF_SCOPE_GROCERIES:
                    return True, "groceries"
    
    return False, None


def get_out_of_scope_response(query: str, category: str) -> dict:
    """
    Generate a graceful "no results" response for out-of-scope queries.
    
    This maintains the same response structure as normal search results
    to ensure frontend compatibility.
    """
    # Category-specific messages (for logging/debugging only, not shown to user)
    category_messages = {
        "finance": "Financial products like mutual funds, stocks, and insurance are not available.",
        "services": "Service bookings like travel, food delivery are not available.",
        "real_estate": "Real estate listings are not available.",
        "jobs_education": "Job listings and educational courses are not available.",
        "healthcare": "Medicines and medical services are not available.",
        "groceries": "Grocery items are not available.",
        "other": "This category is not available."
    }
    
    logger.info(f"OUT-OF-SCOPE query blocked: '{query}' (category: {category})")
    
    return {
        "status": 200,
        "message": "Success",
        "data": {
            "PostV1Productlist": {
                "status": 200,
                "message": "No results found",
                "data": {
                    "totalrecords": 0,
                    "products": [],
                    "filter": None,
                    "query_info": {
                        "original_query": query,
                        "processed_query": query,
                        "out_of_scope": True,
                        "out_of_scope_category": category
                    }
                }
            }
        }
    }


def get_apple_non_phone_category(query: str) -> Optional[str]:
    """
    Check if query is for a non-phone Apple product (macbook, airpods, ipad, etc.)
    Returns the category to search if it's a non-phone Apple product, None otherwise.
    These products should search their category from ALL brands, not just Apple.
    
    IMPORTANT: This function should NOT return "smartphone" for iPhone queries.
    iPhone queries should be handled by is_apple_iphone_query() separately.
    """
    if not query:
        return None
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())
    query_no_space = query_lower.replace(" ", "")
    
    # Check for non-phone Apple products using STRICT WORD BOUNDARY matching
    # This prevents "washing machine" from matching "mac"
    for product, category in APPLE_NON_PHONE_PRODUCTS.items():
        # CRITICAL: Skip smartphone category - iPhone queries should NOT be caught here
        # This function is for NON-PHONE products only
        if category == "smartphone":
            continue
            
        product_words = product.split()
        
        if len(product_words) == 1:
            # Single word like "mac", "ipad", "airpods"
            # Must be a COMPLETE word in query
            if product in query_words:
                return category
            # Also check if query STARTS with this product (compound word like "macbook")
            # But NOT if it's in the middle of another word (like "washing machine" -> "mac")
            if query_lower.startswith(product + " ") or query_lower == product:
                return category
            # Check compound words: "macbook" should match "mac"
            # But only if query_no_space starts with product
            if query_no_space.startswith(product):
                return category
        else:
            # Multi-word like "apple watch", "air pods" - use contains check
            # But verify word boundaries using regex
            pattern = r'\b' + re.escape(product) + r'\b'
            if re.search(pattern, query_lower):
                return category
    
    return None


def is_apple_product_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if query is related to ANY Apple product.
    Returns (True, category_hint) if Apple product detected.
    category_hint can be:
        - "smartphone" for iPhone queries
        - "tablets" for iPad queries
        - "laptops" for MacBook queries
        - "audio video" for AirPods queries
        - "watch and wearable" for Apple Watch queries
        - None for generic "apple" queries (show all Apple products)
    
    BUSINESS RULE: When Apple is detected, show ONLY Apple products (no other brands)
    """
    if not query:
        return False, None
    
    query_lower = query.lower().strip()
    query_cleaned = re.sub(r'[^\w\s]', '', query_lower)
    query_no_space = query_lower.replace(" ", "").replace("-", "").replace("_", "").replace("$", "").replace("@", "")
    query_words = set(query_lower.split())
    
    # CRITICAL: Exclude non-Apple brands - if another brand is mentioned, NOT Apple
    # Use word boundary check for short brand names to avoid false positives (e.g., "mi" in "mini")
    non_apple_brands_short = {"mi", "lg", "hp"}  # Short brands that need word boundary check
    non_apple_brands_long = {
        "samsung", "vivo", "oppo", "realme", "oneplus", "xiaomi", "redmi", 
        "poco", "motorola", "nokia", "tecno", "infinix", "iqoo", "nothing", "google", "pixel",
        "dell", "lenovo", "asus", "acer", "msi", "sony", "jbl", "boat", "boult",
        "noise", "fire-boltt", "fireboltt", "amazfit", "samsung galaxy", "galaxy"
    }
    # Check long brand names with simple substring
    for brand in non_apple_brands_long:
        if brand in query_lower:
            return False, None
    # Check short brand names with word boundary to avoid false positives like "mi" in "mini"
    for brand in non_apple_brands_short:
        if re.search(rf'\b{brand}\b', query_lower):
            return False, None
    
    # CRITICAL: "one plus" / "1 plus" / "1plus" are OnePlus phone queries, NOT Apple
    oneplus_patterns = [r'\bone\s*plus', r'\b1\s*plus', r'\b1plus']
    for pattern in oneplus_patterns:
        if re.search(pattern, query_lower):
            return False, None
    
    # Exclude generic terms that shouldn't trigger Apple detection
    non_apple_terms = {"machine", "appliance", "appliances", "application", "applications", "washing"}
    if query_words & non_apple_terms:
        return False, None
    
    # ===== CHECK 1: Direct match in APPLE_PRODUCTS_MAP =====
    # Check exact match
    if query_lower in APPLE_PRODUCTS_MAP:
        return True, APPLE_PRODUCTS_MAP[query_lower]
    if query_cleaned in APPLE_PRODUCTS_MAP:
        return True, APPLE_PRODUCTS_MAP[query_cleaned]
    if query_no_space in APPLE_PRODUCTS_MAP:
        return True, APPLE_PRODUCTS_MAP[query_no_space]
    
    # ===== CHECK 2: Check each product pattern in the map =====
    for product, category in APPLE_PRODUCTS_MAP.items():
        product_words = product.split()
        
        if len(product_words) == 1:
            # Single word - need word boundary check
            if product in query_words:
                return True, category
            # Check if query starts with product
            if query_lower.startswith(product + " ") or query_lower == product:
                return True, category
            if query_no_space.startswith(product) and len(product) >= 4:
                return True, category
        else:
            # Multi-word - use regex with word boundaries
            pattern = r'\b' + re.escape(product) + r'\b'
            if re.search(pattern, query_lower):
                return True, category
    
    # ===== CHECK 3: Check IPHONE_NOISY_VARIATIONS =====
    if query_lower in IPHONE_NOISY_VARIATIONS:
        return True, "smartphone"
    if query_cleaned in IPHONE_NOISY_VARIATIONS:
        return True, "smartphone"
    if query_no_space in IPHONE_NOISY_VARIATIONS:
        return True, "smartphone"
    
    # ===== CHECK 4: iPhone model noisy patterns =====
    if query_lower in IPHONE_MODEL_NOISY:
        return True, "smartphone"
    if query_cleaned in IPHONE_MODEL_NOISY:
        return True, "smartphone"
    for model_pattern in IPHONE_MODEL_NOISY:
        if len(model_pattern) <= 5:
            if re.search(r'\b' + re.escape(model_pattern) + r'\b', query_lower):
                return True, "smartphone"
        else:
            if model_pattern in query_lower or model_pattern in query_no_space:
                return True, "smartphone"
    
    # ===== CHECK 5: Apple/iPhone typo patterns =====
    for pattern in APPLE_IPHONE_TYPO_PATTERNS:
        if re.search(pattern, query_lower) or re.search(pattern, query_cleaned):
            return True, "smartphone"
    
    # ===== CHECK 6: Apple + number pattern (apple 15, apple 16 pro) =====
    if re.search(APPLE_NUMBER_PATTERN, query_lower):
        return True, "smartphone"
    
    # ===== CHECK 7: Fuzzy matching for very noisy queries =====
    # IMPORTANT: Include all phone/mobile typos to prevent false iPhone detection
    excluded_words = {"phone", "phones", "phon", "phne", "phn", "phonee", "phonne", "phoner", "phonr",
                      "mobile", "mobiles", "android", "smartphone", "smartphones",
                      "feature", "fetur", "feture", "touch", "camera", "best", "dual", "sim", 
                      "cell", "cellular", "smart", "fone", "fon", "fonr", "the", "and", "for",
                      "laptop", "laptops", "tablet", "tablets", "watch", "watches", "earbuds",
                      "headphone", "headphones"}
    
    apple_terms_to_check = ["iphone", "macbook", "airpods", "ipad", "apple"]
    for word in query_words:
        if len(word) >= 4 and word not in excluded_words:
            for apple_term in apple_terms_to_check:
                score = fuzz.ratio(word, apple_term)
                if score >= 80:  # 80% similarity
                    # Determine category based on what matched
                    if apple_term == "iphone":
                        return True, "smartphone"
                    elif apple_term == "macbook":
                        return True, "laptops"
                    elif apple_term == "airpods":
                        return True, "audio video"
                    elif apple_term == "ipad":
                        return True, "tablets"
                    elif apple_term == "apple":
                        return True, None
    
    # ===== CHECK 8: Compound patterns =====
    if "iphone" in query_no_space or "ifone" in query_no_space or "ipone" in query_no_space:
        return True, "smartphone"
    if "applephone" in query_no_space or "phoneapple" in query_no_space:
        return True, "smartphone"
    if "applemobile" in query_no_space or "mobileapple" in query_no_space:
        return True, "smartphone"
    if "macbook" in query_no_space or "macbok" in query_no_space:
        return True, "laptops"
    # MacBook Neo - "macbook neo" or "neo macbook" → Apple MacBook Neo
    # NOTE: "neo" alone should show realme GT NEO phones, not MacBook
    if "neo macbook" in query_lower or "macbook neo" in query_lower:
        return True, "laptops"
    if query_lower.startswith("neo ") and any(w in query_words for w in {"laptop", "laptops", "book", "air", "pro", "max"}):
        return True, "laptops"
    if "airpod" in query_no_space or "arpod" in query_no_space:
        return True, "audio video"
    if "ipad" in query_no_space or "ipd" in query_no_space:
        return True, "tablets"
    if "applewatch" in query_no_space or "iwatch" in query_no_space:
        return True, "watch and wearable"
    
    return False, None


def is_apple_iphone_query(query: str) -> bool:
    """
    DEPRECATED: Use is_apple_product_query() instead.
    Kept for backward compatibility.
    
    Detect if query is related to Apple iPhone products ONLY.
    Returns True ONLY for iPhone queries.
    
    IMPORTANT: 
    - "macbook", "airpods", "ipad" should NOT return True
    - These should search their categories from ALL brands
    - Only iPhone/Apple phone queries should return True
    """
    if not query:
        return False
    
    query_lower = query.lower().strip()
    # Remove special characters for matching (but keep spaces)
    query_cleaned = re.sub(r'[^\w\s]', '', query_lower)
    query_no_space = query_lower.replace(" ", "").replace("-", "").replace("_", "").replace("$", "").replace("@", "")
    query_words = set(query_lower.split())
    
    # FIRST: Check if this is a non-phone Apple product
    # If so, return False so it searches the right category from all brands
    if get_apple_non_phone_category(query):
        return False
    
    # CRITICAL: Plain "phone" or "phones" should NOT be Apple query
    # This allows showing all brands when user just searches "phone"
    plain_phone_queries = {"phone", "phones", "mobile", "mobiles", "mobile phone", "mobile phones", 
                           "smartphone", "smartphones", "fone", "fon", "phn", "phon", "phne"}
    if query_lower in plain_phone_queries or query_cleaned in plain_phone_queries:
        return False
    
    # Also check if query is just "phone" with non-Apple brand (like "samsung phone", "vivo phone")
    non_apple_brands = {"samsung", "vivo", "oppo", "realme", "oneplus", "xiaomi", "mi", "redmi", 
                        "poco", "motorola", "nokia", "tecno", "infinix", "iqoo", "nothing", "google", "pixel"}
    for brand in non_apple_brands:
        if brand in query_lower:
            return False
    
    # CRITICAL: "one plus" / "1 plus" / "1plus" are OnePlus phone queries, NOT Apple
    oneplus_patterns = [r'\bone\s*plus', r'\b1\s*plus', r'\b1plus']
    for pattern in oneplus_patterns:
        if re.search(pattern, query_lower):
            return False
    
    # CRITICAL: Samsung Galaxy S-series models should NOT be detected as Apple
    # Patterns like "s23 ultra", "s 24 ultra", "s25 ultra" contain "ultra" but are Samsung
    samsung_galaxy_pattern = r'\bs\s*(?:2[1-9]|3[0-9])\s*(?:ultra|plus|fe)?\b'
    if re.search(samsung_galaxy_pattern, query_lower):
        return False
    
    # Also check for "galaxy" keyword which is Samsung-specific
    if 'galaxy' in query_lower:
        return False
    
    # Exclude non-Apple queries that contain Apple-like substrings
    non_apple_terms = {"machine", "appliance", "appliances", "application", "applications", "washing"}
    if query_words & non_apple_terms:
        return False
    
    # ===== CHECK 1: Direct match in comprehensive noisy variations =====
    if query_lower in IPHONE_NOISY_VARIATIONS:
        return True
    if query_cleaned in IPHONE_NOISY_VARIATIONS:
        return True
    if query_no_space in IPHONE_NOISY_VARIATIONS:
        return True
    
    # ===== CHECK 2: Model-specific noisy patterns =====
    if query_lower in IPHONE_MODEL_NOISY:
        return True
    if query_cleaned in IPHONE_MODEL_NOISY:
        return True
    # Check if any model pattern is contained in query (with word boundaries for short patterns)
    for model_pattern in IPHONE_MODEL_NOISY:
        # For short patterns like "ultra", "plus", "mini", require word boundary
        if len(model_pattern) <= 5:
            if re.search(r'\b' + re.escape(model_pattern) + r'\b', query_lower):
                return True
        else:
            # For longer patterns, allow substring match
            if model_pattern in query_lower or model_pattern in query_no_space:
                return True
    
    # ===== CHECK 3: Exact term match =====
    for term in APPLE_IPHONE_EXACT_TERMS:
        if re.search(r'\b' + re.escape(term) + r'\b', query_lower):
            return True
        term_no_space = term.replace(" ", "")
        if term_no_space in query_no_space and len(term_no_space) >= 5:
            return True
    
    # ===== CHECK 4: Pattern match for typos =====
    for pattern in APPLE_IPHONE_TYPO_PATTERNS:
        if re.search(pattern, query_lower):
            return True
        if re.search(pattern, query_cleaned):
            return True
    
    # ===== CHECK 5: Apple + number pattern =====
    if re.search(APPLE_NUMBER_PATTERN, query_lower):
        return True
    
    # ===== CHECK 6: Fuzzy match for very noisy queries =====
    # Only for words that look like they could be iPhone variations
    excluded_words = {"phone", "phones", "phon", "phne", "phn", "mobile", "mobiles", "android", 
                      "feature", "fetur", "feture", "touch", "camera", "best", "dual", "sim", 
                      "cell", "cellular", "smart", "fone", "fon", "fonr", "the", "and", "for"}
    
    for word in query_words:
        if len(word) >= 4 and word not in excluded_words:
            # Check similarity to "iphone"
            score = fuzz.ratio(word, "iphone")
            if score >= 75:  # 75% similarity
                return True
            # Check without numbers/special chars
            word_clean = re.sub(r'[^a-z]', '', word)
            if word_clean and len(word_clean) >= 3:
                score = fuzz.ratio(word_clean, "iphone")
                if score >= 75:
                    return True
    
    # ===== CHECK 7: Check no-space version for compound queries =====
    # "phone apple" → "phoneapple", check if contains iphone-like pattern
    if "iphone" in query_no_space or "ifone" in query_no_space or "ipone" in query_no_space:
        return True
    if "applephone" in query_no_space or "phoneapple" in query_no_space:
        return True
    if "applemobile" in query_no_space or "mobileapple" in query_no_space:
        return True
    
    return False

def normalize_apple_query(query: str) -> str:
    """
    Normalize Apple product query for better search.
    Fixes common typos, noise, and standardizes format.
    Handles 50+ noisy variations for ALL Apple products:
    - iPhone, iPad, MacBook, AirPods, Apple Watch
    """
    if not query:
        return query
    
    query_lower = query.lower().strip()
    # Remove special characters that are noise (keep alphanumeric and spaces)
    query_cleaned = re.sub(r'[$@#!*&^%]', '', query_lower)
    
    # ===== STEP -1: Standalone iPhone model number normalization =====
    # Handle queries like "16pro max", "17promax", "15 pro max" without "iphone" prefix
    # These are detected as Apple queries by is_apple_product_query() but need "iphone" prefix
    standalone_model_patterns = [
        # iPhone 13-17 variations: "16pro max" → "iphone 16 pro max"
        (r'^(13|14|15|16|17)\s*pro\s*max$', r'iphone \1 pro max'),
        (r'^(13|14|15|16|17)promax$', r'iphone \1 pro max'),
        (r'^(13|14|15|16|17)\s*pro$', r'iphone \1 pro'),
        (r'^(13|14|15|16|17)pro$', r'iphone \1 pro'),
        (r'^(13|14|15|16|17)\s*plus$', r'iphone \1 plus'),
        (r'^(13|14|15|16|17)plus$', r'iphone \1 plus'),
        (r'^(13|14|15|16|17)\s*mini$', r'iphone \1 mini'),
        (r'^(13|14|15|16|17)mini$', r'iphone \1 mini'),
        # Just model number: "16" → "iphone 16"
        (r'^(13|14|15|16|17)$', r'iphone \1'),
        # With "max" only: "16 max", "16max" → "iphone 16 pro max"
        (r'^(13|14|15|16|17)\s*max$', r'iphone \1 pro max'),
        (r'^(13|14|15|16|17)max$', r'iphone \1 pro max'),
        # SE variations
        (r'^se\s*(\d+)?$', r'iphone se'),
    ]
    for pattern, replacement in standalone_model_patterns:
        match = re.match(pattern, query_cleaned, re.IGNORECASE)
        if match:
            result = re.sub(pattern, replacement, query_cleaned, flags=re.IGNORECASE)
            logger.info(f"Standalone iPhone model normalized: '{query}' → '{result}'")
            return result.strip()
        
    # ===== STEP -0.5: iPhone TYPO normalization (before other processing) =====
    # Handle queries like "17iphone", "iphonpromax", "16iphon" detected as Apple
    import re as regex_module
    iphone_typo_patterns = [
        # Number + iphone compound (17iphone -> iphone 17)
        (r'^(1[3-7])iphone$', r'iphone \1'), (r'^(1[3-7])\s*iphone$', r'iphone \1'),
        # iphone + Number compound without space (iphone17 -> iphone 17)
        (r'^iphone(1[3-7])$', r'iphone \1'),
        # Number + iPhone typos (17iphon, 16iphn, 15iphne)
        (r'^(1[3-7])\s*iphon$', r'iphone \1'), (r'^(1[3-7])iphon$', r'iphone \1'),
        (r'^(1[3-7])\s*iphn$', r'iphone \1'), (r'^(1[3-7])iphn$', r'iphone \1'),
        (r'^(1[3-7])\s*iphne$', r'iphone \1'), (r'^(1[3-7])iphne$', r'iphone \1'),
        (r'^(1[3-7])\s*ipone$', r'iphone \1'), (r'^(1[3-7])ipone$', r'iphone \1'),
        (r'^(1[3-7])\s*ifone$', r'iphone \1'), (r'^(1[3-7])ifone$', r'iphone \1'),
        # Typo + number (iphon17, iphn16)
        (r'^iphon\s*(1[3-7])$', r'iphone \1'), (r'^iphon(1[3-7])$', r'iphone \1'),
        (r'^iphn\s*(1[3-7])$', r'iphone \1'), (r'^iphn(1[3-7])$', r'iphone \1'),
        (r'^iphne\s*(1[3-7])$', r'iphone \1'), (r'^iphne(1[3-7])$', r'iphone \1'),
        (r'^ipone\s*(1[3-7])$', r'iphone \1'), (r'^ipone(1[3-7])$', r'iphone \1'),
        (r'^ifone\s*(1[3-7])$', r'iphone \1'), (r'^ifone(1[3-7])$', r'iphone \1'),
        
        # === PRO patterns ===
        # Number + typo + pro (14iphonpro, 15iphnpro)
        (r'^(1[3-7])\s*iphon\s*pro$', r'iphone \1 pro'), (r'^(1[3-7])iphonpro$', r'iphone \1 pro'),
        (r'^(1[3-7])\s*iphn\s*pro$', r'iphone \1 pro'), (r'^(1[3-7])iphnpro$', r'iphone \1 pro'),
        (r'^(1[3-7])\s*iphne\s*pro$', r'iphone \1 pro'), (r'^(1[3-7])iphnepro$', r'iphone \1 pro'),
        (r'^(1[3-7])\s*ipone\s*pro$', r'iphone \1 pro'), (r'^(1[3-7])iponepro$', r'iphone \1 pro'),
        (r'^(1[3-7])\s*ifone\s*pro$', r'iphone \1 pro'), (r'^(1[3-7])ifonepro$', r'iphone \1 pro'),
        # Typo + number + pro (iphon14pro, iphn15pro)
        (r'^iphon\s*(1[3-7])\s*pro$', r'iphone \1 pro'), (r'^iphon(1[3-7])pro$', r'iphone \1 pro'),
        (r'^iphn\s*(1[3-7])\s*pro$', r'iphone \1 pro'), (r'^iphn(1[3-7])pro$', r'iphone \1 pro'),
        (r'^iphne\s*(1[3-7])\s*pro$', r'iphone \1 pro'), (r'^iphne(1[3-7])pro$', r'iphone \1 pro'),
        (r'^ipone\s*(1[3-7])\s*pro$', r'iphone \1 pro'), (r'^ipone(1[3-7])pro$', r'iphone \1 pro'),
        (r'^ifone\s*(1[3-7])\s*pro$', r'iphone \1 pro'), (r'^ifone(1[3-7])pro$', r'iphone \1 pro'),
        
        # === MAX patterns ===
        # Number + typo + max (15iphonmax, 16iphnmax)
        (r'^(1[3-7])\s*iphon\s*max$', r'iphone \1 pro max'), (r'^(1[3-7])iphonmax$', r'iphone \1 pro max'),
        (r'^(1[3-7])\s*iphn\s*max$', r'iphone \1 pro max'), (r'^(1[3-7])iphnmax$', r'iphone \1 pro max'),
        (r'^(1[3-7])\s*iphne\s*max$', r'iphone \1 pro max'), (r'^(1[3-7])iphnemax$', r'iphone \1 pro max'),
        (r'^(1[3-7])\s*ipone\s*max$', r'iphone \1 pro max'), (r'^(1[3-7])iponemax$', r'iphone \1 pro max'),
        (r'^(1[3-7])\s*ifone\s*max$', r'iphone \1 pro max'), (r'^(1[3-7])ifonemax$', r'iphone \1 pro max'),
        # Typo + number + max (iphon15max, iphn16max)
        (r'^iphon\s*(1[3-7])\s*max$', r'iphone \1 pro max'), (r'^iphon(1[3-7])max$', r'iphone \1 pro max'),
        (r'^iphn\s*(1[3-7])\s*max$', r'iphone \1 pro max'), (r'^iphn(1[3-7])max$', r'iphone \1 pro max'),
        (r'^iphne\s*(1[3-7])\s*max$', r'iphone \1 pro max'), (r'^iphne(1[3-7])max$', r'iphone \1 pro max'),
        (r'^ipone\s*(1[3-7])\s*max$', r'iphone \1 pro max'), (r'^ipone(1[3-7])max$', r'iphone \1 pro max'),
        (r'^ifone\s*(1[3-7])\s*max$', r'iphone \1 pro max'), (r'^ifone(1[3-7])max$', r'iphone \1 pro max'),
        
        # === PRO MAX patterns ===
        # Number + typo + promax (15iphonpromax, 16iphnpromax)
        (r'^(1[3-7])\s*iphon\s*pro\s*max$', r'iphone \1 pro max'), (r'^(1[3-7])iphonpromax$', r'iphone \1 pro max'),
        (r'^(1[3-7])\s*iphn\s*pro\s*max$', r'iphone \1 pro max'), (r'^(1[3-7])iphnpromax$', r'iphone \1 pro max'),
        (r'^(1[3-7])\s*iphne\s*pro\s*max$', r'iphone \1 pro max'), (r'^(1[3-7])iphnepromax$', r'iphone \1 pro max'),
        # Typo + number + promax (iphon15promax)
        (r'^iphon\s*(1[3-7])\s*pro\s*max$', r'iphone \1 pro max'), (r'^iphon(1[3-7])promax$', r'iphone \1 pro max'),
        (r'^iphn\s*(1[3-7])\s*pro\s*max$', r'iphone \1 pro max'), (r'^iphn(1[3-7])promax$', r'iphone \1 pro max'),
        
        # === MINI patterns ===
        # Number + typo + mini (13iphonmini, 14iphnmini)
        (r'^(1[3-7])\s*iphon\s*mini$', r'iphone \1 mini'), (r'^(1[3-7])iphonmini$', r'iphone \1 mini'),
        (r'^(1[3-7])\s*iphn\s*mini$', r'iphone \1 mini'), (r'^(1[3-7])iphnmini$', r'iphone \1 mini'),
        (r'^(1[3-7])\s*iphne\s*mini$', r'iphone \1 mini'), (r'^(1[3-7])iphnemini$', r'iphone \1 mini'),
        (r'^(1[3-7])\s*ipone\s*mini$', r'iphone \1 mini'), (r'^(1[3-7])iponemini$', r'iphone \1 mini'),
        (r'^(1[3-7])\s*ifone\s*mini$', r'iphone \1 mini'), (r'^(1[3-7])ifonemini$', r'iphone \1 mini'),
        # Typo + number + mini (iphon13mini)
        (r'^iphon\s*(1[3-7])\s*mini$', r'iphone \1 mini'), (r'^iphon(1[3-7])mini$', r'iphone \1 mini'),
        (r'^iphn\s*(1[3-7])\s*mini$', r'iphone \1 mini'), (r'^iphn(1[3-7])mini$', r'iphone \1 mini'),
        
        # === PLUS patterns ===
        # Number + typo + plus
        (r'^(1[3-7])\s*iphon\s*plus$', r'iphone \1 plus'), (r'^(1[3-7])iphonplus$', r'iphone \1 plus'),
        (r'^(1[3-7])\s*iphn\s*plus$', r'iphone \1 plus'), (r'^(1[3-7])iphnplus$', r'iphone \1 plus'),
        # Typo + number + plus
        (r'^iphon\s*(1[3-7])\s*plus$', r'iphone \1 plus'), (r'^iphon(1[3-7])plus$', r'iphone \1 plus'),
        
        # promax patterns (without number)
        (r'^iphon\s*pro\s*max$', r'iphone pro max'), (r'^iphonpromax$', r'iphone pro max'),
        (r'^iphn\s*pro\s*max$', r'iphone pro max'), (r'^iphnpromax$', r'iphone pro max'),
        (r'^iphne\s*pro\s*max$', r'iphone pro max'), (r'^iphnepromax$', r'iphone pro max'),
        # Just typo + pro
        (r'^iphon\s*pro$', r'iphone pro'), (r'^iphonpro$', r'iphone pro'),
        (r'^iphn\s*pro$', r'iphone pro'), (r'^iphnpro$', r'iphone pro'),
        # Standalone typos
        (r'^iphon$', r'iphone'), (r'^iphn$', r'iphone'), (r'^iphne$', r'iphone'),
        (r'^ipone$', r'iphone'), (r'^ifone$', r'iphone'),
        # === UPHONE typos (keyboard 'u' near 'i') ===
        (r'^uphone$', r'iphone'), (r'^uphon$', r'iphone'), (r'^upone$', r'iphone'), (r'^ufone$', r'iphone'),
        (r'^uphone\s*(1[3-7])$', r'iphone \1'), (r'^uphone(1[3-7])$', r'iphone \1'),
        (r'^(1[3-7])\s*uphone$', r'iphone \1'), (r'^(1[3-7])uphone$', r'iphone \1'),
        (r'^uphone\s*(1[3-7])\s*pro$', r'iphone \1 pro'), (r'^uphone(1[3-7])pro$', r'iphone \1 pro'),
        (r'^uphone\s*(1[3-7])\s*pro\s*max$', r'iphone \1 pro max'), (r'^uphone(1[3-7])promax$', r'iphone \1 pro max'),
        (r'^uphone\s*pro\s*max$', r'iphone pro max'), (r'^uphonepromax$', r'iphone pro max'),
        (r'^uphone\s*pro$', r'iphone pro'), (r'^uphonepro$', r'iphone pro'),
    ]
    for pattern, replacement in iphone_typo_patterns:
        if regex_module.match(pattern, query_cleaned, regex_module.IGNORECASE):
            result = regex_module.sub(pattern, replacement, query_cleaned, flags=regex_module.IGNORECASE)
            logger.info(f"iPhone typo normalized: '{query}' → '{result}'")
            return result.strip()
        
    
    # ===== STEP 0: iPad typo normalization =====
    # Handle iPad typos FIRST before other processing
    ipad_typo_map = {
        "ipd": "ipad", "ipda": "ipad", "ipaad": "ipad", "ipadd": "ipad",
        "ippad": "ipad", "ipda": "ipad", "i pad": "ipad", "i-pad": "ipad",
        "ip ad": "ipad", "ipa d": "ipad", "ipads": "ipad", "ipd pro": "ipad pro",
        "ipda pro": "ipad pro", "i pad pro": "ipad pro", "ipd air": "ipad air",
        "i pad air": "ipad air", "ipd mini": "ipad mini", "i pad mini": "ipad mini"
    }
    if query_cleaned in ipad_typo_map:
        query_lower = ipad_typo_map[query_cleaned]
        return query_lower
    if query_lower.replace(" ", "") in ipad_typo_map:
        query_lower = ipad_typo_map[query_lower.replace(" ", "")]
        return query_lower
    
    # ===== STEP 0b: MacBook typo normalization =====
    macbook_typo_map = {
        "macbok": "macbook", "macbuk": "macbook", "makbook": "macbook",
        "makbok": "macbook", "macbokk": "macbook", "macboo": "macbook",
        "mackbook": "macbook", "mcbook": "macbook", "macboook": "macbook",
        "macbok pro": "macbook pro", "makbook pro": "macbook pro", "mackbook pro": "macbook pro",
        "macbok air": "macbook air", "makbook air": "macbook air", "mackbook air": "macbook air",
        "mac book": "macbook", "mac bok": "macbook"
    }
    if query_cleaned in macbook_typo_map:
        query_lower = macbook_typo_map[query_cleaned]
        return query_lower
    if query_lower.replace(" ", "") in macbook_typo_map:
        query_lower = macbook_typo_map[query_lower.replace(" ", "")]
        return query_lower
    
    # ===== STEP 0b2: MacBook Neo normalization =====
    # "macbook neo", "neo macbook", "neo laptop" etc. → "macbook neo"
    # NOTE: "neo" alone should show realme GT NEO phones, not MacBook
    neo_normalization_map = {
        "neo laptop": "macbook neo", "neo laptops": "macbook neo",
        "neo book": "macbook neo", "neobook": "macbook neo",
        "neo macbook": "macbook neo", "neo mac book": "macbook neo", "neo macbok": "macbook neo",
        "neo air": "macbook neo", "neo pro": "macbook neo", "neo pro max": "macbook neo",
        "apple neo": "macbook neo", "appleneo": "macbook neo",
        "macbok neo": "macbook neo", "mackbook neo": "macbook neo",
        "mac book neo": "macbook neo", "mcbook neo": "macbook neo",
    }
    if query_cleaned in neo_normalization_map:
        result = neo_normalization_map[query_cleaned]
        logger.info(f"MacBook Neo normalized: '{query}' → '{result}'")
        return result
    
    # ===== STEP 0c: AirPods typo normalization =====
    airpods_typo_map = {
        # Basic variations
        "airpod": "airpods", "airpds": "airpods", "arpods": "airpods",
        "arppods": "airpods", "air pods": "airpods", "air pod": "airpods",
        "airepods": "airpods", "airpods pro": "airpods", "airpod pro": "airpods",
        "airpods max": "airpods", "airpod max": "airpods",
        # Missing 'r' variations
        "aipod": "airpods", "aipods": "airpods", "aipod pro": "airpods",
        "aipods pro": "airpods", "aipod max": "airpods", "aipods max": "airpods",
        # Phonetic/typo variations  
        "aiprpod": "airpods", "aiprpods": "airpods", "airppod": "airpods",
        "airppods": "airpods", "airposd": "airpods", "airpodss": "airpods",
        "airpods ": "airpods", "airpod ": "airpods",
        # EarPods (Apple wired earphones - map to airpods for search)
        "earpod": "airpods", "earpods": "airpods", "ear pod": "airpods",
        "ear pods": "airpods", "earpod pro": "airpods", "earpods pro": "airpods",
        # 'ai' prefix variations
        "aiprd": "airpods", "aiprd pro": "airpods", "aiepod": "airpods",
        "aiepods": "airpods", "aieposd": "airpods",
        # Missing letters
        "arpod": "airpods", "airpo": "airpods", "airpd": "airpods",
        "airods": "airpods", "airpod s": "airpods",
        # Keyboard errors
        "airpoda": "airpods", "airpode": "airpods", "airpodi": "airpods",
        "sirpods": "airpods", "sirpod": "airpods",
        # Spaced variations
        "air-pod": "airpods", "air-pods": "airpods", "air_pod": "airpods",
        "air_pods": "airpods", "a ir pods": "airpods",
        # Apple + airpods
        "apple airpod": "airpods", "apple airpods": "airpods",
        "apple earpod": "airpods", "apple earpods": "airpods",
        "apple earphone": "airpods", "apple earphones": "airpods",
        "apple earbuds": "airpods", "apple buds": "airpods",
        # Common search terms
        "apple wireless earphone": "airpods", "apple bluetooth earphone": "airpods",
        "apple tws": "airpods", "apple wireless earbuds": "airpods"
    }
    if query_cleaned in airpods_typo_map:
        query_lower = airpods_typo_map[query_cleaned]
        return query_lower
    if query_lower.replace(" ", "") in airpods_typo_map:
        query_lower = airpods_typo_map[query_lower.replace(" ", "")]
        return query_lower
    # Also check if query starts with any airpods typo pattern
    for typo, norm in airpods_typo_map.items():
        if query_cleaned.startswith(typo + " ") or query_cleaned == typo:
            query_lower = query_cleaned.replace(typo, norm, 1)
            return query_lower
    
    # ===== STEP 0d: Apple Watch typo normalization =====
    watch_typo_map = {
        "iwatch": "apple watch", "i watch": "apple watch", "iwach": "apple watch",
        "aple watch": "apple watch", "apple wach": "apple watch",
        "apple wtch": "apple watch", "applewatch": "apple watch"
    }
    if query_cleaned in watch_typo_map:
        query_lower = watch_typo_map[query_cleaned]
        return query_lower
    if query_lower.replace(" ", "") in watch_typo_map:
        query_lower = watch_typo_map[query_lower.replace(" ", "")]
        return query_lower
    
    # ===== STEP 1: Direct replacement for known noisy variations =====
    # Map noisy queries directly to "iphone"
    noisy_to_iphone = {
        # Very short/noisy
        "iphn": "iphone", "ipn": "iphone", "ifn": "iphone", "iph": "iphone",
        "i phn": "iphone", "i pn": "iphone", "i ph": "iphone", "i-phn": "iphone",
        "i$phn": "iphone", "i$phone": "iphone", "i@phone": "iphone",
        # Spaced variations
        "i phon": "iphone", "i pone": "iphone", "i fone": "iphone", "i fon": "iphone",
        "i phne": "iphone", "ip hone": "iphone", "iph one": "iphone",
        # Phonetic
        "aifon": "iphone", "aiphone": "iphone", "aifone": "iphone", "eyephone": "iphone",
        "eyefone": "iphone", "ephone": "iphone", "eiphone": "iphone", "eifone": "iphone",
        # Common typos
        "iphon": "iphone", "iphne": "iphone", "ipone": "iphone", "ifone": "iphone",
        "ihone": "iphone", "iphoen": "iphone", "iphohe": "iphone", "iphobe": "iphone",
        "iphome": "iphone", "iphonw": "iphone", "iphonr": "iphone", "iphond": "iphone",
        "iphone": "iphone", "iohone": "iphone", "ipbone": "iphone", "iphine": "iphone",
        # Keyboard errors
        "iphoje": "iphone", "iphomd": "iphone", "iphinr": "iphone", "iphpne": "iphone",
        "ipyone": "iphone", "ipgone": "iphone", "ipjone": "iphone",
        # Number pad
        "1phone": "iphone", "iph0ne": "iphone", "iph0n3": "iphone", "1ph0ne": "iphone",
        # Word order
        "phone iphone": "iphone", "phone apple": "iphone", "mobile apple": "iphone",
        "fone apple": "iphone", "phn apple": "iphone",
        # Plural forms
        "iphones": "iphone", "i phones": "iphone",
    }
    
    # Check direct match first
    if query_cleaned in noisy_to_iphone:
        query_lower = noisy_to_iphone[query_cleaned]
    elif query_lower.replace(" ", "") in noisy_to_iphone:
        query_lower = noisy_to_iphone[query_lower.replace(" ", "")]
    else:
        # ===== STEP 2: Pattern-based normalization =====
        iphone_patterns = [
            # Plural variations (must come first to normalize before other patterns)
            (r'\biphones\b', 'iphone'),
            (r'\bi\s*phones\b', 'iphone'),
            # Space variations with model number: "I phone15" → "iphone 15", "I phone 15" → "iphone 15"
            (r'\bi\s+phone\s*(1[3-7])\b', r'iphone \1'),  # "I phone15", "I phone 15"
            (r'\bi\s+phone\s*(1[3-7])\s*(pro|plus|max|mini)\b', r'iphone \1 \2'),  # "I phone15 pro"
            (r'\bi\s+phone\s*(1[3-7])\s*pro\s*max\b', r'iphone \1 pro max'),  # "I phone15 pro max"
            # Space variations without model number
            (r'\bi\s+phone\b', 'iphone'),
            (r'\bi[-_]phone\b', 'iphone'),
            (r'\bi\s*ph\s*one\b', 'iphone'),
            (r'\bip\s+hone\b', 'iphone'),
            # Typos
            (r'\biphn\b', 'iphone'),
            (r'\bifone\b', 'iphone'),
            (r'\bipone\b', 'iphone'),
            (r'\bipon\b', 'iphone'),
            (r'\biphon\b', 'iphone'),
            (r'\beyephone\b', 'iphone'),
            (r'\biphne\b', 'iphone'),
            (r'\bipohne\b', 'iphone'),
            (r'\biphoen\b', 'iphone'),
            (r'\baphone\b', 'iphone'),
            (r'\biphone\b', 'iphone'),
            (r'\biphome\b', 'iphone'),
            (r'\biphobe\b', 'iphone'),
            # Phonetic
            (r'\baifon[e]?\b', 'iphone'),
            (r'\beifon[e]?\b', 'iphone'),
            (r'\beiphone\b', 'iphone'),
            # Number pad errors
            (r'\b1phone\b', 'iphone'),
            (r'\biph0ne\b', 'iphone'),
            # === UPHONE typos (keyboard 'u' near 'i') ===
            (r'\buphone\b', 'iphone'),
            (r'\buphon\b', 'iphone'),
            (r'\bupone\b', 'iphone'),
            (r'\bufone\b', 'iphone'),
        ]
        
        for pattern, replacement in iphone_patterns:
            query_lower = re.sub(pattern, replacement, query_lower)
    
    # ===== STEP 3: Apple variations =====
    apple_patterns = [
        (r'\bappel\b', 'apple'),
        (r'\baple\b', 'apple'),
        (r'\bapplle\b', 'apple'),
        (r'\baplle\b', 'apple'),
        (r'\bappl\b', 'apple'),
        (r'\bap\s*ple\b', 'apple'),
        (r'\bapp\s*le\b', 'apple'),
        (r'\bapel\b', 'apple'),
        (r'\bapal\b', 'apple'),
    ]
    
    for pattern, replacement in apple_patterns:
        query_lower = re.sub(pattern, replacement, query_lower)
    
    # ===== STEP 3.5: Phone typo normalization =====
    # Handle "phn", "fone", etc. as "phone" - but only for display, not for filtering
    # These are common keyboard shortcuts/typos that should be corrected
    phone_typo_patterns = [
        (r'\bphn\b', 'phone'),
        (r'\bphne\b', 'phone'),
        (r'\bphon\b', 'phone'),
        (r'\bfon\b', 'phone'),
        (r'\bfone\b', 'phone'),
    ]
    for pattern, replacement in phone_typo_patterns:
        query_lower = re.sub(pattern, replacement, query_lower)
    
    # ===== STEP 4: Apple phone → iphone =====
    query_lower = re.sub(r'\bapple\s+(phone|mobile|fone|phn|phon)\b', 'iphone', query_lower)
    query_lower = re.sub(r'\b(phone|mobile|fone|phn)\s+apple\b', 'iphone', query_lower)
    query_lower = re.sub(r'\bappl\s+(phone|phn|fone)\b', 'iphone', query_lower)
    
    # Add space between iphone and number if missing (iphone15 -> iphone 15)
    query_lower = re.sub(r'(iphone)(\d+)', r'\1 \2', query_lower)
    
    # Convert "apple 15" to "iphone 15" for search
    query_lower = re.sub(r'apple\s+(\d+)\s*(pro|max|plus|mini|ultra)?', r'iphone \1 \2', query_lower)
    
    # Plain "apple" → "iphone" for search
    if query_lower.strip() == 'apple':
        query_lower = 'iphone'
    
    # ===== PlayStation normalization =====
    # "ps 5" → "ps5", "ps 4" → "ps4" for better product matching
    query_lower = re.sub(r'\bps\s+5\b', 'ps5', query_lower)
    query_lower = re.sub(r'\bps\s+4\b', 'ps4', query_lower)
    query_lower = re.sub(r'\bplay\s+station\b', 'playstation', query_lower)
    
    # Clean up extra spaces
    query_lower = re.sub(r'\s+', ' ', query_lower).strip()
    
    return query_lower

# Build category filter exclusions with variations
CATEGORY_FILTER_EXCLUSIONS_LOWER = {}
for category, exclusions in CATEGORY_FILTER_EXCLUSIONS.items():
    CATEGORY_FILTER_EXCLUSIONS_LOWER[category.lower()] = exclusions
    if category == "air conditioner":
        CATEGORY_FILTER_EXCLUSIONS_LOWER["ac"] = exclusions
    elif category == "television":
        CATEGORY_FILTER_EXCLUSIONS_LOWER["tv"] = exclusions
    elif category == "washing machines":
        CATEGORY_FILTER_EXCLUSIONS_LOWER["washing machine"] = exclusions
    elif category == "new cars":
        CATEGORY_FILTER_EXCLUSIONS_LOWER["car"] = exclusions
        CATEGORY_FILTER_EXCLUSIONS_LOWER["cars"] = exclusions
    elif category == "two-wheeler":
        CATEGORY_FILTER_EXCLUSIONS_LOWER["two wheeler"] = exclusions
        CATEGORY_FILTER_EXCLUSIONS_LOWER["twowheeler"] = exclusions
        CATEGORY_FILTER_EXCLUSIONS_LOWER["bike"] = exclusions
        CATEGORY_FILTER_EXCLUSIONS_LOWER["scooter"] = exclusions


# =================== COMPREHENSIVE CORRECTED QUERY SYSTEM ===================
# This system generates user-friendly corrected queries for UI display
# It handles: typos, short forms, noise, canonical names

# ===== BRAND TYPO CORRECTIONS =====
# Maps typos/variations to canonical brand names
BRAND_TYPO_CORRECTIONS = {
    # Samsung - 15+ variations
    "samung": "Samsung", "samsun": "Samsung", "samsang": "Samsung", "samsumg": "Samsung",
    "sumsang": "Samsung", "sumsung": "Samsung", "samsng": "Samsung", "samusng": "Samsung",
    "smasung": "Samsung", "samsyng": "Samsung", "samsungg": "Samsung", "samsug": "Samsung",
    "samsaung": "Samsung", "samsuung": "Samsung", "samsunf": "Samsung",
    
    # Vivo - 10+ variations
    "vevo": "Vivo", "viovo": "Vivo", "vivi": "Vivo", "vvo": "Vivo", "voivo": "Vivo",
    "vivio": "Vivo", "vovo": "Vivo", "vivoo": "Vivo", "vivp": "Vivo", "viivo": "Vivo",
    "viv": "Vivo", "vio": "Vivo",
    
    # Oppo - 8+ variations
    "opp0": "Oppo", "opo": "Oppo", "opoo": "Oppo", "0ppo": "Oppo", "oppp": "Oppo",
    "oppoo": "Oppo", "opppo": "Oppo", "oppo0": "Oppo", "opp": "Oppo", "op": "Oppo",
    
    # Realme - 10+ variations
    "relme": "Realme", "realmee": "Realme", "relame": "Realme", "reamle": "Realme",
    "reame": "Realme", "realmy": "Realme", "realmei": "Realme", "realmi": "Realme",
    "reelme": "Realme", "rielme": "Realme",
    
    # Redmi - 6+ variations
    "redmii": "Redmi", "radmi": "Redmi", "ridmi": "Redmi", "redmmi": "Redmi",
    "redme": "Redmi", "redmie": "Redmi",
    
    # Xiaomi/Mi - 8+ variations
    "xiomi": "Xiaomi", "xaomi": "Xiaomi", "xiaomii": "Xiaomi", "xioami": "Xiaomi",
    "xiami": "Xiaomi", "xiaome": "Xiaomi", "shaomi": "Xiaomi", "xaiomi": "Xiaomi",
    
    # Poco - 5+ variations
    "pocco": "Poco", "poko": "Poco", "pocoo": "Poco", "pokko": "Poco", "pooco": "Poco",
    
    # OnePlus - 8+ variations
    "1plus": "OnePlus", "one+": "OnePlus", "onplus": "OnePlus", "onepls": "OnePlus",
    "oneplus+": "OnePlus", "onepluss": "OnePlus", "1+": "OnePlus", "one plus": "OnePlus",
    "oneplus": "OnePlus",  # lowercase correct spelling
    
    # Motorola - 6+ variations
    "motarola": "Motorola", "motorla": "Motorola", "motorolla": "Motorola",
    "motorala": "Motorola", "moto": "Motorola", "motolora": "Motorola",
    
    # Nokia - 5+ variations
    "nokiya": "Nokia", "noika": "Nokia", "nokiaa": "Nokia", "noka": "Nokia", "nokla": "Nokia",
    
    # Apple/iPhone - handled separately in iPhone system
    "appel": "Apple", "aple": "Apple", "applle": "Apple", "aplle": "Apple",
    "appl": "Apple", "apel": "Apple", "apal": "Apple",
    
    # LG - 3+ variations
    "elg": "LG", "l g": "LG", "lg.": "LG", "lg": "LG",  # lowercase correct spelling
    
    # HP - uppercase brand
    "hp": "HP", "h p": "HP", "hp.": "HP",  # lowercase correct spelling
    
    # Sony - 4+ variations
    "soni": "Sony", "sonny": "Sony", "soney": "Sony", "soniy": "Sony",
    
    # Panasonic - 5+ variations
    "panasnic": "Panasonic", "panasoic": "Panasonic", "panasonc": "Panasonic",
    "panasoinc": "Panasonic", "panasonik": "Panasonic",
    
    # Philips - 5+ variations
    "philps": "Philips", "phillips": "Philips", "philipes": "Philips",
    "philps": "Philips", "phillps": "Philips",
    
    # Haier - 4+ variations
    "haior": "Haier", "haire": "Haier", "haiar": "Haier", "haeir": "Haier",
    
    # Whirlpool - 5+ variations
    "whirlpol": "Whirlpool", "whirpool": "Whirlpool", "whirpol": "Whirlpool",
    "whrilpool": "Whirlpool", "wirlpool": "Whirlpool",
    
    # Godrej - 4+ variations
    "godraj": "Godrej", "godeg": "Godrej", "godreg": "Godrej", "godrj": "Godrej",
    
    # Bosch - 4+ variations
    "bosh": "Bosch", "bosche": "Bosch", "boach": "Bosch", "boch": "Bosch",
    
    # IFB - 3+ variations
    "ifbb": "IFB", "i f b": "IFB", "ifb.": "IFB", "ifb": "IFB",
    
    # Voltas - 4+ variations
    "voltaas": "Voltas", "voltes": "Voltas", "voltas": "Voltas", "voltass": "Voltas",
    
    # Daikin - 4+ variations
    "dakin": "Daikin", "daikn": "Daikin", "daiikan": "Daikin", "daikin": "Daikin",
    
    # Lloyd - 3+ variations
    "loyd": "Lloyd", "llloyd": "Lloyd", "llyod": "Lloyd",
    
    # Blue Star - 3+ variations
    "bluestar": "Blue Star", "blue starr": "Blue Star", "blustar": "Blue Star",
    
    # Boat - 3+ variations
    "boatt": "Boat", "bboat": "Boat", "boaat": "Boat",
    
    # Lenovo - 4+ variations
    "lenova": "Lenovo", "lenoov": "Lenovo", "lenov": "Lenovo", "lenvo": "Lenovo",
    
    # HP - 3+ variations
    "hewlett packard": "HP", "h p": "HP", "hp.": "HP",
    
    # Dell - 3+ variations
    "del": "Dell", "dell.": "Dell", "del": "Dell",
    
    # Asus - 4+ variations
    "asus": "Asus", "assus": "Asus", "asuss": "Asus", "aasus": "Asus",
    
    # Acer - 3+ variations
    "acer": "Acer", "accer": "Acer", "aser": "Acer",
    
    # Kent - 3+ variations
    "kentt": "Kent", "knt": "Kent", "ken": "Kent",
    
    # Livpure - 3+ variations
    "livpur": "Livpure", "livepure": "Livpure", "livpuree": "Livpure",
    
    # Racold - 3+ variations
    "racod": "Racold", "racld": "Racold", "raccold": "Racold",
    
    # AO Smith - 3+ variations
    "aosmith": "AO Smith", "ao smth": "AO Smith", "a o smith": "AO Smith",
    
    # Havells - 3+ variations
    "havels": "Havells", "havels": "Havells", "havvels": "Havells",
    
    # ===== VEHICLE BRANDS =====
    # Honda - 8+ variations
    "hoda": "Honda", "hondaa": "Honda", "honada": "Honda", "hondha": "Honda",
    "hona": "Honda", "honds": "Honda", "honfa": "Honda", "hnda": "Honda",
    "hond": "Honda", "hnd": "Honda",
    
    # Hero - 5+ variations
    "heroo": "Hero", "heero": "Hero", "hiro": "Hero", "herro": "Hero", "hearo": "Hero",
    "hro": "Hero", "her": "Hero",
    
    # TVS - 4+ variations (careful: TVS is brand, not TV)
    "tves": "TVS", "tvss": "TVS", "tvs.": "TVS", "t v s": "TVS", "tvs": "TVS",
    
    # Yamaha - 5+ variations
    "yamha": "Yamaha", "yahmaha": "Yamaha", "yamhaa": "Yamaha", "yaamaha": "Yamaha", "yamaha": "Yamaha",
    
    # Suzuki - 4+ variations
    "suzki": "Suzuki", "suzuuki": "Suzuki", "suzkui": "Suzuki", "syzuki": "Suzuki",
    
    # Royal Enfield - 6+ variations
    "royalenfield": "Royal Enfield", "royal enfild": "Royal Enfield", "royalenfiled": "Royal Enfield",
    "royel enfield": "Royal Enfield", "royal enfiled": "Royal Enfield", "royelenfield": "Royal Enfield",
    
    # Bajaj - 5+ variations
    "bajjaj": "Bajaj", "bajaaj": "Bajaj", "bajja": "Bajaj", "bajajj": "Bajaj", "baaj": "Bajaj",
    "bajj": "Bajaj", "baj": "Bajaj", "bajaj": "Bajaj",
    
    # ===== CAR BRANDS =====
    # Hyundai - 6+ variations
    "hundai": "Hyundai", "hyundia": "Hyundai", "hyundaii": "Hyundai",
    "hyudai": "Hyundai", "hyunadi": "Hyundai", "hundayi": "Hyundai",
    
    # Maruti - 6+ variations
    "maruthi": "Maruti", "marutii": "Maruti", "marutti": "Maruti",
    "marut": "Maruti", "maruti suzuki": "Maruti Suzuki", "marutisuuzuki": "Maruti Suzuki",
    
    # Tata - 4+ variations
    "tatta": "Tata", "taata": "Tata", "tat": "Tata", "tataa": "Tata",
    
    # Mahindra - 4+ variations
    "mahendra": "Mahindra", "mahindara": "Mahindra", "mahindr": "Mahindra", "mahindhra": "Mahindra",
    
    # Kia - 3+ variations
    "kiya": "Kia", "kiaa": "Kia", "keya": "Kia",
    
    # Toyota - 4+ variations
    "toyata": "Toyota", "toyotaa": "Toyota", "toyta": "Toyota", "toyoota": "Toyota",
    
    # Skoda - 3+ variations
    "skodaa": "Skoda", "skods": "Skoda", "scoda": "Skoda",
    
    # MG - 3+ variations
    "m g": "MG", "mg.": "MG", "emg": "MG",
    
    # ===== TRACTOR BRANDS =====
    "swaraaj": "Swaraj", "swraj": "Swaraj", "swaraaj": "Swaraj",
    "johndeere": "John Deere", "john deer": "John Deere", "jhon deere": "John Deere",
    "jhondeere": "John Deere", "johndeer": "John Deere",
    "masseyferguson": "Massey Ferguson", "massey fergusson": "Massey Ferguson",
    "sonalica": "Sonalika", "sonalka": "Sonalika",
}

# ===== CATEGORY CANONICAL NAMES FOR UI DISPLAY =====
# Maps various category inputs to clean UI-friendly display names
# These are the EXACT display names for corrected_query
# IMPORTANT: Correct typos but keep original word meaning, not category mapping
# e.g., "phon" → "Phone" (not "Smartphone"), "scooty" → "Scooter" (not "Two Wheeler")
CATEGORY_CANONICAL_NAMES = {
    # ===== SMARTPHONES / MOBILES =====
    # Correct word is "Phone" or "Mobile", not category name "Smartphone"
    "phone": None, "phones": None,  # Correct - no change needed
    "mobile": None, "mobiles": None,  # Correct - no change needed
    "smartphone": None, "smartphones": None,  # Correct - no change needed
    "smart phone": None,  # Correct
    "phn": "Phone", "phne": "Phone", "phon": "Phone", "fone": "Phone", "fon": "Phone",
    "mbl": "Mobile", "mobl": "Mobile", "mobil": "Mobile", "mob": "Mobile",
    "mobail": "Mobile", "smrtphone": "Smartphone", "smartfone": "Smartphone",
    "moblie": "Mobile", "moble": "Mobile", "mobie": "Mobile", "mobole": "Mobile",
    
    # ===== TELEVISION =====
    "tv": "TV",  # Short form is fine, just capitalize
    "tvs": None,  # TVS is a brand, don't change
    "television": None,  # Correct - no change needed
    "telivision": "Television", "telvision": "Television",
    "televison": "Television", "televisn": "Television", "televishion": "Television",
    "televisoin": "Television", "televsion": "Television", "televesion": "Television",
    "led tv": None, "smart tv": None, "oled tv": None, "qled tv": None,  # Correct
    "led": None, "oled": None, "tv and home entertainment": None,
    
    # ===== AIR CONDITIONER =====
    "ac": "AC",  # Short form is fine, just capitalize
    "acs": "AC", "a.c": "AC", "a.c.": "AC",
    "airconditioner": "Air Conditioner", "aircon": "Air Conditioner", 
    "air conditioner": None,  # Correct - no change needed
    "aircondtioner": "Air Conditioner", "aircondionter": "Air Conditioner", "airconditoner": "Air Conditioner",
    "split ac": None, "window ac": None, "inverter ac": None,  # Correct
    "ac mbk": "AC",
    
    # ===== REFRIGERATOR =====
    "fridge": None,  # Correct - fridge is valid word
    "frig": "Fridge", "frdge": "Fridge", "fridg": "Fridge",
    "refrigerator": None, "refrigerators": None,  # Correct - no change needed
    "refridgerator": "Refrigerator", "refregerator": "Refrigerator", "refrgrator": "Refrigerator",
    "refrigerater": "Refrigerator", "refridgerater": "Refrigerator", "refrigrater": "Refrigerator",
    "ref": "Refrigerator", "double door fridge": None, "single door fridge": None,
    
    # ===== WASHING MACHINE =====
    "wm": "Washing Machine", "washer": None, "washingmachine": "Washing Machine",
    "washing machine": None, "washing machines": None,  # Correct - no change needed
    "washing machin": "Washing Machine", "front load": None, "top load": None,
    "washng machine": "Washing Machine", "washinng machine": "Washing Machine",
    "washing machne": "Washing Machine", "washingmachne": "Washing Machine",
    "washing mashine": "Washing Machine", "washingmashine": "Washing Machine",
    
    # ===== LAPTOP =====
    "laptop": None, "laptops": None,  # Correct - no change needed
    "lptop": "Laptop", "laptp": "Laptop", "lapto": "Laptop",
    "laoptop": "Laptop", "laptoop": "Laptop", "lapop": "Laptop", "loptop": "Laptop",
    "notebook": None, "notbook": "Notebook", "notebk": "Notebook",
    
    # ===== WATER HEATER / GEYSER =====
    "geyser": None, "geysers": None,  # Correct - no change needed
    "geysar": "Geyser", "gayser": "Geyser", "giser": "Geyser", "gizer": "Geyser",
    "gijar": "Geyser", "gisar": "Geyser", "gyser": "Geyser",
    "water heater": None, "waterheater": "Water Heater",  # Correct
    "water heaters and geysers": None,
    
    # ===== WATER PURIFIER =====
    "ro": "RO", "water purifier": None, "waterpurifier": "Water Purifier",
    "water purifir": "Water Purifier", "purifier": None,
    
    # ===== AIR COOLER =====
    "cooler": None, "air cooler": None, "air coolers": None,  # Correct - no change needed
    "aircooler": "Air Cooler", "desert cooler": None,
    
    # ===== AIR FRYER =====
    "air fryer": None, "airfryer": "Air Fryer", "air frier": "Air Fryer",
    "airfrier": "Air Fryer", "air fryers": None,
    
    # ===== AIR PURIFIER =====
    "air purifier": None, "air purifiers": None,  # Correct
    "airpurifier": "Air Purifier",
    
    # ===== MICROWAVE OVEN =====
    "microwave": None, "micro wave": "Microwave",  # Correct
    "microwaveoven": "Microwave Oven", "microwave oven": None,
    "microwave ovens": None, "oven": None,
    "micorwave": "Microwave", "micowave": "Microwave", "microave": "Microwave",
    
    # ===== TWO WHEELER =====
    # Keep original words, don't map to "Two Wheeler"
    "bike": None, "bikes": None, "motorcycle": None,  # Correct - no change
    "motorbike": None, "scooter": None,  # Correct - no change
    "scooty": "Scooter", "scooti": "Scooter", "scoty": "Scooter",  # Typo correction
    "two wheeler": None, "two-wheeler": None,  # Correct
    "twowheeler": "Two Wheeler", "2 wheeler": None,
    
    # ===== CAR =====
    "car": None, "cars": None, "new car": None, "new cars": None,  # Correct
    "sedan": None, "suv": None, "hatchback": None,
    
    # ===== TRACTOR =====
    "tractor": None, "tractors": None,  # Correct - no change needed
    "tactor": "Tractor", "tractr": "Tractor", "tracktor": "Tractor",
    
    # ===== KITCHEN APPLIANCES =====
    "kitchen appliances": None, "kitchen appliance": None,  # Correct
    "mixer": None, "mixar": "Mixer", "miksi": "Mixer",
    "mikser": "Mixer", "mixer grinder": None,
    "juicer": None, "jusar": "Juicer", "juisar": "Juicer", "jucer": "Juicer",
    "chimney": None, "chimny": "Chimney",
    "electric kettle": None, "electric kettles": None,
    
    # ===== DISHWASHER =====
    "dishwasher": None, "dishwashers": None, "dish washer": None,  # Correct
    
    # ===== VACUUM CLEANER =====
    "vacuum cleaner": None, "vacuum cleaners": None,  # Correct
    "vacuumcleaner": "Vacuum Cleaner", "vacuum": None,
    
    # ===== AUDIO VIDEO =====
    "audio & video": None, "audio and video": None, "audio video": None,  # Correct
    "earphone": None, "headphone": None, "speaker": None,
    "home entertainment systems": None,
    
    # ===== CAMERA =====
    "camera": None, "camera & accessories": None, "camera and accessories": None,  # Correct
    "camera accessories": None, "dslr": None,
    
    # ===== WATCHES / WEARABLES =====
    "watch": None, "watches": None,  # Correct - no change needed
    "smartwatch": "Smart Watch", "smart watch": None,  # Correct
    "watches and wearables": None, "watch and wearable": None,
    
    # ===== TABLET =====
    "tablet": None, "tablets": None,  # Correct
    
    # ===== GAMING =====
    "gaming": None, "gaming and accessories": None, "gaming accessories": None,  # Correct
    
    # ===== FURNITURE =====
    "sofa": None, "sopha": "Sofa", "soffa": "Sofa", "couch": None,  # Correct
    "mattress": None, "mattresses": None, "matress": "Mattress", "mattres": "Mattress",
    "bed": None, "beds": None, "furniture": None,  # Correct
    "farnichar": "Furniture", "furnichar": "Furniture", "furnitur": "Furniture",
    
    # ===== CYCLE =====
    "cycle": None, "cycles": None, "bicycle": None,  # Correct
    
    # ===== INVERTER =====
    "inverter": None, "inverters": None,  # Correct
    
    # ===== PRINTER =====
    "printer": None, "printers": None,  # Correct
    
    # ===== PROJECTOR =====
    "projector": None, "projectors": None,  # Correct
    
    # ===== DESKTOP / MONITOR =====
    "desktop": None, "monitor": None, "desktop monitor": None,  # Correct
    
    # ===== TYRE =====
    "tyre": None, "tyres": None, "tire": None, "tires": None,  # Correct
    
    # ===== FLOUR MILL =====
    "flour mill": None, "flour mills": None,  # Correct
    "atta chakki": None, "atta chakki machines": None,
    
    # ===== WATER DISPENSER =====
    "water dispenser": None, "water dispensers": None,  # Correct
    
    # ===== MASSAGER =====
    "massager": None, "massagers": None,  # Correct
    
    # ===== SPORTS FITNESS =====
    "sports & fitness equipment": None, "sports and fitness equipment": None,  # Correct
    "sports fitness equipment": None, "sports fitness": None,
    
    # ===== WINE COOLER =====
    "wine cooler": None, "wine coolers": None,  # Correct
    
    # ===== VISI COOLER =====
    "visi cooler": None,  # Correct
    
    # ===== BINOCULARS =====
    "binoculars": None, "binocular": None,  # Correct
    
    # ===== MUSICAL INSTRUMENTS =====
    "musical instruments": None, "musical instrument": None,  # Correct
    
    # ===== HEALTH CARE =====
    "health and personal care appliances": None,
    "health and personal care appliance": None,
    
    # ===== HOME APPLIANCE =====
    "home appliances": None, "home appliance": None,  # Correct
    
    # ===== TRAVEL =====
    "travel and accessories": None,  # Correct
    
    # ===== WALL ART =====
    "wall art": None,  # Correct
    
    # ===== ELECTRONICS =====
    "electronics": None,  # Correct
}

# ===== MODEL NAME CORRECTIONS =====
# Maps model typos to canonical model names
MODEL_TYPO_CORRECTIONS = {
    # ===== APPLE PRODUCTS =====
    # iPhone models
    "iphone": "iPhone", "iphon": "iPhone", "ipone": "iPhone", "ifone": "iPhone",
    "i phone": "iPhone", "i-phone": "iPhone", "iphne": "iPhone", "iphn": "iPhone",
    "iphobe": "iPhone", "iphome": "iPhone", "aifon": "iPhone", "aiphone": "iPhone",
    "eyephone": "iPhone", "aifone": "iPhone", "ipohne": "iPhone",
    "uphone": "iPhone", "uphon": "iPhone", "upone": "iPhone", "ufone": "iPhone",  # keyboard typo (u near i)
    
    # iPad models
    "ipad": "iPad", "ipd": "iPad", "ipda": "iPad", "i pad": "iPad", "i-pad": "iPad",
    "ipaad": "iPad", "ipadd": "iPad", "ippad": "iPad",
    
    # MacBook models
    "macbook": "MacBook", "macbok": "MacBook", "macbuk": "MacBook", "makbook": "MacBook",
    "makbok": "MacBook", "mac book": "MacBook", "macboo": "MacBook", "macbookk": "MacBook",
    "macboook": "MacBook", "mcbook": "MacBook", "mackbook": "MacBook",
    
    # iMac models
    "imac": "iMac", "imak": "iMac", "i mac": "iMac", "i-mac": "iMac", "imacc": "iMac",
    
    # AirPods models
    "airpods": "AirPods", "airpod": "AirPods", "aipod": "AirPods", "aipods": "AirPods",
    "aiprpod": "AirPods", "aiprpods": "AirPods", "arpod": "AirPods", "arpods": "AirPods",
    "airpds": "AirPods", "airpos": "AirPods", "airppods": "AirPods", "airppod": "AirPods",
    "air pods": "AirPods", "air pod": "AirPods", "earpod": "AirPods", "earpods": "AirPods",
    "ear pods": "AirPods", "ear pod": "AirPods", "aiepod": "AirPods", "aiepods": "AirPods",
    
    # Apple Watch
    "iwatch": "Apple Watch", "i watch": "Apple Watch", "iwach": "Apple Watch",
    "applewatch": "Apple Watch", "apple watch": "Apple Watch",
    
    # Mac Mini - all variations and typos
    "mac mini": "Mac Mini", "macmini": "Mac Mini", "mac-mini": "Mac Mini",
    "mini mac": "Mac Mini", "minimac": "Mac Mini", "mini-mac": "Mac Mini",
    "mac min": "Mac Mini", "macmin": "Mac Mini", "mac mni": "Mac Mini",
    "macmni": "Mac Mini", "mac miin": "Mac Mini", "mac mnii": "Mac Mini",
    "mac minni": "Mac Mini", "mac minii": "Mac Mini", "macminni": "Mac Mini",
    "mac minie": "Mac Mini", "mack mini": "Mac Mini", "mack minni": "Mac Mini",
    "mackmini": "Mac Mini", "mini mack": "Mac Mini", "minni mac": "Mac Mini",
    "mni mac": "Mac Mini", "mini mc": "Mac Mini", "minii mac": "Mac Mini",
    
    # Mac Studio
    "mac studio": "Mac Studio", "macstudio": "Mac Studio", "mac-studio": "Mac Studio",
    "mack studio": "Mac Studio", "mac studeo": "Mac Studio", "mac stdio": "Mac Studio",
    
    # ===== TWO WHEELER MODELS =====
    "activa": "Activa", "actva": "Activa", "aktiva": "Activa", "activaa": "Activa",
    "acktiva": "Activa", "activia": "Activa", "activ": "Activa", "acteva": "Activa",
    "ectiva": "Activa", "aktva": "Activa", "akitva": "Activa",
    "splendor": "Splendor", "splendr": "Splendor", "splendour": "Splendor", "splendar": "Splendor",
    "splender": "Splendor", "splndr": "Splendor", "splandor": "Splendor",
    "jupiter": "Jupiter", "jupter": "Jupiter", "jupitr": "Jupiter", "jupitar": "Jupiter",
    "juipter": "Jupiter", "jupitor": "Jupiter", "jupeter": "Jupiter",
    "pulsar": "Pulsar", "pulser": "Pulsar", "pulsr": "Pulsar", "pulsaar": "Pulsar",
    "pulzar": "Pulsar", "pulser": "Pulsar", "puslar": "Pulsar",
    "bullet": "Bullet", "bullt": "Bullet", "bullat": "Bullet", "bulet": "Bullet",
    "bullit": "Bullet", "bulllet": "Bullet",
    "classic": "Classic", "clasic": "Classic", "classik": "Classic", "classix": "Classic",
    "klasic": "Classic", "classick": "Classic",
    "ntorq": "NTorq", "ntork": "NTorq", "n torq": "NTorq", "entorq": "NTorq",
    "access": "Access", "acces": "Access", "acess": "Access", "axcess": "Access",
    "fascino": "Fascino", "facino": "Fascino", "fassino": "Fascino", "fasino": "Fascino",
    "dio": "Dio", "diio": "Dio", "deio": "Dio",
    "pleasure": "Pleasure", "pleasur": "Pleasure", "plesure": "Pleasure", "plezure": "Pleasure",
    "destini": "Destini", "destiny": "Destini", "desitni": "Destini", "destny": "Destini",
    "raider": "Raider", "raidar": "Raider", "rayder": "Raider", "ryder": "Raider",
    "shine": "Shine", "shien": "Shine", "shin": "Shine", "shayne": "Shine",
    "unicorn": "Unicorn", "unicron": "Unicorn", "unicrn": "Unicorn", "uincorn": "Unicorn",
    "fz": "FZ", "fzs": "FZ-S", "fz s": "FZ-S",
    "r15": "R15", "r 15": "R15",
    "mt15": "MT 15", "mt 15": "MT 15", "mt15": "MT 15",
    "chetak": "Chetak", "cheatak": "Chetak", "chetk": "Chetak", "chetek": "Chetak",
    "ather": "Ather", "athar": "Ather", "aather": "Ather",
    "ola": "Ola", "ola s1": "Ola S1", "oola": "Ola",
    # Royal Enfield models
    "hunter": "Hunter", "huntr": "Hunter", "huntar": "Hunter", "huntter": "Hunter",
    "meteor": "Meteor", "metor": "Meteor", "meetor": "Meteor",
    "himalayan": "Himalayan", "himalayn": "Himalayan", "himlayan": "Himalayan",
    "interceptor": "Interceptor", "intercptor": "Interceptor",
    "continental": "Continental", "contintal": "Continental",
    # TVS models
    "apache": "Apache", "apche": "Apache", "apachi": "Apache",
    "ntorq": "NTorq", "ntork": "NTorq",
    "ronin": "Ronin", "roinin": "Ronin",
    
    # Car models
    "creta": "Creta", "creat": "Creta", "creata": "Creta", "cretta": "Creta",
    "nexon": "Nexon", "nexn": "Nexon", "nexxon": "Nexon", "nexxn": "Nexon",
    "swift": "Swift", "swft": "Swift", "swfit": "Swift", "swiftt": "Swift",
    "city": "City", "cirty": "City", "citty": "City", "cyti": "City", "citi": "City",
    "cocty": "City", "citiy": "City",
    "seltos": "Seltos", "sletos": "Seltos", "seltoss": "Seltos", "celtos": "Seltos",
    "venue": "Venue", "venu": "Venue", "vanue": "Venue", "vinue": "Venue",
    "brezza": "Brezza", "breza": "Brezza", "breeza": "Brezza", "breza": "Brezza",
    "baleno": "Baleno", "beleno": "Baleno", "balenno": "Baleno",
    "thar": "Thar", "tharr": "Thar", "thaar": "Thar",
    "fortuner": "Fortuner", "fortunar": "Fortuner", "fortnr": "Fortuner", "fortner": "Fortuner",
    "innova": "Innova", "inova": "Innova", "innva": "Innova", "inovva": "Innova",
    "punch": "Punch", "puch": "Punch", "punhc": "Punch",
    "tiago": "Tiago", "tago": "Tiago", "tiagoo": "Tiago",
    "tigor": "Tigor", "tigr": "Tigor", "tiggor": "Tigor",
    "harrier": "Harrier", "harier": "Harrier", "harriar": "Harrier",
    "safari": "Safari", "safri": "Safari", "safarii": "Safari",
    "altroz": "Altroz", "altrz": "Altroz", "altros": "Altroz",
    "dzire": "Dzire", "dezire": "Dzire", "dzir": "Dzire", "desire": "Dzire",
    "scorpio": "Scorpio", "scorpion": "Scorpio", "scorpeo": "Scorpio", "scorpoi": "Scorpio",
    "xuv700": "XUV700", "xuv 700": "XUV700", "xuv7oo": "XUV700",
    "xuv300": "XUV300", "xuv 300": "XUV300",
    "sonet": "Sonet", "sonett": "Sonet", "sonnet": "Sonet",
    "carens": "Carens", "carns": "Carens", "carrens": "Carens",
    "ertiga": "Ertiga", "ertga": "Ertiga", "ertica": "Ertiga",
    "amaze": "Amaze", "amaz": "Amaze", "amazee": "Amaze",
    "civic": "Civic", "civec": "Civic", "civick": "Civic",
    "elantra": "Elantra", "elatra": "Elantra",
    "verna": "Verna", "vena": "Verna", "vernaa": "Verna",
    "i20": "i20", "i 20": "i20",
    "i10": "i10", "i 10": "i10",
    "grand": "Grand", "gand": "Grand",
    
    # Samsung Galaxy series typos (Galaxy is a model/series name)
    "galaxy": "Galaxy", "galxy": "Galaxy", "galalxy": "Galaxy", "galaxxy": "Galaxy",
    "galazy": "Galaxy", "galaxi": "Galaxy", "gallaxy": "Galaxy", "glxy": "Galaxy",
    "galexy": "Galaxy", "galaxsy": "Galaxy", "galxey": "Galaxy", "galaxay": "Galaxy",
    
    # ===== GOOGLE PIXEL MODELS =====
    # Pixel phone typos (map typos to correct spelling)
    "pixle": "Pixel", "pixxel": "Pixel", "pixal": "Pixel", "pxel": "Pixel",
    "piksel": "Pixel", "pixcel": "Pixel", "pixell": "Pixel", "pixeel": "Pixel",
    "pixl": "Pixel", "pixle 6": "Pixel 6", "pixle 7": "Pixel 7", "pixle 8": "Pixel 8",
    "pixle 9": "Pixel 9", "pixxel 6": "Pixel 6", "pixxel 7": "Pixel 7",
    "pixxel 8": "Pixel 8", "pixxel 9": "Pixel 9", "pixle pro": "Pixel Pro",
    "pixxel pro": "Pixel Pro", "pixel pro": "Pixel Pro", "pixel fold": "Pixel Fold",
    "pixle fold": "Pixel Fold", "pixxel fold": "Pixel Fold",
    
    # Note: Common word typos like "amchine", "washin" are handled in 
    # WORD_TYPO_FIXES in generate_corrected_query() - NOT here
}

# Brands that should be fully uppercase
UPPERCASE_BRANDS = {"lg", "tvs", "ifb", "hp", "mi", "jbl", "tcl", "vu"}

def format_brand_name(brand: str) -> str:
    """Format brand name with proper casing."""
    if brand.lower() in UPPERCASE_BRANDS:
        return brand.upper()
    return brand.title()

def generate_corrected_query(original_query: str, query_info: Dict = None, 
                              search_result: Dict = None) -> Optional[str]:
    """
    Generate a corrected query for UI display.
    
    This function ONLY corrects typos in the user's query.
    It does NOT add brand names, categories, or any additional context.
    
    Returns:
        - Corrected query string if typo was fixed
        - None if no typo correction needed (query was already correct)
    
    Examples:
        "samsng" → "Samsung" (typo fix)
        "acktiva" → "Activa" (typo fix)
        "to wheeler" → "Two Wheeler" (typo fix)
        "activa" → None (no correction needed - already correct)
        "samsung" → None (no correction needed)
        "mi" → None (no correction needed)
    """
    if not original_query:
        return None
    
    original_lower = original_query.lower().strip()
    
    # Track what we've corrected (typo fixes only)
    typo_corrections = {}  # word -> corrected_word
    
    # Clean noise: remove special characters but keep spaces
    cleaned = re.sub(r'[$@#!*&^%_\-]+', ' ', original_lower)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # ===== PRE-PROCESSING: Fix common word typos =====
    # These are general word typos (not model names)
    WORD_TYPO_FIXES = {
        # Washing machine word typos
        "amchine": "machine", "amachine": "machine", "machne": "machine",
        "machin": "machine", "macine": "machine", "mchine": "machine", "mahcine": "machine",
        "washin": "washing", "washng": "washing", "wahsing": "washing",
        "washign": "washing", "washnig": "washing", "washig": "washing",
        # Air conditioner typos
        "conditoner": "conditioner", "condtioner": "conditioner", "conditionr": "conditioner",
        # Refrigerator typos  
        "refridgerator": "refrigerator", "refregerator": "refrigerator", "refrigerater": "refrigerator",
        "refrigrator": "refrigerator", "refridgrator": "refrigerator", "refrijerator": "refrigerator",
        "refridger": "refrigerator", "refrgator": "refrigerator", "frigrator": "refrigerator",
        # Television typos
        "telivision": "television", "televison": "television", "televishion": "television",
        "televisn": "television", "telvision": "television", "televisin": "television",
        "televsin": "television", "televisoin": "television", "televison": "television",
        # Galaxy typos
        "galalxy": "galaxy", "galxy": "galaxy", "galaxxy": "galaxy",
        "galazy": "galaxy", "galaxi": "galaxy", "gallaxy": "galaxy",
        "glxy": "galaxy", "galexy": "galaxy", "galxey": "galaxy",
        # Wheeler typos
        "wheler": "wheeler", "wheelers": "wheelers", "weeler": "wheeler",
        "wheleer": "wheeler", "wheelar": "wheeler", "whelar": "wheeler",
        # Royal Enfield typos
        "royel": "royal", "royl": "royal", "royall": "royal", "roayl": "royal",
        "enfild": "enfield", "enfeild": "enfield", "enfied": "enfield", "enfeeld": "enfield",
        "enfld": "enfield", "enflied": "enfield", "enfiled": "enfield",
        # Panasonic typos
        "panasnic": "panasonic", "pansonic": "panasonic", "panasonc": "panasonic",
        # Philips typos
        "philps": "philips", "philips": "philips", "phillps": "philips",
        # Haier typos
        "haieer": "haier", "haior": "haier", "hier": "haier",
        # Voltas typos
        "volts": "voltas", "voltss": "voltas",
        # Godrej typos
        "godrag": "godrej", "godreg": "godrej", "godraj": "godrej",
        # Hero typos
        "hiro": "hero", "herro": "hero",
        # TVS typos
        "tvss": "tvs",
        # Suzuki typos
        "suzki": "suzuki", "suzuky": "suzuki", "suziuki": "suzuki",
        # Kawasaki typos
        "kawaskai": "kawasaki", "kawaski": "kawasaki", "kawsaki": "kawasaki",
        # KTM typos
        "ktmm": "ktm",
        # Cabinet typos
        "cabinate": "cabinet", "cabinat": "cabinet", "cabnet": "cabinet", "cabinent": "cabinet",
        "cabnit": "cabinet", "cabinett": "cabinet", "cabint": "cabinet",
        "cabiner": "cabinet", "cabiner": "cabinet", "cabimet": "cabinet", "cabniet": "cabinet",
        "cabiniet": "cabinet", "cabinaet": "cabinet", "cabnate": "cabinet", "cabnat": "cabinet",
        # Heater typos (NOT including "heater" as it's correct)
        "heatar": "heater", "heter": "heater", "heetar": "heater",
        "heatter": "heater", "hateer": "heater", "heeter": "heater",
        # Geyser typos
        "geysar": "geyser", "geezer": "geyser", "geysir": "geyser", "geizar": "geyser",
        # Table typos
        "tabel": "table", "teble": "table", "tablle": "table",
        # Dining typos
        "dinning": "dining", "dininng": "dining", "dinin": "dining",
        "dinner": "dining",  # "dinner table" → "dining table"
        "diner": "dining",   # "diner table" → "dining table"
        # Furniture typos
        "furnitur": "furniture", "furnture": "furniture", "furnituer": "furniture",
        # Chair typos
        "chiar": "chair", "chaire": "chair", "cher": "chair",
        # Sofa typos
        "sofaa": "sofa", "soofa": "sofa", "soffa": "sofa",
        # Bed typos
        "bedd": "bed", "bead": "bed",
        # Mattress typos
        "matress": "mattress", "mattrass": "mattress", "mattres": "mattress", "matras": "mattress",
        # Laptop typos
        "labtop": "laptop", "laptoop": "laptop", "leptop": "laptop", "laptp": "laptop",
        # Camera typos
        "camra": "camera", "cammera": "camera", "camara": "camera",
        # Water typos
        "watar": "water", "watr": "water", "wateer": "water",
        # Purifier typos
        "purifiar": "purifier", "purifyer": "purifier", "purifir": "purifier",
        # Cooler typos
        "coolar": "cooler", "coolr": "cooler", "cooller": "cooler",
        # Printer typos
        "printr": "printer", "printar": "printer", "prnter": "printer", "printeer": "printer",
        # Microwave typos
        "microwav": "microwave", "microwve": "microwave", "micrwave": "microwave", "microwavee": "microwave",
        # Smartphone typos
        "smartfone": "smartphone", "smartphon": "smartphone", "smarphone": "smartphone",
        # Mobile typos - keyboard adjacent keys and common misspellings
        "moblie": "mobile", "mobil": "mobile", "moble": "mobile",
        "mobail": "mobile", "mobaile": "mobile", "mobel": "mobile",
        "mobal": "mobile", "moblile": "mobile", "mboile": "mobile",
        "mobiile": "mobile", "mobiil": "mobile", "mobille": "mobile",
        "mobole": "mobile", "mobiel": "mobile", "moibel": "mobile",
        "moible": "mobile", "mobyle": "mobile", "mobale": "mobile",
        # Phone typos - keyboard adjacent keys (r near e, o near i, etc.)
        "phonr": "phone", "phoen": "phone", "pohne": "phone",
        "phin": "phone", "phine": "phone", "pone": "phone",
        "phonne": "phone", "phonee": "phone", "phoner": "phone",
        "phons": "phone", "phonse": "phone", "fonr": "phone",
        "phn": "phone", "phne": "phone", "phon": "phone",
        "fone": "phone", "fon": "phone",
        # Earphone/Earbud/Headphone typos - CRITICAL for corrected_query display
        "earphn": "earphone", "earphon": "earphone", "earfon": "earphone",
        "earfone": "earphone", "earpone": "earphone", "erphn": "earphone",
        "erpone": "earphone", "erphone": "earphone", "erfone": "earphone",
        "erfon": "earphone", "earphne": "earphone", "eaphone": "earphone",
        "earpho": "earphone",
        # raphon typos (missing 'e' at start)
        "raphon": "earphone", "raphne": "earphone", "raphone": "earphone", "raphn": "earphone",
        # airphon typos ('air' instead of 'ear')
        "airphon": "earphone", "airphone": "earphone", "airphn": "earphone",
        "airfon": "earphone", "airfone": "earphone",
        # Earbud typos
        "erbud": "earbud", "airbud": "earbud", "earbds": "earbuds", "earbd": "earbud",
        "erbuds": "earbuds", "airbuds": "earbuds",
        # Headphone typos
        "headphon": "headphone", "headfon": "headphone", "headfone": "headphone",
        "hedphone": "headphone", "headphn": "headphone", "haedphone": "headphone",
        "headphne": "headphone",
        # Neckband typos
        "neckbnd": "neckband", "nekband": "neckband", "neckbad": "neckband",
        "nckband": "neckband", "necband": "neckband",
    }
    
    words = cleaned.split()
    fixed_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower in WORD_TYPO_FIXES:
            fixed_words.append(WORD_TYPO_FIXES[word_lower])
            typo_corrections[word_lower] = WORD_TYPO_FIXES[word_lower]
        else:
            fixed_words.append(word)
    cleaned = " ".join(fixed_words)
    
    # ===== PHRASE-LEVEL CORRECTIONS =====
    # Fix common phrase typos (word substitutions)
    PHRASE_CORRECTIONS = {
        "to wheeler": "two wheeler",
        "to wheelers": "two wheelers", 
        "tow wheeler": "two wheeler",
        "too wheeler": "two wheeler",
        "for wheeler": "four wheeler",
        "fore wheeler": "four wheeler",
        "4 wheler": "4 wheeler",
        "2 wheler": "2 wheeler",
        # Mac Mini reversed order corrections
        "mini mac": "mac mini",
        "minimac": "mac mini",
        "mini-mac": "mac mini",
        "mini mack": "mac mini",
        "minni mac": "mac mini",
        "mni mac": "mac mini",
        "mini mc": "mac mini",
        "minii mac": "mac mini",
    }
    
    cleaned_lower = cleaned.lower()
    for wrong_phrase, correct_phrase in PHRASE_CORRECTIONS.items():
        if wrong_phrase in cleaned_lower:
            cleaned = re.sub(re.escape(wrong_phrase), correct_phrase, cleaned, flags=re.IGNORECASE)
            typo_corrections[wrong_phrase] = correct_phrase
    
    # ===== SIMPLE TYPO-ONLY CORRECTION =====
    # Only fix typos in words, don't add any additional context
    
    words = cleaned.split() if cleaned else []
    result_words = []
    
    for word in words:
        word_lower = word.lower()
        
        # Check brand typos
        if word_lower in BRAND_TYPO_CORRECTIONS:
            corrected = BRAND_TYPO_CORRECTIONS[word_lower]
            # Only count as typo if the word changed (not just casing)
            if word_lower != corrected.lower():
                typo_corrections[word_lower] = corrected
                result_words.append(corrected)
            else:
                # Already correct brand, just fix casing
                result_words.append(corrected)
            continue
        
        # Check model typos
        if word_lower in MODEL_TYPO_CORRECTIONS:
            corrected = MODEL_TYPO_CORRECTIONS[word_lower]
            # Only count as typo if the word changed
            if word_lower != corrected.lower():
                typo_corrections[word_lower] = corrected
                result_words.append(corrected)
            else:
                # Already correct model, just fix casing
                result_words.append(corrected)
            continue
        
        # Check category canonical names (for short forms like "tv" → "Television")
        if word_lower in CATEGORY_CANONICAL_NAMES:
            canonical = CATEGORY_CANONICAL_NAMES[word_lower]
            if canonical:
                # Short form expansion counts as correction
                if word_lower != canonical.lower():
                    typo_corrections[word_lower] = canonical
                result_words.append(canonical)
            else:
                result_words.append(word.title())
            continue
        
        # Keep word as-is with title casing
        if word.isdigit():
            result_words.append(word)
        elif re.match(r'^\d+[a-zA-Z]+$', word):  # e.g., "6g", "55inch"
            result_words.append(word.upper())
        else:
            result_words.append(word.title())
    
    # Build final corrected query
    if result_words:
        corrected = " ".join(result_words)
        
        # Only return if we actually made typo corrections
        if typo_corrections:
            return corrected
        
        # No typo corrections - return None
        return None
    
    return None


# =================== DISPLAY QUERY ENHANCER ===================
def enhance_display_query(original_query: str, query_info: Dict = None) -> str:
    """
    Enhance query for user-friendly display in suggested_search_keyword.
    This function creates a clean, corrected version for UI display ONLY.
    It does NOT affect search functionality.
    
    Handles:
    1. Short forms: tv → Television, ac → Air Conditioner, phn → Phone
    2. Typos: samsng → Samsung, vevo → Vivo
    3. Compound words: washingmachine → Washing Machine
    4. Brand+Category: samsung tv → Samsung Television
    5. Canonical names: Use proper display names
    """
    if not original_query:
        return original_query or ""
    
    import re
    query = original_query.lower().strip()
    
    # =================== STEP 1: FIX TYPOS ===================
    # Brand typos - comprehensive list for all categories
    brand_fixes = [
        # Samsung
        (r'\bsamung\b', 'samsung'), (r'\bsamsun\b', 'samsung'), (r'\bsamsang\b', 'samsung'),
        (r'\bsamsumg\b', 'samsung'), (r'\bsumsang\b', 'samsung'), (r'\bsumsung\b', 'samsung'),
        (r'\bsamsng\b', 'samsung'), (r'\bsamusng\b', 'samsung'), (r'\bsmasung\b', 'samsung'),
        (r'\bsamsyng\b', 'samsung'), (r'\bsamsungg\b', 'samsung'),
        # Vivo
        (r'\bvevo\b', 'vivo'), (r'\bviovo\b', 'vivo'), (r'\bvivi\b', 'vivo'), (r'\bvvo\b', 'vivo'),
        (r'\bvoivo\b', 'vivo'), (r'\bvivio\b', 'vivo'), (r'\bvovo\b', 'vivo'), (r'\bvivoo\b', 'vivo'),
        (r'\bvivp\b', 'vivo'),
        # Oppo
        (r'\bopp0\b', 'oppo'), (r'\bopo\b', 'oppo'), (r'\bopoo\b', 'oppo'), (r'\b0ppo\b', 'oppo'),
        (r'\boppp\b', 'oppo'), (r'\boppoo\b', 'oppo'),
        # Realme
        (r'\brelme\b', 'realme'), (r'\brealmee\b', 'realme'), (r'\brelame\b', 'realme'), 
        (r'\breamle\b', 'realme'), (r'\breame\b', 'realme'), (r'\brealmy\b', 'realme'),
        (r'\brealmei\b', 'realme'), (r'\brealmi\b', 'realme'),
        # Redmi
        (r'\bredmii\b', 'redmi'), (r'\bradmi\b', 'redmi'), (r'\bridmi\b', 'redmi'),
        (r'\bredmmi\b', 'redmi'),
        # Poco
        (r'\bpocco\b', 'poco'), (r'\bpoko\b', 'poco'), (r'\bpocoo\b', 'poco'),
        # Motorola
        (r'\bmotarola\b', 'motorola'), (r'\bmotorla\b', 'motorola'), (r'\bmotorolla\b', 'motorola'),
        # OnePlus
        (r'\b1plus\b', 'oneplus'), (r'\bone\+\b', 'oneplus'), (r'\bonplus\b', 'oneplus'),
        (r'\bonepls\b', 'oneplus'),
        # Nokia
        (r'\bnokiya\b', 'nokia'), (r'\bnoika\b', 'nokia'),
        # Apple
        (r'\bappel\b', 'apple'), (r'\baple\b', 'apple'),
        # LG
        (r'\belg\b', 'lg'),
        # Haier
        (r'\bhaior\b', 'haier'), (r'\bhaire\b', 'haier'),
        # Whirlpool
        (r'\bwhirlpol\b', 'whirlpool'), (r'\bwhirpool\b', 'whirlpool'), (r'\bwhirpol\b', 'whirlpool'),
        # Godrej
        (r'\bgodraj\b', 'godrej'), (r'\bgodeg\b', 'godrej'),
        # Lenovo
        (r'\blenova\b', 'lenovo'), (r'\blenoov\b', 'lenovo'),
        # HP
        (r'\bhewlett\s*packard\b', 'hp'),
        # Dell
        (r'\bdel\b(?!\s+tv)', 'dell'),  # Avoid matching "del" in other contexts
        # Panasonic
        (r'\bpanasnic\b', 'panasonic'), (r'\bpanasoinc\b', 'panasonic'),
        # Philips
        (r'\bphilps\b', 'philips'), (r'\bphillips\b', 'philips'), (r'\bphilipes\b', 'philips'),
        # Sony
        (r'\bsoni\b', 'sony'), (r'\bsonny\b', 'sony'),
        # Boat
        (r'\bboatt\b', 'boat'),
        # Voltas
        (r'\bvoltaas\b', 'voltas'), (r'\bvoltes\b', 'voltas'),
        # Daikin
        (r'\bdakin\b', 'daikin'), (r'\bdaikn\b', 'daikin'),
        # Lloyd
        (r'\blloyd\b', 'lloyd'), (r'\bloyd\b', 'lloyd'),
        # Blue Star
        (r'\bbluestar\b', 'blue star'),
        # IFB
        (r'\bifbb\b', 'ifb'),
        # Bosch
        (r'\bbosh\b', 'bosch'), (r'\bbosche\b', 'bosch'),
        # Kent
        (r'\bkent\b', 'kent'),
        # Livpure
        (r'\blivpur\b', 'livpure'), (r'\blivepure\b', 'livpure'),
        # Racold
        (r'\bracod\b', 'racold'), (r'\bracld\b', 'racold'),
        # AO Smith
        (r'\bao\s*smith\b', 'ao smith'), (r'\baosmith\b', 'ao smith'),
        # Havells
        (r'\bhavels\b', 'havells'), (r'\bhavells\b', 'havells'),
        
        # =================== VEHICLE BRANDS ===================
        # Honda - very important, includes typos
        (r'\bhoda\b', 'honda'), (r'\bhondaa\b', 'honda'), (r'\bhonada\b', 'honda'),
        (r'\bhondha\b', 'honda'), (r'\bhona\b', 'honda'),
        # Hero
        (r'\bheroo\b', 'hero'), (r'\bheero\b', 'hero'), (r'\bhero\b', 'hero'),
        # TVS
        (r'\btves\b', 'tvs'), (r'\btvss\b', 'tvs'), (r'\btvs\b', 'tvs'),
        # Yamaha
        (r'\byamha\b', 'yamaha'), (r'\byahmaha\b', 'yamaha'), (r'\byamhaa\b', 'yamaha'),
        # Suzuki
        (r'\bsuzki\b', 'suzuki'), (r'\bsuzuuki\b', 'suzuki'),
        # Royal Enfield
        (r'\broyalenfield\b', 'royal enfield'), (r'\broyal\s*enfield\b', 'royal enfield'),
        (r'\broyal\s*enfild\b', 'royal enfield'), (r'\broyalenfiled\b', 'royal enfield'),
        (r'\broyel\s*enfield\b', 'royal enfield'), (r'\broyal\s*enfiled\b', 'royal enfield'),
        # Bajaj
        (r'\bbajaj\b', 'bajaj'), (r'\bbajjaj\b', 'bajaj'), (r'\bbajaaj\b', 'bajaj'),
        (r'\bbajja\b', 'bajaj'), (r'\bbajajj\b', 'bajaj'),
        
        # =================== CAR BRANDS ===================
        # Hyundai
        (r'\bhundai\b', 'hyundai'), (r'\bhyundia\b', 'hyundai'), (r'\bhyundaii\b', 'hyundai'),
        (r'\bhyudai\b', 'hyundai'), (r'\bhyunadi\b', 'hyundai'),
        # Maruti
        (r'\bmaruthi\b', 'maruti'), (r'\bmaruti\s*suzuki\b', 'maruti suzuki'),
        (r'\bmarutii\b', 'maruti'), (r'\bmarutti\b', 'maruti'), (r'\bmarut\b', 'maruti'),
        # Hitachi
        (r'\bhitaachi\b', 'hitachi'), (r'\bhitachi\b', 'hitachi'), (r'\bhitaci\b', 'hitachi'),
        # iPhone typos
        (r'\biphon\b', 'iphone'), (r'\biphn\b', 'iphone'), (r'\biphone\b', 'iphone'),
        (r'\bifone\b', 'iphone'), (r'\biph\b', 'iphone'),
        # Hero model typos
        (r'\bsplendr\b', 'splendor'), (r'\bsplendour\b', 'splendor'), (r'\bsplendr\b', 'splendor'),
        # Tata
        (r'\btatta\b', 'tata'), (r'\btata\b', 'tata'),
        # Mahindra
        (r'\bmahendra\b', 'mahindra'), (r'\bmahindara\b', 'mahindra'),
        # Kia
        (r'\bkia\b', 'kia'), (r'\bkiya\b', 'kia'),
        # Toyota
        (r'\btoyata\b', 'toyota'), (r'\btoyota\b', 'toyota'),
        
        # =================== CAR MODELS (for display) ===================
        # Honda City - common car model
        (r'\bcirty\b', 'city'), (r'\bcity\b', 'city'),
        # Hyundai Creta
        (r'\bcreta\b', 'creta'), (r'\bcreat\b', 'creta'),
        # Tata Nexon
        (r'\bnexon\b', 'nexon'), (r'\bnexn\b', 'nexon'),
        # Maruti Swift
        (r'\bswift\b', 'swift'), (r'\bswft\b', 'swift'),
        
        # =================== TRACTOR BRANDS ===================
        (r'\bswaraj\b', 'swaraj'), (r'\bswaraaj\b', 'swaraj'),
        (r'\bjohn\s*deere\b', 'john deere'), (r'\bjohndeere\b', 'john deere'),
        (r'\bjohn\s*deer\b', 'john deere'), (r'\bjhon\s*deere\b', 'john deere'),
        
        # =================== CATEGORY TYPOS ===================
        # Tractor
        (r'\btactor\b', 'tractor'), (r'\btractr\b', 'tractor'), (r'\btracktor\b', 'tractor'),
        # Furniture
        (r'\bsopha\b', 'sofa'), (r'\bsoffa\b', 'sofa'),
        (r'\bmatress\b', 'mattress'), (r'\bmattres\b', 'mattress'), (r'\bmatras\b', 'mattress'),
        (r'\bfarnichar\b', 'furniture'), (r'\bfurnichar\b', 'furniture'), (r'\bfurnitur\b', 'furniture'),
        (r'\bfarniture\b', 'furniture'), (r'\bfernicher\b', 'furniture'),
        # Washing machine
        (r'\bwashin\b', 'washing'), (r'\bwasing\b', 'washing'),
        # Kitchen appliances
        (r'\bmiksi\b', 'mixer'), (r'\bmikser\b', 'mixer'), (r'\bmixar\b', 'mixer'),
        (r'\bgijer\b', 'geyser'), (r'\bgijar\b', 'geyser'), (r'\bgizer\b', 'geyser'), (r'\bgeysar\b', 'geyser'),
        (r'\bjusar\b', 'juicer'), (r'\bjuisar\b', 'juicer'), (r'\bjucer\b', 'juicer'),
        # Microwave
        (r'\bmicrowaveoven\b', 'microwave oven'),
    ]
    
    for pattern, replacement in brand_fixes:
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    # =================== STEP 2: EXPAND SHORT FORMS ===================
    # Map short forms to full canonical display names
    short_form_map = {
        # Phone/Mobile variations
        'phn': 'phone', 'phne': 'phone', 'phon': 'phone', 'fone': 'phone', 'fon': 'phone',
        # Phone typos - keyboard adjacent keys (r near e, o near i, etc.)
        'phonr': 'phone', 'phoen': 'phone', 'pohne': 'phone', 'phin': 'phone', 'phine': 'phone',
        'pone': 'phone', 'phonne': 'phone', 'phonee': 'phone', 'phoner': 'phone',
        'phons': 'phone', 'phonse': 'phone', 'fonr': 'phone',
        'mbl': 'mobile', 'mobl': 'mobile', 'mobil': 'mobile', 'mob': 'mobile', 'mobiles': 'mobile',
        # Mobile typos - keyboard adjacent keys and common misspellings
        'mobail': 'mobile', 'mobaile': 'mobile', 'mobel': 'mobile', 'mobal': 'mobile',
        'moblie': 'mobile', 'moblile': 'mobile', 'moble': 'mobile', 'mboile': 'mobile',
        'mobiile': 'mobile', 'mobiil': 'mobile', 'mobille': 'mobile', 'mobole': 'mobile',
        'mobiel': 'mobile', 'moibel': 'mobile', 'moible': 'mobile', 'mobyle': 'mobile', 'mobale': 'mobile',
        'smrtphone': 'smartphone', 'smartfone': 'smartphone', 'smartphn': 'smartphone',
        # TV variations - Note: 'tvs' NOT mapped here because TVS is a vehicle brand
        'tv': 'television', 'telivision': 'television', 'telvision': 'television',
        'televison': 'television', 'televisn': 'television',
        'led': 'led television', 'smart tv': 'smart television', 'led tv': 'led television',
        'oled tv': 'oled television', 'qled tv': 'qled television',
        # AC variations
        'ac': 'air conditioner', 'acs': 'air conditioner', 'a.c': 'air conditioner', 'a.c.': 'air conditioner',
        'airconditioner': 'air conditioner', 'aircon': 'air conditioner', 'split ac': 'split air conditioner',
        'window ac': 'window air conditioner', 'inverter ac': 'inverter air conditioner',
        # Refrigerator variations
        'fridge': 'refrigerator', 'frig': 'refrigerator', 'frdge': 'refrigerator', 'fridg': 'refrigerator',
        'refridgerator': 'refrigerator', 'refregerator': 'refrigerator', 'refrgrator': 'refrigerator',
        'double door fridge': 'double door refrigerator', 'single door fridge': 'single door refrigerator',
        '4 door fridge': 'four door refrigerator', '4 door refrigerator': 'four door refrigerator',
        'four door fridge': 'four door refrigerator', '3 door fridge': 'triple door refrigerator',
        '3 door refrigerator': 'triple door refrigerator', 'three door fridge': 'triple door refrigerator',
        # Washing machine variations
        'wm': 'washing machine', 'washer': 'washing machine', 'washingmachine': 'washing machine',
        'washing machin': 'washing machine',
        # Note: 'front load' and 'top load' removed to prevent duplication
        # Laptop variations
        'lptop': 'laptop', 'laptp': 'laptop', 'lapto': 'laptop',
        'notbook': 'notebook', 'notebk': 'notebook',
        # Water heater/Geyser variations
        'geyser': 'water heater', 'geysers': 'water heater', 'geysar': 'water heater',
        'gayser': 'water heater', 'giser': 'water heater', 'gizer': 'water heater',
        'gijar': 'water heater', 'gisar': 'water heater', 'gyser': 'water heater',
        'waterheater': 'water heater',
        # Water purifier variations
        'waterpurifier': 'water purifier', 'ro': 'water purifier', 'water purifir': 'water purifier',
        # Air cooler variations
        'aircooler': 'air cooler', 'cooler': 'air cooler', 'desert cooler': 'desert air cooler',
        # Air purifier variations  
        'airpurifier': 'air purifier', 'air purifir': 'air purifier',
        # Microwave variations (avoid mapping 'microwave' alone to prevent duplication)
        'ovn': 'oven', 'microwav': 'microwave',
        # Note: 'microwave' and 'oven' not expanded to prevent duplication
        # Earphone/Headphone variations - comprehensive typo coverage
        'earphone': 'earphones', 'earphn': 'earphones', 'earfon': 'earphones',
        'earphon': 'earphones', 'earfone': 'earphones', 'earpone': 'earphones',
        'erphn': 'earphones', 'erpone': 'earphones', 'erphone': 'earphones',
        'erfone': 'earphones', 'erfon': 'earphones', 'earphne': 'earphones',
        'eaphone': 'earphones', 'earpho': 'earphones',
        # raphon typos (missing 'e' at start)
        'raphon': 'earphones', 'raphne': 'earphones', 'raphone': 'earphones', 'raphn': 'earphones',
        # airphon typos ('air' instead of 'ear')
        'airphon': 'earphones', 'airphone': 'earphones', 'airphn': 'earphones',
        'airfon': 'earphones', 'airfone': 'earphones',
        'earbud': 'earbuds', 'tws': 'true wireless earbuds',
        # Earbud typos
        'erbud': 'earbuds', 'airbud': 'earbuds', 'earbds': 'earbuds', 'earbd': 'earbuds',
        'headphone': 'headphones', 'headfon': 'headphones', 'headset': 'headphones',
        # Headphone typos
        'headphon': 'headphones', 'headfone': 'headphones', 'hedphone': 'headphones',
        'headphn': 'headphones', 'haedphone': 'headphones', 'headphne': 'headphones',
        'neckband': 'neckband earphones',
        # Neckband typos
        'neckbnd': 'neckband earphones', 'nekband': 'neckband earphones', 'nckband': 'neckband earphones',
        # Speaker variations
        'speaker': 'speaker', 'spkr': 'speaker', 'soundbar': 'soundbar',
        # Watch variations
        'smartwatch': 'smart watch', 'smrt watch': 'smart watch', 'smart wach': 'smart watch',
        'fitness band': 'fitness band',
        # Printer variations
        'printr': 'printer', 'prnter': 'printer',
        # Vacuum cleaner variations
        'vacuum': 'vacuum cleaner', 'vacum cleaner': 'vacuum cleaner',
        # Camera variations
        'cam': 'camera', 'dslr': 'dslr camera', 'webcam': 'webcam',
        # Cycle variations
        'bicycle': 'cycle', 'bike cycle': 'cycle',
        # Two wheeler variations - Keep original words for display, don't expand to 'two wheeler'
        # The search handles these, display should show user's intent
        # Note: 'bike', 'scooter', etc. are NOT expanded to preserve display intent
        # Car variations
        'car': 'car', 'cars': 'car', '4 wheeler': 'car', 'four wheeler': 'car',
        # Mixer variations (avoid mapping 'mixer' alone to prevent 'mixer grinder' → 'mixer grinder grinder')
        'mixie': 'mixer grinder', 'mixi': 'mixer grinder',
        'blender': 'blender',
        # Note: 'mixer' alone not mapped to prevent duplication
        # Inverter variations
        'ups': 'inverter', 'solar panel': 'solar inverter',
        # Tablet variations
        'tab': 'tablet', 'tabs': 'tablet', 'ipad': 'tablet',
    }
    
    # Split query into words for processing
    words = query.split()
    enhanced_words = []
    
    i = 0
    while i < len(words):
        # Check for multi-word matches first (led tv, smart tv, etc.)
        if i < len(words) - 1:
            two_word = f"{words[i]} {words[i+1]}"
            if two_word in short_form_map:
                enhanced_words.append(short_form_map[two_word])
                i += 2
                continue
        
        # Single word match
        word = words[i]
        if word in short_form_map:
            enhanced_words.append(short_form_map[word])
        else:
            enhanced_words.append(word)
        i += 1
    
    query = ' '.join(enhanced_words)
    
    # =================== STEP 3: MODEL NORMALIZATION ===================
    # Vivo model: vivox200 → vivo x200, vivox200pro → vivo x200 pro
    # Added "elite" suffix for V70 Elite models
    vivo_pattern = r'\b(vivo)\s*([xyvt])\s*(\d+[a-z]?)(?:\s+(pro|plus|lite|ultra|fe|elite))?\b'
    def normalize_vivo(m):
        brand, series, num, suffix = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        return f"{brand} {series}{num} {suffix}".strip()
    query = re.sub(vivo_pattern, normalize_vivo, query, flags=re.IGNORECASE)
    
    # Oppo model: oppof19 → oppo f19
    oppo_pattern = r'\b(oppo)\s*([afkr])\s*(\d+)(?:\s+(pro|plus|lite|s|x|k))?\b'
    def normalize_oppo(m):
        brand, series, num, suffix = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        return f"{brand} {series}{num} {suffix}".strip()
    query = re.sub(oppo_pattern, normalize_oppo, query, flags=re.IGNORECASE)
    
    # Realme model: realme12 → realme 12, BUT keep series+number together (p3 → p3, c73 → c73)
    # IMPORTANT: [cp]\d+ must come BEFORE [cp] for correct matching order
    realme_pattern = r'\b(realme)\s*([cp]\d+[a-z]?|\d+|gt|narzo|[cp])(?:\s+(pro|plus|neo|master|ultra|x|5g))?\b'
    def normalize_realme(m):
        brand, series, suffix = m.group(1), m.group(2), m.group(3) or ""
        parts = [brand, series]
        if suffix: parts.append(suffix)
        return ' '.join(parts)
    query = re.sub(realme_pattern, normalize_realme, query, flags=re.IGNORECASE)
    
    # Samsung model: samsungs24 → samsung s24, samsung galaxy s24
    samsung_pattern = r'\b(samsung)\s*(galaxy)?\s*([asmzf])\s*(\d+)(?:\s+(fe|ultra|plus|lite|s))?\b'
    def normalize_samsung(m):
        brand, galaxy, series, num, suffix = m.group(1), m.group(2) or "", m.group(3), m.group(4), m.group(5) or ""
        parts = [brand]
        if galaxy: parts.append(galaxy)
        parts.append(f"{series}{num}")
        if suffix: parts.append(suffix)
        return ' '.join(parts)
    query = re.sub(samsung_pattern, normalize_samsung, query, flags=re.IGNORECASE)
    
    # Redmi model: redminote13 → redmi note 13
    redmi_pattern = r'\b(redmi)\s*(note)?\s*(\d+)(?:\s+(pro|plus|a|c|s|i))?\b'
    def normalize_redmi(m):
        brand, note, num, suffix = m.group(1), m.group(2) or "", m.group(3), m.group(4) or ""
        parts = [brand]
        if note: parts.append(note)
        parts.append(num)
        if suffix: parts.append(suffix)
        return ' '.join(parts)
    query = re.sub(redmi_pattern, normalize_redmi, query, flags=re.IGNORECASE)
    
    # iPhone model: iphone15 → iphone 15
    iphone_pattern = r'\b(iphone)\s*(\d+)(?:\s+(pro|plus|max|mini|se))?(?:\s+(max))?\b'
    def normalize_iphone(m):
        brand, num, suffix1, suffix2 = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
        parts = [brand, num]
        if suffix1: parts.append(suffix1)
        if suffix2: parts.append(suffix2)
        return ' '.join(parts)
    query = re.sub(iphone_pattern, normalize_iphone, query, flags=re.IGNORECASE)
    
    # =================== STEP 4: CAPITALIZE PROPERLY ===================
    # Title case but keep known brand capitalization
    brand_caps = {
        # Mobile brands
        'samsung': 'Samsung', 'vivo': 'Vivo', 'oppo': 'Oppo', 'realme': 'Realme',
        'redmi': 'Redmi', 'poco': 'Poco', 'oneplus': 'OnePlus', 'motorola': 'Motorola',
        'nokia': 'Nokia', 'apple': 'Apple', 'iphone': 'iPhone', 'lg': 'LG', 'hp': 'HP',
        'dell': 'Dell', 'lenovo': 'Lenovo', 'asus': 'Asus', 'acer': 'Acer',
        'xiaomi': 'Xiaomi', 'mi': 'Mi', 'iqoo': 'iQOO', 'nothing': 'Nothing',
        'google': 'Google', 'infinix': 'Infinix', 'tecno': 'Tecno', 'lava': 'Lava',
        # Appliance brands
        'sony': 'Sony', 'panasonic': 'Panasonic', 'philips': 'Philips',
        'haier': 'Haier', 'whirlpool': 'Whirlpool', 'godrej': 'Godrej',
        'voltas': 'Voltas', 'lloyd': 'Lloyd', 'daikin': 'Daikin', 'hitachi': 'Hitachi',
        'ifb': 'IFB', 'bosch': 'Bosch', 'kent': 'Kent', 'livpure': 'Livpure',
        'carrier': 'Carrier', 'midea': 'Midea', 'onida': 'Onida', 'croma': 'Croma',
        'blue star': 'Blue Star', 'o general': 'O General',
        # Audio brands
        'jbl': 'JBL', 'boat': 'boAt', 'bose': 'Bose', 'marshall': 'Marshall',
        'sennheiser': 'Sennheiser', 'harman kardon': 'Harman Kardon', 'mivi': 'Mivi',
        'zebronics': 'Zebronics',
        # Camera brands
        'canon': 'Canon', 'nikon': 'Nikon', 'gopro': 'GoPro', 'dji': 'DJI',
        'fujifilm': 'Fujifilm', 'insta360': 'Insta360',
        # Two-wheeler brands
        'honda': 'Honda', 'hero': 'Hero', 'tvs': 'TVS', 'bajaj': 'Bajaj',
        'yamaha': 'Yamaha', 'suzuki': 'Suzuki', 'royal enfield': 'Royal Enfield',
        'ktm': 'KTM', 'kawasaki': 'Kawasaki', 'bmw': 'BMW',
        # Car brands
        'hyundai': 'Hyundai', 'maruti': 'Maruti', 'maruti suzuki': 'Maruti Suzuki',
        'tata': 'Tata', 'mahindra': 'Mahindra', 'kia': 'Kia', 'toyota': 'Toyota',
        'ford': 'Ford', 'volkswagen': 'Volkswagen', 'skoda': 'Skoda', 'mg': 'MG',
        # Tractor brands
        'swaraj': 'Swaraj', 'john deere': 'John Deere', 'sonalika': 'Sonalika',
        'eicher': 'Eicher', 'kubota': 'Kubota',
        # Water heater brands
        'ao smith': 'AO Smith', 'racold': 'Racold', 'havells': 'Havells',
        'crompton': 'Crompton', 'bajaj electricals': 'Bajaj Electricals',
        # Furniture brands
        'nilkamal': 'Nilkamal', 'godrej interio': 'Godrej Interio', 
        'urban ladder': 'Urban Ladder', 'sleepwell': 'Sleepwell',
        'kurlon': 'Kurlon', 'duroflex': 'Duroflex',
        # Kitchen brands
        'prestige': 'Prestige', 'pigeon': 'Pigeon', 'butterfly': 'Butterfly',
        'preethi': 'Preethi', 'maharaja': 'Maharaja', 'sujata': 'Sujata',
        'usha': 'Usha', 'orient': 'Orient', 'atomberg': 'Atomberg',
        # Abbreviations
        'oled': 'OLED', 'qled': 'QLED', 'led': 'LED', 'uhd': 'UHD', 'hd': 'HD',
        'dslr': 'DSLR', 'tws': 'TWS', 'ro': 'RO', 'ups': 'UPS', 'lcd': 'LCD',
        # Car models for proper casing
        'city': 'City', 'creta': 'Creta', 'nexon': 'Nexon', 'swift': 'Swift',
        'i20': 'i20', 'venue': 'Venue', 'seltos': 'Seltos', 'sonet': 'Sonet',
        # Two-wheeler models
        'activa': 'Activa', 'splendor': 'Splendor', 'pulsar': 'Pulsar',
        'jupiter': 'Jupiter', 'apache': 'Apache', 'bullet': 'Bullet',
        'classic': 'Classic', 'dio': 'Dio', 'shine': 'Shine',
    }
    
    # Category proper names
    category_caps = {
        'television': 'Television', 'air conditioner': 'Air Conditioner',
        'refrigerator': 'Refrigerator', 'washing machine': 'Washing Machine',
        'water heater': 'Water Heater', 'water purifier': 'Water Purifier',
        'air cooler': 'Air Cooler', 'air purifier': 'Air Purifier',
        'microwave oven': 'Microwave Oven', 'vacuum cleaner': 'Vacuum Cleaner',
        'mixer grinder': 'Mixer Grinder', 'smart watch': 'Smart Watch',
        'two wheeler': 'Two Wheeler', 'laptop': 'Laptop', 'tablet': 'Tablet',
        'smartphone': 'Smartphone', 'mobile': 'Mobile', 'phone': 'Phone',
        'earphones': 'Earphones', 'headphones': 'Headphones', 'earbuds': 'Earbuds',
        'neckband earphones': 'Neckband Earphones', 'soundbar': 'Soundbar',
        'speaker': 'Speaker', 'printer': 'Printer', 'camera': 'Camera',
        'cycle': 'Cycle', 'car': 'Car', 'inverter': 'Inverter', 'tractor': 'Tractor',
        'fitness band': 'Fitness Band', 'blender': 'Blender',
        'sofa': 'Sofa', 'bed': 'Bed', 'mattress': 'Mattress', 'wardrobe': 'Wardrobe',
        'dining table': 'Dining Table', 'office chair': 'Office Chair',
    }
    
    # Apply capitalization
    words = query.split()
    result_words = []
    
    i = 0
    while i < len(words):
        # Check multi-word brands/categories first
        found_multi = False
        for length in [3, 2]:  # Check 3-word then 2-word phrases
            if i + length <= len(words):
                phrase = ' '.join(words[i:i+length]).lower()
                if phrase in brand_caps:
                    result_words.append(brand_caps[phrase])
                    i += length
                    found_multi = True
                    break
                elif phrase in category_caps:
                    result_words.append(category_caps[phrase])
                    i += length
                    found_multi = True
                    break
        
        if found_multi:
            continue
        
        # Single word
        word = words[i].lower()
        if word in brand_caps:
            result_words.append(brand_caps[word])
        elif word in category_caps:
            result_words.append(category_caps[word])
        else:
            # Title case for unknown words
            result_words.append(words[i].capitalize())
        i += 1
    
    return ' '.join(result_words)


# =================== FUZZY FALLBACK SYSTEM ===================
# Comprehensive character-level edit distance matching for typo correction
# This acts as a universal fallback for any unrecognized query

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_similarity(s1: str, s2: str) -> float:
    """Calculate normalized similarity (0 to 1) based on edit distance"""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = levenshtein_distance(s1.lower(), s2.lower())
    return 1 - (distance / max_len)


# Master dictionary of all known terms for fuzzy matching
FUZZY_MATCH_DICTIONARY = {
    # =================== CATEGORY KEYWORDS ===================
    "categories": {
        # Vehicle categories
        "car": ["car", "cars", "sedan", "suv", "hatchback", "vehicle", "automobile", "four wheeler", "4 wheeler", "fourwheeler", "4wheeler"],
        "two wheeler": ["bike", "bikes", "motorcycle", "scooter", "scooty", "two wheeler", "2 wheeler", "twowheeler", "2wheeler", "moped"],
        # Electronics
        "smartphone": ["mobile", "mobiles", "phone", "phones", "smartphone", "smartphones", "cell phone", "cellphone", "handset"],
        "television": ["tv", "television", "televisions", "led tv", "smart tv", "oled tv", "qled tv", "telly"],
        "laptops": ["laptop", "laptops", "notebook", "notebooks", "ultrabook"],
        "tablets": ["tablet", "tablets", "tab", "ipad"],
        # Appliances - IMPORTANT: Use canonical names that match ES index
        "refrigerators": ["refrigerator", "refrigerators", "fridge", "fridges", "ref", "double door", "single door"],
        "washing machine": ["washing machine", "washer", "washers", "wm", "front load", "top load"],
        "air conditioner": ["air conditioner", "ac", "acs", "airconditioner", "split ac", "window ac", "inverter ac"],
        "microwave oven": ["microwave", "microwave oven", "oven", "microwaves", "otg"],
        "water heater and geysers": ["geyser", "geysers", "water heater", "water heaters", "instant geyser", "storage geyser"],
        "kitchen appliances": ["ro", "water purifier", "purifier", "water filter", "kent", "aquaguard"],
        "air cooler": ["cooler", "air cooler", "coolers", "desert cooler", "room cooler"],
        "home appliance": ["air fryer", "airfryer", "air fryers"],
        # Other categories
        "watch and wearable": ["watch", "watches", "smartwatch", "smart watch", "fitness band", "wearable", "wearables"],
        "kitchen appliances": ["mixer", "mixer grinder", "grinder", "juicer", "blender", "chimney", "induction"],
        "furniture": ["furniture", "sofa", "bed", "mattress", "wardrobe", "table", "chair"],
        "tractor": ["tractor", "tractors", "farm equipment"],
        "tyres": ["tyre", "tyres", "tire", "tires"],
        "cycle": ["cycle", "cycles", "bicycle", "bicycles"],
    },
    
    # =================== BRAND NAMES ===================
    "brands": {
        # Mobile brands
        "samsung": ["samsung", "sumsung", "samsang", "samung", "samsng", "smasung"],
        "apple": ["apple", "iphone", "appel", "aple"],
        "vivo": ["vivo", "vevo", "viovo", "vivio"],
        "oppo": ["oppo", "opo", "opp0"],
        "realme": ["realme", "relme", "realmee", "reamle", "realmi"],
        "redmi": ["redmi", "redmii", "radmi", "ridmi"],
        "poco": ["poco", "pocco", "poko"],
        "oneplus": ["oneplus", "one plus", "1plus", "1+", "onplus"],
        "motorola": ["motorola", "moto", "motarola", "motorla"],
        "nokia": ["nokia", "nokiya", "noika"],
        "google": ["google", "pixel", "googel"],
        "nothing": ["nothing", "nothin"],
        # Appliance brands
        "lg": ["lg", "elg", "lgg"],
        "whirlpool": ["whirlpool", "whirpool", "whrilpool", "whirpol"],
        "godrej": ["godrej", "godrage", "godraj"],
        "haier": ["haier", "haiar", "hier", "haire"],
        "voltas": ["voltas", "voltas", "voltass", "volts"],
        "daikin": ["daikin", "daikinn", "daikan", "dakin"],
        "bluestar": ["bluestar", "blue star", "bluestarr"],
        "carrier": ["carrier", "carrir", "carier"],
        "hitachi": ["hitachi", "hitaachi", "hitchi"],
        "panasonic": ["panasonic", "panasoic", "panasnic"],
        "bosch": ["bosch", "bosh", "bosche"],
        "ifb": ["ifb", "ifbb"],
        "kent": ["kent", "kentt"],
        "livpure": ["livpure", "livpur", "livepure"],
        # Vehicle brands
        "honda": ["honda", "hunda", "hoda", "honada", "hnda"],
        "hero": ["hero", "heero", "hiro", "heroo"],
        "tvs": ["tvs", "tves", "tvss"],
        "yamaha": ["yamaha", "yamha", "yemaha", "yahmaha"],
        "bajaj": ["bajaj", "bajag", "bajaaj", "bajjaj"],
        "suzuki": ["suzuki", "suzki", "suzuski", "maruti suzuki", "maruti"],
        "royal enfield": ["royal enfield", "royalenfield", "re", "bullet", "enfield"],
        "ktm": ["ktm", "ktmm"],
        "hyundai": ["hyundai", "hundai", "hyundia", "hyundaii"],
        "tata": ["tata", "tatta", "tataa"],
        "mahindra": ["mahindra", "mahendra", "mahindhra"],
        "kia": ["kia", "kiaa"],
        "mg": ["mg", "morris garages"],
        # Furniture brands
        "nilkamal": ["nilkamal", "nilkaml", "nillkamal"],
        "godrej interio": ["godrej interio", "godrej"],
        "urban ladder": ["urban ladder", "urbanladder"],
    },
    
    # =================== PRODUCT MODELS ===================
    "models": {
        # Two-wheeler models
        "activa": ["activa", "actva", "aktivia", "activaa", "aktiva"],
        "splendor": ["splendor", "splendour", "splendr", "splendar", "splndr"],
        "pulsar": ["pulsar", "pulser", "plsar", "pulsaar", "pulzar", "pulsr"],
        "bullet": ["bullet", "bullt", "bullat", "bulet", "bulleet"],
        "jupiter": ["jupiter", "jupitar", "jupter", "jupitr"],
        "apache": ["apache", "apachi", "apche", "apachy"],
        "unicorn": ["unicorn", "unicron", "unikorn", "unicrn"],
        "access": ["access", "acces", "acess", "accses"],
        "ntorq": ["ntorq", "ntork", "n torq", "ntroq"],
        "fascino": ["fascino", "facino", "fascno"],
        "chetak": ["chetak", "chetk", "chetek", "chetaak"],
        "ather": ["ather", "athar", "atherrr"],
        "ola s1": ["ola s1", "ola", "ola electric"],
        "hunter": ["hunter", "huntr", "huntar"],
        "meteor": ["meteor", "metor", "meetor"],
        "classic": ["classic", "clasic", "klassic", "classik"],
        "himalayan": ["himalayan", "himalyan", "himlayan"],
        # Car models
        "creta": ["creta", "creat", "creata", "kreta"],
        "nexon": ["nexon", "nexn", "nexxon", "naxon"],
        "swift": ["swift", "swft", "swfit", "swiift"],
        "baleno": ["baleno", "beleno", "balleno"],
        "brezza": ["brezza", "breza", "breeza", "brezaa"],
        "seltos": ["seltos", "sletos", "seltoss"],
        "venue": ["venue", "venu", "vanue"],
        "thar": ["thar", "tharr", "tharr"],
        "fortuner": ["fortuner", "fortunar", "fortnr", "fortunerr"],
        "innova": ["innova", "inova", "innva", "inovaa"],
        "xuv700": ["xuv700", "xuv 700", "xuw700"],
        "harrier": ["harrier", "harier", "harrierr"],
        "safari": ["safari", "safri", "safarri"],
        # Phone models
        "galaxy": ["galaxy", "galalxy", "galxy", "galaxxy", "galazy", "gallaxy"],
        "iphone": ["iphone", "iphon", "ipone", "ifone", "i phone"],
        "note": ["note", "notte", "noote"],
        "pixel": ["pixel", "pixle", "pixxel"],
    }
}


def fuzzy_match_word(word: str, dictionary_type: str = "all", min_similarity: float = 0.75) -> Optional[Dict]:
    """
    Find the best fuzzy match for a word in the dictionary.
    
    Args:
        word: The word to match
        dictionary_type: "categories", "brands", "models", or "all"
        min_similarity: Minimum similarity score (0.0 to 1.0) to consider a match
    
    Returns:
        Dict with 'match', 'canonical', 'type', 'similarity' or None if no match
    """
    if not word or len(word) < 2:
        return None
    
    word_lower = word.lower().strip()
    best_match = None
    best_similarity = min_similarity
    
    # Determine which dictionaries to search
    if dictionary_type == "all":
        dict_types = ["categories", "brands", "models"]
    else:
        dict_types = [dictionary_type]
    
    for dtype in dict_types:
        if dtype not in FUZZY_MATCH_DICTIONARY:
            continue
            
        for canonical, variations in FUZZY_MATCH_DICTIONARY[dtype].items():
            for variation in variations:
                # Calculate similarity
                similarity = normalized_similarity(word_lower, variation.lower())
                
                # Check for partial match (word is part of variation or vice versa)
                # BUT only apply boost if:
                # 1. Lengths are similar (within 2 chars)
                # 2. BOTH word and variation are at least 3 chars (prevents "ac" in "rack")
                # This prevents "car" matching "carrier" or "rack" matching "ac"
                len_diff = abs(len(word_lower) - len(variation.lower()))
                var_lower = variation.lower()
                if (len_diff <= 2 and 
                    len(word_lower) >= 3 and len(var_lower) >= 3 and
                    (word_lower in var_lower or var_lower in word_lower)):
                    similarity = max(similarity, 0.85)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "match": variation,
                        "canonical": canonical,
                        "type": dtype,
                        "similarity": similarity
                    }
    
    return best_match


def fuzzy_correct_query(query: str, min_similarity: float = 0.70) -> Dict:
    """
    Apply fuzzy correction to an entire query.
    
    Args:
        query: The input query string
        min_similarity: Minimum similarity threshold
    
    Returns:
        Dict with 'corrected_query', 'corrections', 'detected_category', 'detected_brand'
    """
    if not query:
        return {
            "corrected_query": query,
            "corrections": [],
            "detected_category": None,
            "detected_brand": None
        }
    
    words = query.lower().strip().split()
    corrections = []
    corrected_words = []
    detected_category = None
    detected_brand = None
    detected_model = None
    
    # First, try matching full query (for multi-word matches)
    full_query = " ".join(words)
    full_match = fuzzy_match_word(full_query, "categories", min_similarity)
    if full_match and full_match["similarity"] >= 0.80:
        detected_category = full_match["canonical"]
        corrections.append({
            "original": full_query,
            "corrected": full_match["canonical"],
            "type": "category",
            "similarity": full_match["similarity"]
        })
        return {
            "corrected_query": full_match["canonical"],
            "corrections": corrections,
            "detected_category": detected_category,
            "detected_brand": None
        }
    
    # Try matching word by word
    for word in words:
        if len(word) < 3:
            # Keep short words as-is (numbers, prepositions, etc.)
            corrected_words.append(word)
            continue
        
        # Try to match against all dictionaries
        match = fuzzy_match_word(word, "all", min_similarity)
        
        if match:
            corrected_words.append(match["canonical"])
            corrections.append({
                "original": word,
                "corrected": match["canonical"],
                "type": match["type"],
                "similarity": match["similarity"]
            })
            
            # Track detected entities
            if match["type"] == "categories" and not detected_category:
                detected_category = match["canonical"]
            elif match["type"] == "brands" and not detected_brand:
                detected_brand = match["canonical"]
            elif match["type"] == "models" and not detected_model:
                detected_model = match["canonical"]
        else:
            corrected_words.append(word)
    
    # Try two-word combinations (for "four wheeler", "washing machine", etc.)
    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        match = fuzzy_match_word(two_word, "categories", min_similarity)
        if match and match["similarity"] >= 0.75:
            detected_category = match["canonical"]
            corrections.append({
                "original": two_word,
                "corrected": match["canonical"],
                "type": "category",
                "similarity": match["similarity"]
            })
    
    corrected_query = " ".join(corrected_words)
    
    return {
        "corrected_query": corrected_query,
        "corrections": corrections,
        "detected_category": detected_category,
        "detected_brand": detected_brand,
        "detected_model": detected_model
    }


def get_category_from_fuzzy(word: str, threshold: float = 0.70) -> Optional[str]:
    """
    Get category from a potentially misspelled word using fuzzy matching.
    This is a quick helper for category detection fallback.
    """
    match = fuzzy_match_word(word, "categories", threshold)
    if match:
        return match["canonical"]
    return None


# Known category keywords that should NOT be matched to brands
CATEGORY_KEYWORDS_EXCLUDE_FROM_BRAND_MATCH = {
    "car", "cars", "bike", "bikes", "mobile", "phone", "phones", "tv", "television",
    "refrigerator", "fridge", "ac", "air", "conditioner", "washing", "machine", "laptop", "tablet", 
    "watch", "camera", "cooler", "microwave", "oven", "chimney", "mixer", "grinder",
    "tyre", "tyres", "cycle", "bicycle", "tractor", "scooter", "scooty", "furniture",
    "sofa", "bed", "mattress", "wardrobe", "table", "chair", "smartphone", "headphone",
    "speaker", "printer", "monitor", "keyboard", "mouse", "router", "inverter",
    "battery", "fan", "heater", "purifier", "vacuum", "iron", "dryer", "freezer",
    "dishwasher", "geyser", "two wheeler", "four wheeler", "vehicle", "automobile"
}


def get_brand_from_fuzzy(word: str, threshold: float = 0.70) -> Optional[str]:
    """
    Get brand from a potentially misspelled word using fuzzy matching.
    This is a quick helper for brand detection fallback.
    
    Note: Excludes known category keywords to prevent "car" → "carrier" type matches.
    """
    # Don't try to match category keywords to brands
    if word.lower().strip() in CATEGORY_KEYWORDS_EXCLUDE_FROM_BRAND_MATCH:
        return None
    
    match = fuzzy_match_word(word, "brands", threshold)
    if match:
        return match["canonical"]
    return None


# =================== QUERY PROCESSOR ===================
class QueryProcessor:
    """Handles query understanding, typo correction, and entity extraction"""
    
    def __init__(self):
        self.category_list = list(CATEGORY_CANONICAL.keys())
        self.synonym_map = BUSINESS_SYNONYMS
        self._build_brand_set()
    
    def _build_brand_set(self):
        """Build set of all known brands"""
        self.brand_set = set(b.lower() for b in BRAND_NAMES)
        for synonyms in self.synonym_map.values():
            for syn in synonyms:
                if len(syn) > 2:
                    self.brand_set.add(syn.lower())
    
    def correct_typos(self, query: str) -> str:
        """Apply typo correction while protecting brand names and common words"""
        if not query:
            return query
        
        query_lower = query.lower().strip()
        
        # =================== PRE-PROCESSING: COMMON TYPO FIXES ===================
        import re
        
        # Fix letter O used instead of zero (common keyboard typo)
        # x2oo -> x200, y1oo -> y100, etc. (only in context of model numbers)
        query_lower = re.sub(r'([xyvt])(\d*)oo\b', r'\g<1>\g<2>00', query_lower)
        query_lower = re.sub(r'([xyvt])(\d*)o(\d)\b', r'\g<1>\g<2>0\3', query_lower)
        
        # Common brand typos - fix at start of word or standalone
        # Using lookahead to handle compound words like "poccom6" -> "pocom6"
        brand_typo_fixes = [
            (r'\bvevo', 'vivo'), (r'\bviovo', 'vivo'), (r'\bvivi(?=[^o]|$)', 'vivo'), 
            (r'\bvvo', 'vivo'), (r'\bvoivo', 'vivo'), (r'\bvivio', 'vivo'),
            (r'\bopp0', 'oppo'), (r'\bopo(?=[^o])', 'oppo'), (r'\bopoo', 'oppo'), (r'\b0ppo', 'oppo'),
            (r'\bsamung', 'samsung'), (r'\bsamsun(?![g])', 'samsung'), (r'\bsamsang', 'samsung'), 
            (r'\bsamsumg', 'samsung'), (r'\bsumsang', 'samsung'), (r'\bsumsung', 'samsung'),
            (r'\brelme', 'realme'), (r'\brealmee', 'realme'), (r'\brelame', 'realme'), (r'\breamle', 'realme'),
            (r'\brealmy', 'realme'), (r'\brealmi', 'realme'), (r'\brealmei', 'realme'),
            (r'\bredmii', 'redmi'), (r'\bradmi', 'redmi'), (r'\bridmi', 'redmi'),
            (r'\bpocco', 'poco'), (r'\bpoko', 'poco'), (r'\bpocoo', 'poco'),
            (r'\bmotarola', 'motorola'), (r'\bmotorla', 'motorola'), (r'\bmotorolla', 'motorola'),
            (r'\bmotarolla', 'motorola'), (r'\bmotrola', 'motorola'), (r'\bmotorloa', 'motorola'),
            (r'\b1plus', 'oneplus'), (r'\bone\+', 'oneplus'), (r'\bonplus', 'oneplus'),
            # "one plus" -> "oneplus" normalization (IMPORTANT: fixes "one plus 15" issue)
            (r'\bone\s+plus\b', 'oneplus'),
            # Tablet typos (device, not medicine)
            (r'\btabelt\b', 'tablet'), (r'\btblet\b', 'tablet'), (r'\btabet\b', 'tablet'),
            (r'\btablet\b', 'tablet'), (r'\btablat\b', 'tablet'), (r'\btablte\b', 'tablet'),
            (r'\btablt\b', 'tablet'), (r'\btabelet\b', 'tablet'), (r'\btablets\b', 'tablet'),
            # Compound word fixes (full word match) - DO NOT expand short forms here, just fix typos
            (r'\bwashingmachine\b', 'washing machine'), (r'\bairconditioner\b', 'air conditioner'),
            (r'\bwaterpurifier\b', 'water purifier'), (r'\bwaterheater\b', 'water heater'),
            (r'\bmicrowaveoven\b', 'microwave oven'),
            # Electric vehicle compounds
            (r'\belectricscooter\b', 'electric scooter'), (r'\belectriccar\b', 'electric car'),
            (r'\belectricbike\b', 'electric bike'), (r'\bevscooter\b', 'ev scooter'),
            (r'\bevcar\b', 'ev car'), (r'\bevbike\b', 'ev bike'),
            # Air fryer compound words
            (r'\bairfryer\b', 'air fryer'), (r'\bairfrier\b', 'air fryer'), (r'\bairfriar\b', 'air fryer'),
            
            # =================== SAMSUNG GALAXY MODEL NORMALIZATION ===================
            # Pattern: s + number + optional suffix (ultra/plus/fe)
            # Handles: s25ultra, s 25 ultra, s25 ultr, s25ultr, s25altra, s25eltra, etc.
            # Normalize to "samsung sXX <suffix>" format for better search matching
            # S26 series (future-proofing + user queries for upcoming models)
            (r'\bs\s*26\s*ultra\b', 'samsung s26 ultra'), (r'\bs\s*26\s*ultr\b', 'samsung s26 ultra'),
            (r'\bs\s*26\s*altra\b', 'samsung s26 ultra'), (r'\bs\s*26\s*eltra\b', 'samsung s26 ultra'),
            (r'\bs\s*26\s*utra\b', 'samsung s26 ultra'), (r'\bs\s*26\s*utlra\b', 'samsung s26 ultra'),
            (r'\bs26ultra\b', 'samsung s26 ultra'), (r'\bs\s*26\s*ulta\b', 'samsung s26 ultra'),
            (r'\bs\s*26\s*plus\b', 'samsung s26 plus'), (r'\bs26plus\b', 'samsung s26 plus'),
            (r'\bs\s*26\s*fe\b', 'samsung s26 fe'), (r'\bs26fe\b', 'samsung s26 fe'),
            # S25 series - all patterns must handle optional space after "s"
            (r'\bs\s*25\s*ultra\b', 'samsung s25 ultra'), (r'\bs\s*25\s*ultr\b', 'samsung s25 ultra'),
            (r'\bs\s*25\s*altra\b', 'samsung s25 ultra'), (r'\bs\s*25\s*eltra\b', 'samsung s25 ultra'),
            (r'\bs\s*25\s*utra\b', 'samsung s25 ultra'), (r'\bs\s*25\s*utlra\b', 'samsung s25 ultra'),
            (r'\bs25ultra\b', 'samsung s25 ultra'), (r'\bs\s*25\s*ulta\b', 'samsung s25 ultra'),
            (r'\bs\s*25\s*plus\b', 'samsung s25 plus'), (r'\bs25plus\b', 'samsung s25 plus'),
            (r'\bs\s*25\s*fe\b', 'samsung s25 fe'), (r'\bs25fe\b', 'samsung s25 fe'),
            # S24 series
            (r'\bs\s*24\s*ultra\b', 'samsung s24 ultra'), (r'\bs\s*24\s*ultr\b', 'samsung s24 ultra'),
            (r'\bs\s*24\s*altra\b', 'samsung s24 ultra'), (r'\bs\s*24\s*eltra\b', 'samsung s24 ultra'),
            (r'\bs\s*24\s*utra\b', 'samsung s24 ultra'), (r'\bs\s*24\s*utlra\b', 'samsung s24 ultra'),
            (r'\bs24ultra\b', 'samsung s24 ultra'), (r'\bs\s*24\s*ulta\b', 'samsung s24 ultra'),
            (r'\bs\s*24\s*plus\b', 'samsung s24 plus'), (r'\bs24plus\b', 'samsung s24 plus'),
            (r'\bs\s*24\s*fe\b', 'samsung s24 fe'), (r'\bs24fe\b', 'samsung s24 fe'),
            # S23 series
            (r'\bs\s*23\s*ultra\b', 'samsung s23 ultra'), (r'\bs\s*23\s*ultr\b', 'samsung s23 ultra'),
            (r'\bs\s*23\s*altra\b', 'samsung s23 ultra'), (r'\bs\s*23\s*eltra\b', 'samsung s23 ultra'),
            (r'\bs\s*23\s*utra\b', 'samsung s23 ultra'), (r'\bs\s*23\s*utlra\b', 'samsung s23 ultra'),
            (r'\bs23ultra\b', 'samsung s23 ultra'), (r'\bs\s*23\s*ulta\b', 'samsung s23 ultra'),
            (r'\bs\s*23\s*plus\b', 'samsung s23 plus'), (r'\bs23plus\b', 'samsung s23 plus'),
            (r'\bs\s*23\s*fe\b', 'samsung s23 fe'), (r'\bs23fe\b', 'samsung s23 fe'),
            # S22 series
            (r'\bs\s*22\s*ultra\b', 's22 ultra'), (r'\bs22ultra\b', 's22 ultra'),
            (r'\bs\s*22\s*plus\b', 's22 plus'), (r'\bs22plus\b', 's22 plus'),
            # S21 series
            (r'\bs\s*21\s*ultra\b', 's21 ultra'), (r'\bs21ultra\b', 's21 ultra'),
            (r'\bs\s*21\s*plus\b', 's21 plus'), (r'\bs21plus\b', 's21 plus'),
            (r'\bs\s*21\s*fe\b', 's21 fe'), (r'\bs21fe\b', 's21 fe'),
            # A series (A55, A54, A35, A34, etc.)
            (r'\ba\s*55\s*ultra\b', 'a55'), (r'\ba55ultra\b', 'a55'),
            (r'\ba\s*54\s*ultra\b', 'a54'), (r'\ba54ultra\b', 'a54'),
            (r'\ba\s*35\b', 'a35'), (r'\ba\s*34\b', 'a34'),
            # Galaxy + model compound (galaxys25 -> galaxy s25)
            (r'\bgalaxys25\b', 'galaxy s25'), (r'\bgalaxys24\b', 'galaxy s24'),
            (r'\bgalaxys23\b', 'galaxy s23'), (r'\bgalaxys22\b', 'galaxy s22'),
            (r'\bgalaxys21\b', 'galaxy s21'), (r'\bgalaxya55\b', 'galaxy a55'),
            (r'\bgalaxya54\b', 'galaxy a54'), (r'\bgalaxya35\b', 'galaxy a35'),
            # Samsung + model compound (samsungs25 -> samsung s25)
            (r'\bsamsungs26\b', 'samsung s26'), (r'\bsamsungs25\b', 'samsung s25'), (r'\bsamsungs24\b', 'samsung s24'),
            (r'\bsamsungs23\b', 'samsung s23'), (r'\bsamsungm34\b', 'samsung m34'),
            (r'\bsamsungf54\b', 'samsung f54'), (r'\bsamsungf55\b', 'samsung f55'),
            
            # Samsung Galaxy + number without 's' prefix (galaxy 24 → samsung s24, galaxy 26 ultra → samsung s26 ultra)
            (r'\bgalaxy\s*(2[1-9])\s*(ultra|plus|fe|lite)?\b', r'samsung s\1 \2'),
            # Standalone Samsung S-series without 'samsung' or 'galaxy' prefix (s26 → samsung s26)
            # Uses negative lookbehind to avoid double-prefixing when samsung/galaxy already present
            (r'(?<!samsung )(?<!galaxy )\bs(2[1-9])\b(?!\s*(ultra|plus|fe|inch|star|kg|ton|litre|cm|mm))', r'samsung s\1'),
            
            # =================== iPhone COMPOUND FORMS ===================
            (r'\biphone16promax\b', 'iphone 16 pro max'), (r'\biphone15promax\b', 'iphone 15 pro max'),
            (r'\biphone16pro\b', 'iphone 16 pro'), (r'\biphone15pro\b', 'iphone 15 pro'),
            (r'\biphone17promax\b', 'iphone 17 pro max'), (r'\biphone17pro\b', 'iphone 17 pro'),
            (r'\biphone14promax\b', 'iphone 14 pro max'), (r'\biphone14pro\b', 'iphone 14 pro'),
            (r'\biphone13promax\b', 'iphone 13 pro max'), (r'\biphone13pro\b', 'iphone 13 pro'),
            # Number + iphone compound (17iphone -> iphone 17, 16iphone -> iphone 16)
            (r'\b(1[3-7])iphone\b', r'iphone \1'), (r'\b(1[3-7])\s*iphone\b', r'iphone \1'),
            # iPhone + Number compound without space (iphone17 -> iphone 17)
            (r'\biphone(1[3-7])\b', r'iphone \1'),
            
            # =================== iPhone TYPO VARIATIONS ===================
            # Pattern: Number + iPhone typos (17iphon, 16iphn, 15iphne, etc.)
            # iphon typos (missing 'e')
            (r'\b(1[3-7])\s*iphon\b', r'iphone \1'), (r'\b(1[3-7])iphon\b', r'iphone \1'),
            (r'\biphon\s*(1[3-7])\b', r'iphone \1'), (r'\biphon(1[3-7])\b', r'iphone \1'),
            # iphn typos (missing 'o' and 'e')
            (r'\b(1[3-7])\s*iphn\b', r'iphone \1'), (r'\b(1[3-7])iphn\b', r'iphone \1'),
            (r'\biphn\s*(1[3-7])\b', r'iphone \1'), (r'\biphn(1[3-7])\b', r'iphone \1'),
            # iphne typos (swapped 'ne')
            (r'\b(1[3-7])\s*iphne\b', r'iphone \1'), (r'\b(1[3-7])iphne\b', r'iphone \1'),
            (r'\biphne\s*(1[3-7])\b', r'iphone \1'), (r'\biphne(1[3-7])\b', r'iphone \1'),
            # ipone typos (missing 'h')
            (r'\b(1[3-7])\s*ipone\b', r'iphone \1'), (r'\b(1[3-7])ipone\b', r'iphone \1'),
            (r'\bipone\s*(1[3-7])\b', r'iphone \1'), (r'\bipone(1[3-7])\b', r'iphone \1'),
            # ifone typos ('f' instead of 'ph')
            (r'\b(1[3-7])\s*ifone\b', r'iphone \1'), (r'\b(1[3-7])ifone\b', r'iphone \1'),
            (r'\bifone\s*(1[3-7])\b', r'iphone \1'), (r'\bifone(1[3-7])\b', r'iphone \1'),
            # iphone typos with 'pro' suffix - Number + typo + pro (17iphonpro, 16iphnpro)
            (r'\b(1[3-7])\s*iphon\s*pro\b', r'iphone \1 pro'), (r'\b(1[3-7])iphonpro\b', r'iphone \1 pro'),
            (r'\b(1[3-7])\s*iphn\s*pro\b', r'iphone \1 pro'), (r'\b(1[3-7])iphnpro\b', r'iphone \1 pro'),
            (r'\b(1[3-7])\s*iphne\s*pro\b', r'iphone \1 pro'), (r'\b(1[3-7])iphnepro\b', r'iphone \1 pro'),
            # Typo + number + pro (iphon17pro, iphn16pro)
            (r'\biphon\s*(1[3-7])\s*pro\b', r'iphone \1 pro'), (r'\biphon(1[3-7])pro\b', r'iphone \1 pro'),
            (r'\biphn\s*(1[3-7])\s*pro\b', r'iphone \1 pro'), (r'\biphn(1[3-7])pro\b', r'iphone \1 pro'),
            (r'\biphne\s*(1[3-7])\s*pro\b', r'iphone \1 pro'), (r'\biphne(1[3-7])pro\b', r'iphone \1 pro'),
            # Typo + promax / pro max patterns
            (r'\biphon\s*pro\s*max\b', 'iphone pro max'), (r'\biphonpromax\b', 'iphone pro max'),
            (r'\biphn\s*pro\s*max\b', 'iphone pro max'), (r'\biphnpromax\b', 'iphone pro max'),
            (r'\biphne\s*pro\s*max\b', 'iphone pro max'), (r'\biphnepromax\b', 'iphone pro max'),
            (r'\bipone\s*pro\s*max\b', 'iphone pro max'), (r'\biponepromax\b', 'iphone pro max'),
            # Just typo with pro (iphnpro, iphonpro without number)
            (r'\biphon\s*pro\b', 'iphone pro'), (r'\biphonpro\b', 'iphone pro'),
            (r'\biphn\s*pro\b', 'iphone pro'), (r'\biphnpro\b', 'iphone pro'),
            (r'\biphne\s*pro\b', 'iphone pro'), (r'\biphnepro\b', 'iphone pro'),
            # Standalone typos (fix to "iphone")
            (r'\biphon\b', 'iphone'), (r'\biphn\b', 'iphone'), (r'\biphne\b', 'iphone'),
            (r'\bipone\b', 'iphone'), (r'\bifone\b', 'iphone'), (r'\biphoe\b', 'iphone'),
            (r'\biph0ne\b', 'iphone'), (r'\b1phone\b', 'iphone'), (r'\biphoone\b', 'iphone'),
            # uphone typos (keyboard 'u' is adjacent to 'i')
            (r'\buphone\b', 'iphone'), (r'\buphon\b', 'iphone'), (r'\bupone\b', 'iphone'), (r'\bufone\b', 'iphone'),
            
            # =================== STANDALONE iPhone MODEL NUMBERS ===================
            # Critical: "17pro max", "16promax", etc. without "iphone" prefix → treat as iPhone
            # These patterns ONLY match when no other brand context exists
            (r'^17\s*pro\s*max$', 'iphone 17 pro max'), (r'^17promax$', 'iphone 17 pro max'),
            (r'^17\s*pro$', 'iphone 17 pro'), (r'^17pro$', 'iphone 17 pro'),
            (r'^16\s*pro\s*max$', 'iphone 16 pro max'), (r'^16promax$', 'iphone 16 pro max'),
            (r'^16\s*pro$', 'iphone 16 pro'), (r'^16pro$', 'iphone 16 pro'),
            (r'^15\s*pro\s*max$', 'iphone 15 pro max'), (r'^15promax$', 'iphone 15 pro max'),
            (r'^15\s*pro$', 'iphone 15 pro'), (r'^15pro$', 'iphone 15 pro'),
            (r'^14\s*pro\s*max$', 'iphone 14 pro max'), (r'^14promax$', 'iphone 14 pro max'),
            (r'^14\s*pro$', 'iphone 14 pro'), (r'^14pro$', 'iphone 14 pro'),
            (r'^13\s*pro\s*max$', 'iphone 13 pro max'), (r'^13promax$', 'iphone 13 pro max'),
            (r'^13\s*pro$', 'iphone 13 pro'), (r'^13pro$', 'iphone 13 pro'),
            
            # =================== VIVO MODEL NORMALIZATION ===================
            # vivo300 → vivo x300, vivo200 → vivo x200, etc.
            # Vivo X series models typically have format "X###"
            (r'\bvivo\s*300\b', 'vivo x300'), (r'\bvivo300\b', 'vivo x300'),
            (r'\bvivo\s*200\b', 'vivo x200'), (r'\bvivo200\b', 'vivo x200'),
            (r'\bvivo\s*100\b', 'vivo x100'), (r'\bvivo100\b', 'vivo x100'),
            (r'\bvivo\s*90\b', 'vivo x90'), (r'\bvivo90\b', 'vivo x90'),
            (r'\bvivo\s*80\b', 'vivo x80'), (r'\bvivo80\b', 'vivo x80'),
            (r'\bvivo\s*70\b', 'vivo x70'), (r'\bvivo70\b', 'vivo x70'),
            (r'\bvivo\s*60\b', 'vivo x60'), (r'\bvivo60\b', 'vivo x60'),
            (r'\bvivo\s*50\b', 'vivo x50'), (r'\bvivo50\b', 'vivo x50'),
            # Vivo V series
            (r'\bvivov30\b', 'vivo v30'), (r'\bvivov29\b', 'vivo v29'), (r'\bvivov27\b', 'vivo v27'),
            # Vivo Y series  
            (r'\bvivoy100\b', 'vivo y100'), (r'\bvivoy200\b', 'vivo y200'), (r'\bvivoy300\b', 'vivo y300'),
            # Vivo T series
            (r'\bvivot3\b', 'vivo t3'), (r'\bvivot2\b', 'vivo t2'), (r'\bvivot1\b', 'vivo t1'),
            
            # Brand + Category compound words (lgac -> lg ac, samsungtv -> samsung tv)
            (r'\blgac\b', 'lg ac'), (r'\blgtv\b', 'lg tv'), (r'\blgfridge\b', 'lg fridge'),
            (r'\blgrefrigerator\b', 'lg refrigerator'), (r'\blgwashingmachine\b', 'lg washing machine'),
            (r'\bsamsungtv\b', 'samsung tv'), (r'\bsamsungac\b', 'samsung ac'),
            (r'\bsamsungfridge\b', 'samsung fridge'), (r'\bsamsungrefrigerator\b', 'samsung refrigerator'),
            (r'\bsamsungwashingmachine\b', 'samsung washing machine'),
            (r'\bsonytv\b', 'sony tv'), (r'\bsonyac\b', 'sony ac'),
            (r'\bhaierac\b', 'haier ac'), (r'\bvoltasac\b', 'voltas ac'), (r'\bdaikinac\b', 'daikin ac'),
            (r'\bgodrejac\b', 'godrej ac'), (r'\bgodrejfridge\b', 'godrej fridge'),
            (r'\bwhirlpoolfridge\b', 'whirlpool fridge'), (r'\bifbwashingmachine\b', 'ifb washing machine'),
            (r'\bboschfridge\b', 'bosch fridge'), (r'\bboschwashingmachine\b', 'bosch washing machine'),
            # Concatenated product names - add space
            (r'\baccess125\b', 'access 125'), (r'\bntorq125\b', 'ntorq 125'), (r'\bntorq150\b', 'ntorq 150'),
            (r'\bpulsar125\b', 'pulsar 125'), (r'\bpulsar150\b', 'pulsar 150'), (r'\bpulsar160\b', 'pulsar 160'),
            (r'\bpulsar220\b', 'pulsar 220'), (r'\bpulsarns\b', 'pulsar ns'),
            (r'\br15v3\b', 'r15 v3'), (r'\br15v4\b', 'r15 v4'), (r'\br15v5\b', 'r15 v5'),
            (r'\bgt650\b', 'gt 650'), (r'\bgt310\b', 'gt 310'),
            (r'\bxsr155\b', 'xsr 155'), (r'\bfz25\b', 'fz 25'), (r'\bmt15\b', 'mt 15'),
            
            # Samsung Galaxy typos
            (r'\bgalalxy\b', 'galaxy'), (r'\bgalxy\b', 'galaxy'), (r'\bgalaxxy\b', 'galaxy'),
            (r'\bgalazy\b', 'galaxy'), (r'\bgalaxi\b', 'galaxy'), (r'\bgallaxy\b', 'galaxy'),
            (r'\bglxy\b', 'galaxy'), (r'\bgalexy\b', 'galaxy'), (r'\bgalxey\b', 'galaxy'),
            
            # Washing machine typos
            (r'\bamchine\b', 'machine'), (r'\bamachine\b', 'machine'), (r'\bmachne\b', 'machine'),
            (r'\bmachin\b', 'machine'), (r'\bmacine\b', 'machine'), (r'\bmchine\b', 'machine'),
            (r'\bmahcine\b', 'machine'),
            (r'\bwashin\b', 'washing'), (r'\bwashng\b', 'washing'), (r'\bwahsing\b', 'washing'),
            (r'\bwashign\b', 'washing'), (r'\bwashnig\b', 'washing'),
            
            # Short word typos - only fix obvious typos, NOT short forms (those are handled in display)
            (r'\bphn\b', 'phone'), (r'\bphne\b', 'phone'), (r'\bphon\b', 'phone'),
            (r'\bfone\b', 'phone'), (r'\bfon\b', 'phone'),
            # Phone typos - keyboard adjacent keys (r near e, o near i, etc.)
            (r'\bphonr\b', 'phone'), (r'\bphoen\b', 'phone'), (r'\bpohne\b', 'phone'),
            (r'\bphin\b', 'phone'), (r'\bphine\b', 'phone'), (r'\bpone\b', 'phone'),
            (r'\bphonne\b', 'phone'), (r'\bphonee\b', 'phone'), (r'\bphoner\b', 'phone'),
            (r'\bphons\b', 'phone'), (r'\bphonse\b', 'phone'), (r'\bfonr\b', 'phone'),
            (r'\bmbl\b', 'mobile'), (r'\bmobl\b', 'mobile'), (r'\bmobil\b', 'mobile'),
            # Mobile typos - keyboard adjacent keys and common misspellings
            (r'\bmobail\b', 'mobile'), (r'\bmobaile\b', 'mobile'), (r'\bmobel\b', 'mobile'),
            (r'\bmobal\b', 'mobile'), (r'\bmoblie\b', 'mobile'), (r'\bmoblile\b', 'mobile'),
            (r'\bmoble\b', 'mobile'), (r'\bmboile\b', 'mobile'), (r'\bmobiile\b', 'mobile'),
            (r'\bmobiil\b', 'mobile'), (r'\bmobille\b', 'mobile'), (r'\bmobole\b', 'mobile'),
            (r'\bmobiel\b', 'mobile'), (r'\bmoibel\b', 'mobile'), (r'\bmoible\b', 'mobile'),
            (r'\bmobyle\b', 'mobile'), (r'\bmobale\b', 'mobile'), (r'\bmobails\b', 'mobiles'),
            (r'\bmibile\b', 'mobile'), (r'\bmobail\b', 'mobile'),  # Additional typos
            (r'\bsmrtphone\b', 'smartphone'), (r'\bsmartfone\b', 'smartphone'), (r'\bsmartphn\b', 'smartphone'),
            
            # =================== EARPHONE/EARBUD/HEADPHONE TYPOS ===================
            # Earphone typos - comprehensive coverage for common misspellings
            (r'\bearphn\b', 'earphone'), (r'\bearphon\b', 'earphone'), (r'\bearfon\b', 'earphone'),
            (r'\bearfone\b', 'earphone'), (r'\bearpone\b', 'earphone'), (r'\berphn\b', 'earphone'),
            (r'\berpone\b', 'earphone'), (r'\berphone\b', 'earphone'), (r'\berfone\b', 'earphone'),
            (r'\berfon\b', 'earphone'), (r'\bearphne\b', 'earphone'), (r'\beaphone\b', 'earphone'),
            (r'\bearpho\b', 'earphone'), (r'\bearphons\b', 'earphones'), (r'\bearphnes\b', 'earphones'),
            # raphon typos (missing 'e' at start) - CRITICAL
            (r'\braphon\b', 'earphone'), (r'\braphne\b', 'earphone'), (r'\braphone\b', 'earphone'),
            (r'\braphn\b', 'earphone'), (r'\braphons\b', 'earphones'),
            # airphon typos ('air' instead of 'ear') - CRITICAL  
            (r'\bairphon\b', 'earphone'), (r'\bairphone\b', 'earphone'), (r'\bairphn\b', 'earphone'),
            (r'\bairfon\b', 'earphone'), (r'\bairfone\b', 'earphone'), (r'\bairphons\b', 'earphones'),
            (r'\bairphones\b', 'earphones'),
            # Earbud typos
            (r'\berbud\b', 'earbud'), (r'\berbuds\b', 'earbuds'), (r'\bairbud\b', 'earbud'),
            (r'\bairbuds\b', 'earbuds'), (r'\bearbds\b', 'earbuds'), (r'\bearbd\b', 'earbud'),
            # Headphone typos
            (r'\bheadphon\b', 'headphone'), (r'\bheadfon\b', 'headphone'), (r'\bheadfone\b', 'headphone'),
            (r'\bhedphone\b', 'headphone'), (r'\bhedphones\b', 'headphones'), (r'\bheadphn\b', 'headphone'),
            (r'\bhaedphone\b', 'headphone'), (r'\bheadphne\b', 'headphone'),
            # Neckband typos
            (r'\bneckbnd\b', 'neckband'), (r'\bnekband\b', 'neckband'), (r'\bneckbad\b', 'neckband'),
            (r'\bnckband\b', 'neckband'), (r'\bnecband\b', 'neckband'),
            
            # TV typos (fix typos only, keep "tv" as is for search)
            (r'\btelivision\b', 'television'), (r'\btelvision\b', 'television'),
            # Laptop/Computer typos
            (r'\blptop\b', 'laptop'), (r'\blaptp\b', 'laptop'),
            (r'\bnotbook\b', 'notebook'), (r'\bnotebk\b', 'notebook'),
            # Fridge typos (fix typos only)
            (r'\bfrig\b', 'fridge'), (r'\bfrdge\b', 'fridge'), (r'\bfridg\b', 'fridge'),
            (r'\brefridgerator\b', 'refrigerator'), (r'\brefregerator\b', 'refrigerator'),
            (r'\brefrigrater\b', 'refrigerator'), (r'\brefigerator\b', 'refrigerator'),
            (r'\brefirgerator\b', 'refrigerator'), (r'\brefrgrator\b', 'refrigerator'),
            # Kitchen appliance typos
            (r'\bmiksi\b', 'mixer'), (r'\bmikser\b', 'mixer'), (r'\bmixar\b', 'mixer'),
            (r'\bgijer\b', 'geyser'), (r'\bgijar\b', 'geyser'), (r'\bgizer\b', 'geyser'), (r'\bgeysar\b', 'geyser'),
            (r'\bjusar\b', 'juicer'), (r'\bjuisar\b', 'juicer'), (r'\bjucer\b', 'juicer'),
            (r'\bfarnichar\b', 'furniture'), (r'\bfurnichar\b', 'furniture'), (r'\bfurnitur\b', 'furniture'),
            (r'\bfarniture\b', 'furniture'), (r'\bfernicher\b', 'furniture'),
            # Mattress typos
            (r'\bmatras\b', 'mattress'), (r'\bmatress\b', 'mattress'), (r'\bmattres\b', 'mattress'),
            (r'\bmattrass\b', 'mattress'),
            # Sofa typos
            (r'\bsopha\b', 'sofa'), (r'\bsoffa\b', 'sofa'),
            # Microwave typos
            (r'\bmicrovave\b', 'microwave'), (r'\bmicrovawe\b', 'microwave'), (r'\bmicrowav\b', 'microwave'),
            (r'\bmicrowve\b', 'microwave'), (r'\bmicroave\b', 'microwave'),
            # AC/Conditioner typos
            (r'\bairconditionar\b', 'air conditioner'), (r'\bairconditionr\b', 'air conditioner'),
            (r'\bcondionar\b', 'conditioner'), (r'\bconditionr\b', 'conditioner'),
            # Speaker typos
            (r'\bspeeker\b', 'speaker'), (r'\bspeker\b', 'speaker'), (r'\bspeakar\b', 'speaker'),
            (r'\bspeekar\b', 'speaker'), (r'\bspekr\b', 'speaker'),
            # Tablet typos
            (r'\btablit\b', 'tablet'), (r'\btablet\b', 'tablet'), (r'\btablt\b', 'tablet'),
            (r'\btablat\b', 'tablet'), (r'\btabet\b', 'tablet'),
            # Laptop typos (more variations)
            (r'\blaptap\b', 'laptop'), (r'\blaptob\b', 'laptop'), (r'\bleptop\b', 'laptop'),
            (r'\blabtop\b', 'laptop'), (r'\blapptop\b', 'laptop'),
            # Godrej typo
            (r'\bgodraz\b', 'godrej'), (r'\bgodraj\b', 'godrej'), (r'\bgodrag\b', 'godrej'),
            
            # =================== VEHICLE BRAND TYPOS (Two-wheeler, Car, Tractor) ===================
            # Two-wheeler brands
            (r'\bhoda\b', 'honda'), (r'\bhonada\b', 'honda'), (r'\bhondaa\b', 'honda'), (r'\bhunda\b', 'honda'),
            (r'\bhiro\b', 'hero'), (r'\bheero\b', 'hero'), (r'\bhero\b', 'hero'),
            (r'\btvss\b', 'tvs'), (r'\btvz\b', 'tvs'), (r'\btves\b', 'tvs'),
            (r'\bbajja\b', 'bajaj'), (r'\bbajjaj\b', 'bajaj'), (r'\bbajj\b', 'bajaj'), (r'\bbjaj\b', 'bajaj'),
            (r'\byemaha\b', 'yamaha'), (r'\byamha\b', 'yamaha'), (r'\byamahaa\b', 'yamaha'), (r'\bymaha\b', 'yamaha'),
            (r'\bsuzki\b', 'suzuki'), (r'\bsuzukii\b', 'suzuki'), (r'\bsuzuky\b', 'suzuki'), (r'\bsuzki\b', 'suzuki'),
            (r'\broyalenfield\b', 'royal enfield'), (r'\broyal\s*enfild\b', 'royal enfield'), 
            (r'\broyalenfiled\b', 'royal enfield'), (r'\broyalenfld\b', 'royal enfield'),
            (r'\bkawasaki\b', 'kawasaki'), (r'\bkawaskai\b', 'kawasaki'), (r'\bkawsaki\b', 'kawasaki'),
            
            # Four-wheeler/Car brands
            (r'\bhundai\b', 'hyundai'), (r'\bhyudai\b', 'hyundai'), (r'\bhyundaii\b', 'hyundai'), 
            (r'\bhyndai\b', 'hyundai'), (r'\bhyundia\b', 'hyundai'),
            (r'\bmaruthi\b', 'maruti'), (r'\bmaruti\b', 'maruti'), (r'\bmaruri\b', 'maruti'),
            (r'\bmaruty\b', 'maruti'), (r'\bmaritti\b', 'maruti'),
            (r'\btatta\b', 'tata'), (r'\btataa\b', 'tata'), (r'\btatta\b', 'tata'),
            (r'\bmahindara\b', 'mahindra'), (r'\bmhindra\b', 'mahindra'), (r'\bmahindhra\b', 'mahindra'),
            (r'\bkiya\b', 'kia'), (r'\bkiaa\b', 'kia'), (r'\bkea\b', 'kia'),
            (r'\btoyata\b', 'toyota'), (r'\btoyotaa\b', 'toyota'), (r'\btoyoto\b', 'toyota'),
            (r'\bford\b', 'ford'), (r'\bfoord\b', 'ford'),
            (r'\bvolkswagon\b', 'volkswagen'), (r'\bvolkswagen\b', 'volkswagen'), (r'\bvw\b', 'volkswagen'),
            (r'\bskodaa\b', 'skoda'), (r'\bskodha\b', 'skoda'),
            (r'\bmg\b', 'mg'), (r'\bmgg\b', 'mg'),
            (r'\brenalt\b', 'renault'), (r'\brenaut\b', 'renault'), (r'\brenualt\b', 'renault'),
            (r'\bnisaan\b', 'nissan'), (r'\bnisssn\b', 'nissan'), (r'\bnisan\b', 'nissan'),
            
            # Car model typos
            (r'\bcirty\b', 'city'), (r'\bciti\b', 'city'), (r'\bcitty\b', 'city'),
            (r'\bcreat\b', 'creta'), (r'\bcreata\b', 'creta'), (r'\bcretta\b', 'creta'),
            (r'\bnexn\b', 'nexon'), (r'\bnexonn\b', 'nexon'), (r'\bnaxon\b', 'nexon'),
            (r'\bseltos\b', 'seltos'), (r'\bsaltos\b', 'seltos'), (r'\bseltoss\b', 'seltos'),
            (r'\bswfit\b', 'swift'), (r'\bswift\b', 'swift'), (r'\bswifft\b', 'swift'),
            (r'\bbalano\b', 'baleno'), (r'\bbalenoo\b', 'baleno'), (r'\bballeno\b', 'baleno'),
            (r'\bi20\b', 'i20'), (r'\bi 20\b', 'i20'), (r'\bi-20\b', 'i20'),
            (r'\bverna\b', 'verna'), (r'\bvernaa\b', 'verna'),
            (r'\bfortner\b', 'fortuner'), (r'\bfortunnr\b', 'fortuner'), (r'\bfortunar\b', 'fortuner'),
            (r'\binnova\b', 'innova'), (r'\binova\b', 'innova'), (r'\binnvoa\b', 'innova'),
            
            # Two-wheeler model typos
            (r'\bsplendour\b', 'splendor'), (r'\bsplendorr\b', 'splendor'), (r'\bsplendr\b', 'splendor'),
            (r'\bactivaa\b', 'activa'), (r'\bactva\b', 'activa'), (r'\bactivva\b', 'activa'),
            (r'\bjupitar\b', 'jupiter'), (r'\bjupiter\b', 'jupiter'), (r'\bjupiterr\b', 'jupiter'),
            (r'\bpulser\b', 'pulsar'), (r'\bpulsarr\b', 'pulsar'), (r'\bpulser\b', 'pulsar'),
            (r'\bclassic\b', 'classic'), (r'\bclassik\b', 'classic'), (r'\bclasic\b', 'classic'),
            (r'\bscooter\b', 'scooter'), (r'\bscootr\b', 'scooter'), (r'\bscooty\b', 'scooty'),
            
            # Tractor brands
            (r'\bswaraaj\b', 'swaraj'), (r'\bswarj\b', 'swaraj'), (r'\bswraj\b', 'swaraj'),
            (r'\bjohndeere\b', 'john deere'), (r'\bjohn\s*deer\b', 'john deere'), (r'\bjhon\s*deere\b', 'john deere'),
            (r'\bmassyferguson\b', 'massey ferguson'), (r'\bmassey\s*fergusn\b', 'massey ferguson'),
            (r'\bnewholland\b', 'new holland'), (r'\bnew\s*holand\b', 'new holland'),
            (r'\bescher\b', 'eicher'), (r'\beichre\b', 'eicher'), (r'\beacher\b', 'eicher'),
            (r'\bsonalka\b', 'sonalika'), (r'\bsonalike\b', 'sonalika'), (r'\bsonalica\b', 'sonalika'),
            (r'\bkubotaa\b', 'kubota'), (r'\bkubotta\b', 'kubota'),
            
            # Appliance brands
            (r'\bwhirpool\b', 'whirlpool'), (r'\bwhrilpool\b', 'whirlpool'), (r'\bwhirpol\b', 'whirlpool'),
            (r'\bhaiar\b', 'haier'), (r'\bhaierr\b', 'haier'), (r'\bhier\b', 'haier'),
            (r'\bgodrage\b', 'godrej'), (r'\bgodrej\b', 'godrej'), (r'\bgodrage\b', 'godrej'),
            (r'\blgg\b', 'lg'), (r'\belg\b', 'lg'),
            (r'\bbluestarr\b', 'blue star'), (r'\bblue\s*starr?\b', 'blue star'), (r'\bblustar\b', 'blue star'),
            (r'\bvolts\b', 'voltas'), (r'\bvoltass\b', 'voltas'),
            (r'\bdaikinn\b', 'daikin'), (r'\bdaikan\b', 'daikin'),
            (r'\bpanasnic\b', 'panasonic'), (r'\bpanasoic\b', 'panasonic'), (r'\bpanasonik\b', 'panasonic'),
            (r'\bhitaachi\b', 'hitachi'), (r'\bhitchi\b', 'hitachi'), (r'\bhitache\b', 'hitachi'),
            (r'\bcarrier\b', 'carrier'), (r'\bcarrir\b', 'carrier'), (r'\bcarier\b', 'carrier'),
            (r'\bifbb\b', 'ifb'), (r'\bifbb\b', 'ifb'),
            (r'\bbosche\b', 'bosch'), (r'\bbosh\b', 'bosch'), (r'\bbossh\b', 'bosch'),
            
            # Furniture brands
            (r'\bgodrej\b', 'godrej'), (r'\bgodrj\b', 'godrej'),
            (r'\bikea\b', 'ikea'), (r'\bikeaa\b', 'ikea'),
            (r'\burbanladder\b', 'urban ladder'), (r'\burban\s*ladr\b', 'urban ladder'),
            (r'\bpeppefry\b', 'pepperfry'), (r'\bpepper\s*fry\b', 'pepperfry'),
            (r'\bhometownn\b', 'hometown'), (r'\bhome\s*town\b', 'hometown'),
            (r'\bnilkaml\b', 'nilkamal'), (r'\bnilkaml\b', 'nilkamal'), (r'\bnilkamel\b', 'nilkamal'),
            
            # Cabinet/Furniture typos - IMPORTANT for "TV cabinet" queries
            (r'\bcabiner\b', 'cabinet'), (r'\bcabinet\b', 'cabinet'),
            (r'\bcabinate\b', 'cabinet'), (r'\bcabinat\b', 'cabinet'),
            (r'\bcabnet\b', 'cabinet'), (r'\bcabinent\b', 'cabinet'),
            (r'\bcabnit\b', 'cabinet'), (r'\bcabinett\b', 'cabinet'),
            (r'\bcabimet\b', 'cabinet'), (r'\bcabniet\b', 'cabinet'),
            (r'\bcabiniet\b', 'cabinet'), (r'\bcabinaet\b', 'cabinet'),
        ]
        for pattern, replacement in brand_typo_fixes:
            query_lower = re.sub(pattern, replacement, query_lower, flags=re.IGNORECASE)
        
        # De-duplicate brand names that may have been doubled by overlapping normalization patterns
        # e.g., "samsung samsung s26 ultra" → "samsung s26 ultra"
        # e.g., "samsung galaxy samsung s24" → "samsung s24"
        query_lower = re.sub(r'\bsamsung\s+samsung\b', 'samsung', query_lower, flags=re.IGNORECASE)
        query_lower = re.sub(r'\bsamsung\s+galaxy\s+samsung\b', 'samsung', query_lower, flags=re.IGNORECASE)
        # "galaxy samsung sNN" → "samsung sNN" (galaxy before samsung is redundant)
        query_lower = re.sub(r'\bgalaxy\s+samsung\b', 'samsung', query_lower, flags=re.IGNORECASE)
        # "galaxy sNN" → "samsung sNN" (users mean Samsung Galaxy S-series)
        query_lower = re.sub(r'\bgalaxy\s+(s\d{2})\b', r'samsung \1', query_lower, flags=re.IGNORECASE)
        query_lower = re.sub(r'\bgalaxy\s+galaxy\b', 'galaxy', query_lower, flags=re.IGNORECASE)
        # Clean trailing/extra whitespace from replacements
        query_lower = re.sub(r'\s+', ' ', query_lower).strip()
        
        # =================== VIVO MODEL NORMALIZATION ===================
        # Handle Vivo model variations: "vivox200", "vivo x 200", "vivox 200", "vivo x200 pro" etc.
        # Normalize to proper format: "vivo x200", "vivo x200 pro"
        
        # Pattern: vivo + optional space + model letter + optional space + number + optional suffix
        # Matches: vivox200, vivo x200, vivo x 200, vivox 200, vivox200pro, vivo x200 pro, vivo x 200 pro
        # NOTE: suffix must be preceded by whitespace (not glued to number) to avoid
        # splitting model names like 't4x' into 't4 x'. Single-char suffixes (e, s, x, z)
        # only match when separated by space from the number.
        # Added "elite" suffix for V70 Elite models
        vivo_pattern = r'\b(vivo)\s*([xyvt])\s*(\d+)\s*(pro|plus|lite|ultra|fe|elite|e|s|a|i|x|z)?\b'
        
        def normalize_vivo(match):
            brand = match.group(1)  # vivo
            series = match.group(2)  # x, y, v, t
            number = match.group(3)  # 200, 100, 40, etc.
            suffix = match.group(4) or ""  # pro, plus, lite, etc.
            if suffix:
                # Single-letter suffixes like 'x', 'e', 's' are model variants (t4x, v40e)
                # They should be appended without space, unlike multi-letter suffixes (pro, plus)
                if len(suffix) == 1:
                    return f"{brand} {series}{number}{suffix}"
                return f"{brand} {series}{number} {suffix}"
            return f"{brand} {series}{number}"
        
        query_lower = re.sub(vivo_pattern, normalize_vivo, query_lower, flags=re.IGNORECASE)
        
        # Also handle: "vivo neo", "vivo iqoo" style queries
        # Handle "vivoy" standalone -> "vivo y" etc.
        query_lower = re.sub(r'\bvivo([xyvt])(\s|$)', r'vivo \1\2', query_lower)
        
        # =================== OPPO MODEL NORMALIZATION ===================
        # Handle Oppo model variations: "oppof19", "oppo f 19", "oppoa53" etc.
        # NOTE: suffix must be preceded by whitespace to avoid space consumption
        oppo_pattern = r'\b(oppo)\s*([afkr])\s*(\d+)(?:\s+(pro|plus|lite|s|x|k))?\b'
        
        def normalize_oppo(match):
            brand = match.group(1)  # oppo
            series = match.group(2)  # a, f, k, r
            number = match.group(3)  # 19, 53, etc.
            suffix = match.group(4) or ""
            if suffix:
                return f"{brand} {series}{number} {suffix}"
            return f"{brand} {series}{number}"
        
        query_lower = re.sub(oppo_pattern, normalize_oppo, query_lower, flags=re.IGNORECASE)
        
        # =================== REALME MODEL NORMALIZATION ===================
        # Handle Realme model variations: "realme12", "realme 12 pro", "realmegt" etc.
        # Keep series+number together: "realme p3" → "realme p3", NOT "realme p 3"
        # IMPORTANT: [cp]\d+ must come BEFORE [cp] for correct matching
        realme_pattern = r'\b(realme)\s*([cp]\d+[a-z]?|\d+|gt|narzo|[cp])(?:\s+(pro|plus|neo|master|ultra|x|5g))?\b'
        
        def normalize_realme(match):
            brand = match.group(1)  # realme
            series = match.group(2)  # p3, c73, 12, gt, etc.
            suffix = match.group(3) or ""
            parts = [brand, series]
            if suffix:
                parts.append(suffix)
            return " ".join(parts)
        
        query_lower = re.sub(realme_pattern, normalize_realme, query_lower, flags=re.IGNORECASE)
        
        # =================== SAMSUNG MODEL NORMALIZATION ===================
        # Handle Samsung Galaxy variations: "samsunggalaxys24", "samsung galaxy s 24" etc.
        # NOTE: suffix must be preceded by whitespace to avoid space consumption
        samsung_pattern = r'\b(samsung)\s*(galaxy)?\s*([asmzf])\s*(\d+)(?:\s+(fe|ultra|plus|lite|s))?\b'
        
        def normalize_samsung(match):
            brand = match.group(1)  # samsung
            galaxy = match.group(2) or ""  # galaxy (optional)
            series = match.group(3)  # a, s, m, z, f
            number = match.group(4)  # 24, 54, etc.
            suffix = match.group(5) or ""
            parts = [brand]
            if galaxy:
                parts.append(galaxy)
            parts.append(f"{series}{number}")
            if suffix:
                parts.append(suffix)
            return " ".join(parts)
        
        query_lower = re.sub(samsung_pattern, normalize_samsung, query_lower, flags=re.IGNORECASE)
        
        # =================== REDMI/POCO MODEL NORMALIZATION ===================
        # Handle Redmi variations: "redminote13", "redmi note 13 pro" etc.
        # NOTE: suffix must be preceded by whitespace to avoid space consumption
        redmi_pattern = r'\b(redmi)\s*(note)?\s*(\d+)(?:\s+(pro|plus|a|c|s|i))?\b'
        
        def normalize_redmi(match):
            brand = match.group(1)  # redmi
            note = match.group(2) or ""  # note (optional)
            number = match.group(3)  # 13, 12, etc.
            suffix = match.group(4) or ""
            parts = [brand]
            if note:
                parts.append(note)
            parts.append(number)
            if suffix:
                parts.append(suffix)
            return " ".join(parts)
        
        query_lower = re.sub(redmi_pattern, normalize_redmi, query_lower, flags=re.IGNORECASE)
        
        # Handle Poco: "pocom6", "poco m 6 pro" etc.
        # NOTE: suffix must be preceded by whitespace to avoid space consumption
        poco_pattern = r'\b(poco)\s*([mxfc])\s*(\d+)(?:\s+(pro|plus|gt))?\b'
        
        def normalize_poco(match):
            brand = match.group(1)  # poco
            series = match.group(2)  # m, x, f, c
            number = match.group(3)  # 6, 5, etc.
            suffix = match.group(4) or ""
            if suffix:
                return f"{brand} {series}{number} {suffix}"
            return f"{brand} {series}{number}"
        
        query_lower = re.sub(poco_pattern, normalize_poco, query_lower, flags=re.IGNORECASE)
        
        # =================== ONEPLUS MODEL NORMALIZATION ===================
        # Handle OnePlus variations: "oneplus12", "one plus 12 pro", "oneplus nord ce4" etc.
        # First normalize "one plus" -> "oneplus" to avoid double "plus"
        query_lower = re.sub(r'\bone\s+plus\b', 'oneplus', query_lower, flags=re.IGNORECASE)
        query_lower = re.sub(r'\b1\s*\+', 'oneplus', query_lower, flags=re.IGNORECASE)
        
        # Pattern for OnePlus models including Nord CE with number
        # Handles: oneplus 12, oneplus nord, oneplus nord ce4, oneplus ace 3 pro
        oneplus_pattern = r'\b(oneplus)\s*(nord|ace|\d+)?\s*(ce|buds)?\s*(\d*)\s*(pro|r|t|lite|plus)?\b'
        
        def normalize_oneplus(match):
            brand = "oneplus"
            series = match.group(2) or ""  # nord/ace/number
            subseries = match.group(3) or ""  # ce/buds
            number = match.group(4) or ""
            suffix = match.group(5) or ""
            
            parts = [brand]
            if series:
                parts.append(series)
            if subseries:
                parts.append(subseries)
            if number:
                parts.append(number)
            if suffix:
                parts.append(suffix)
            return " ".join(parts)
        
        query_lower = re.sub(oneplus_pattern, normalize_oneplus, query_lower, flags=re.IGNORECASE)
        
        # =================== MOTOROLA MODEL NORMALIZATION ===================
        # Handle Motorola variations: "motorolag84", "motorola g 84" etc.
        # NOTE: suffix must be preceded by whitespace to avoid consuming the space
        # between model number and unrelated tokens (e.g., 'g96 5g' → 'g965g')
        moto_pattern = r'\b(motorola|moto)\s*([gex])\s*(\d+)(?:\s+(power|plus|stylus|pro|5g))?\b'
        
        def normalize_moto(match):
            brand = "motorola"  # normalize brand
            series = match.group(2)  # g, e, x
            number = match.group(3)  # 84, 73, etc.
            suffix = match.group(4) or ""
            if suffix:
                return f"{brand} {series}{number} {suffix}"
            return f"{brand} {series}{number}"
        
        query_lower = re.sub(moto_pattern, normalize_moto, query_lower, flags=re.IGNORECASE)

        # First, normalize compound words (before splitting)
        # NOTE: These are substring replacements, so only use for COMPOUND words
        # DO NOT add simple typos like "mobil" -> they cause issues (e.g., "mobil" in "mobile")
        # Simple typos are handled by regex patterns in brand_typo_fixes above
        compound_normalizations = {
            "mobilephone": "mobile phone",
            # REMOVED: "mobail", "mobil", "mobale" - these are simple typos handled by regex
            "cellphone": "cell phone",
            "smartphone": "smartphone",
            "smartfone": "smartphone",
            "smartfon": "smartphone",
            "featurephone": "feature phone",
            "touchphone": "touch phone",
            "cameraphone": "camera phone",
            "dualsim": "dual sim",
            "feturphone": "feature phone",
            "feturphon": "feature phone",
            "fetur phon": "feature phone",
            # Geyser/Water heater typo corrections
            "gijar": "geyser",
            "gizer": "geyser",
            "giser": "geyser",
            "gisar": "geyser",
            "geysar": "geyser",
            "gayser": "geyser",
            "geaser": "geyser",
            "geasar": "geyser",
            "geizer": "geyser",
            "geezer": "geyser",
            "gyser": "geyser",
            "gaiser": "geyser",
            "gaisar": "geyser",
            "waterheater": "water heater",
            # Pulsar/Two-wheeler typo corrections
            "pulser": "pulsar",
            "plsar": "pulsar",
            "plser": "pulsar",
            "pulsr": "pulsar",
            "pulsaar": "pulsar",
            "pulsur": "pulsar",
            "pulzar": "pulsar",
            "pulsarr": "pulsar",
            "pulsser": "pulsar",
            "pluser": "pulsar",
            "plusar": "pulsar",
            # Hunter model typos
            "huntr": "hunter",
            "huntar": "hunter",
            "hunteR": "hunter",
            "hunter350": "hunter 350",
            "hunter 350": "hunter 350",
            # Unicorn model typos
            "unicron": "unicorn",
            "unikorn": "unicorn",
            "unocorn": "unicorn",
            "unicrn": "unicorn",
            "unikrn": "unicorn",
            # Other bike model typos
            "apachi": "apache",
            "apche": "apache",
            "aktivia": "activa",
            "actva": "activa",
            "splendour": "splendor",
            "splendr": "splendor",
            "splendar": "splendor",
            "splndr": "splendor",
            "bullat": "bullet",
            "bullt": "bullet",
            "bulet": "bullet",
            "bulat": "bullet",
            # Two-wheeler brand typos
            "hunda": "honda",
            "hnda": "honda",
            "hona": "honda",
            "heero": "hero",
            "hiro": "hero",
            "yemaha": "yamaha",
            "yamha": "yamaha",
            "suzki": "suzuki",
            "suzuky": "suzuki",
            "roal enfield": "royal enfield",
            "royal enfild": "royal enfield",
            "royalenfield": "royal enfield",
            # Four-wheeler brand typos
            "maruti suzuki": "maruti",
            "maruthi": "maruti",
            "maruti suzki": "maruti",
            "hyndai": "hyundai",
            "hundai": "hyundai",
            "hyundia": "hyundai",
            "mahindara": "mahindra",
            "mahendra": "mahindra",
            "tata motors": "tata",
            # Brand+model compound normalizations
            "vivox": "vivo x",
            "vivoy": "vivo y",
            "vivov": "vivo v",
            "vivot": "vivo t",
            "motorola": "motorola",
            "motorolag": "motorola g",
            "motorolaedge": "motorola edge",
            "samsunggalaxy": "samsung galaxy",
            "realme1": "realme 1",
            "realme2": "realme 2",
            "realme3": "realme 3",
            "oppof": "oppo f",
            "oppoa": "oppo a",
            "oppok": "oppo k",
            "redminote": "redmi note",
            "oneplus": "oneplus",
            "iqooneo": "iqoo neo",
            "iqooz": "iqoo z",
            "pocom": "poco m",
            "pocox": "poco x",
            "pocof": "poco f",
            # Apple compound normalizations
            "iphonepro": "iphone pro",
            "iphonepromax": "iphone pro max",
            "iphone15pro": "iphone 15 pro",
            "iphone16pro": "iphone 16 pro",
            # Vehicle compound normalizations
            "twowheeler": "two wheeler",
            "2wheeler": "two wheeler",
            "fourwheeler": "four wheeler",
            "4wheeler": "four wheeler",
            "motorbike": "motor bike",
            "motorcycle": "motor cycle",
            "electricscooter": "electric scooter",
            "electricbike": "electric bike",
            "ebike": "e bike",
            "evscooter": "ev scooter",
            "royalenfield": "royal enfield",
            "scootypep": "scooty pep",
            # Car model compounds
            "wagonr": "wagon r",
            "grandvitara": "grand vitara",
            "evitara": "e vitara",
            "nexonev": "nexon ev",
            "punchev": "punch ev",
            "tiagoev": "tiago ev",
            "tigorev": "tigor ev",
            "cretaelectric": "creta electric",
            "scorpion": "scorpio n",
            "xuv700": "xuv 700",
            "tvs" :" tvs",
            "xuv500": "xuv 500",
            "xuv400": "xuv 400",
            "xuv300": "xuv 300",
            "xuv3xo": "xuv 3xo",
            "be6e": "be 6e",
            "xev9e": "xev 9e",
            "zsev": "zs ev",
            # Sports equipment typo correction
            "sport and equipment": "sports and equipment",
            "sport and equipments": "sports and equipment",
            "sport equipment": "sports equipment",
            "sport equipments": "sports equipment",
            # NOTE: Earphone/Earbud/Headphone typos are handled by regex patterns in brand_typo_fixes
            # Do NOT add them here as substring replacement causes issues (e.g., "earphon" in "earphone")
            
            # ===== REVERSED WORD ORDER NORMALIZATIONS =====
            # Air Conditioner reversed patterns
            "conditioner air": "air conditioner",
            "conditionar air": "air conditioner",
            "conditionr air": "air conditioner",
            "condisioner air": "air conditioner",
            # Washing Machine reversed patterns
            "machine washing": "washing machine",
            "machne washing": "washing machine",
            "machin washing": "washing machine",
            # Door Refrigerator reversed patterns
            "door double refrigerator": "double door refrigerator",
            "door single refrigerator": "single door refrigerator",
            "door side by side": "side by side door",
            # Front/Top Load reversed
            "load front washing": "front load washing",
            "load top washing": "top load washing",
        }
        for compound, normalized in compound_normalizations.items():
            if compound in query_lower:
                query_lower = query_lower.replace(compound, normalized)
        
        words = query_lower.split()
        corrected = []
        
        # Words that should NEVER be corrected (common product terms)
        protected_words = {"phone", "phones", "mobile", "mobiles", "watch", "watches", 
                          "tv", "laptop", "laptops", "tablet", "tablets", "fan", "fans",
                          "bed", "beds", "sofa", "chair", "table", "ac", "fridge", "washing",
                          "bike", "bikes", "car", "cars", "scooter", "scooty", "suv",
                          "wheeler", "wheelers", "wheler", "weeler", "four", "two",
                          # Appliance-related words (prevent fuzzy matching to compound words)
                          "air", "conditioner", "water", "purifier", "heater", "cooler", "machine",
                          # Electric vehicle terms (prevent "electric" -> "electronic" fuzzy match)
                          "electric", "ev", "electrical",
                          # =================== ATTRIBUTE-RELATED WORDS ===================
                          # These are commonly used in attribute searches and should NOT be fuzzy matched
                          # Star rating (1-5 star): "star" should NOT match "sitar" brand
                          "star", "stars",
                          # Tonnage (AC): "ton" should be preserved
                          "ton", "tons", "tonne",
                          # Door type (refrigerator): preserve door-related words
                          "door", "doors", "single", "double", "triple", "side", "french", "multi",
                          # Capacity (washing machine, refrigerator)
                          "kg", "litre", "litres", "liter", "liters",
                          # Screen size (TV)
                          "inch", "inches",
                          # Other common attribute terms
                          "inverter", "split", "window", "front", "top", "load", "smart",
                          "led", "oled", "qled", "lcd", "hd", "uhd", "full"}
        
        for word in words:
            if len(word) <= 2 or word.isdigit():
                corrected.append(word)
                continue
            
            # Protect common product terms from typo correction
            if word in protected_words:
                corrected.append(word)
                continue
            
            # Protect brand names
            if word in self.brand_set:
                corrected.append(word)
                continue
            
            # Try fuzzy match against brands
            brand_match = fuzzproc.extractOne(word, list(self.brand_set), scorer=fuzz.ratio, score_cutoff=85)
            if brand_match:
                corrected.append(brand_match[0])
                continue
            
            corrected.append(word)
        
        return " ".join(corrected)
    
    def detect_category(self, query: str) -> Optional[str]:
        """Detect product category from query with exact match priority"""
        if not query:
            return None
        
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        query_no_space = query_lower.replace(" ", "")

        # Treat standalone 'led' as television (users often type 'LED' to mean LED TVs)
        if query_lower == 'led' or (len(query_words) == 1 and 'led' in query_words):
            return "television"
        
        # print("Corrected Query for Category Detection:", query_lower)
        # print("Query Words Set:", query_words)
        # print("Query No Space:", query_no_space)
        
        # =================== HIGHEST PRIORITY: WASHING MACHINE DETECTION ===================
        # MUST come BEFORE brand detection (realme, motorola, samsung etc.)
        # "realme washing machine", "motorola washing machine" should return washing machine, NOT smartphone
        # _washing_machine_keywords = [
        #     "washing machine", "washingmachine", "washing machines", "washingmachines",
        #     "washer", "washers", "clothes washer", "laundry machine",
        #     "front load", "frontload", "top load", "topload", "semi automatic",
        #     "semi-automatic", "fully automatic", "fully-automatic"
        # ]
        # for wmkw in _washing_machine_keywords:
        #     if wmkw in query_lower:
        #         return "washing machine"
        
        if "dish" not in query_lower:
           _washing_machine_keywords = [
               "washing machine", "washingmachine", "washing machines", "washingmachines",
               "washer", "washers", "clothes washer", "laundry machine",
               "front load", "frontload", "top load", "topload", "semi automatic",
               "semi-automatic", "fully automatic", "fully-automatic"
           ]
           for wmkw in _washing_machine_keywords:
               if wmkw in query_lower:
                   return "washing machine"

        # Fuzzy match for typos
        _wm_fuzzy_targets = ["washing", "machine", "washer"]
        _wm_found_washing = False
        _wm_found_machine = False
        for word in query_words:
            if len(word) >= 5:
                # Using rapidfuzz (already imported as fuzz at top of file)
                if fuzz.ratio(word, "washing") >= 80:
                    _wm_found_washing = True
                if fuzz.ratio(word, "machine") >= 80:
                    _wm_found_machine = True
        if _wm_found_washing and _wm_found_machine:
            return "washing machine"
            
        # =================== HIGHEST PRIORITY: ELECTRIC TWO-WHEELER DETECTION ===================
        # MUST come BEFORE electric car detection to handle "electric scooter", "electric bike" correctly
        ELECTRIC_TWO_WHEELER_KEYWORDS = [
            "electric scooter", "electric scooters", "e-scooter", "escooter", "e scooter",
            "electric bike", "electric bikes", "e-bike", "ebike", "e bike",
            "electric two wheeler", "electric 2 wheeler", "ev scooter", "ev bike",
            "electric moped", "electric motorcycle"
        ]
        for keyword in ELECTRIC_TWO_WHEELER_KEYWORDS:
            if keyword in query_lower:
                return "two wheeler"
        
        # =================== HIGHEST PRIORITY: ELECTRIC CAR DETECTION ===================
        # MUST come FIRST before any two-wheeler detection to handle "electric car" correctly
        # This prevents "ct" in "electric" from matching Bajaj CT model
        ELECTRIC_CAR_KEYWORDS = [
            "electric car", "electric cars", "ev car", "ev cars", "e-car", "ecar",
            "electric suv", "ev suv", "electric vehicle", "electric hatchback", "electric sedan",
            "creta electric", "nexon ev", "punch ev", "tiago ev", "tigor ev", "curvv ev",
            "zs ev", "comet ev", "windsor ev", "ioniq", "kona ev", "ev6", "ev9"
        ]
        for keyword in ELECTRIC_CAR_KEYWORDS:
            if keyword in query_lower:
                return "car"
        
        # =================== HIGH PRIORITY: FOUR-WHEELER/CAR DETECTION ===================             
                
        # TVS brand keywords - TVS is primarily a two-wheeler brand
        TVS_KEYWORDS = [
            "tvs", "tvs motor", "tvs motors", "tvs sooty", "tvs scooter"
        ]
        # Avoid matching 'tvs' when user means televisions (e.g., "led tvs", "sony tvs", "55 inch tvs")
        # Includes TV context indicators, TV brands, and size indicators
        television_word_indicators = {
            "tv", "television", "televisions", "led", "lcd", "oled", "qled", "smart", "uhd", "hdr", "4k", "8k",
            "inch", "inches",  # Size indicators
            # TV brands - if these brands are mentioned with "tvs", it's likely televisions
            "sony", "lg", "samsung", "philips", "toshiba", "tcl", "vu", "hisense", "panasonic", "mi", "xiaomi",
            "oneplus", "realme", "motorola", "acer", "iffalcon", "kodak", "lloyd", "haier", "thomson", "bpl"
        }
        # TV-related phrases that indicate televisions, not TVS brand
        tv_phrases = ["smart tv", "led tv", "4k tv", "8k tv", "inch tv", "inch led", "oled tv", "qled tv",
                      "ultra hd", "android tv", "google tv", "fire tv"]
        for keyword in TVS_KEYWORDS:
            if keyword in query_lower:
                # If query contains explicit TV indicators as separate words or TV phrases,
                # prefer television detection instead of TVS brand (two-wheeler).
                if any(ind in query_words for ind in television_word_indicators) or \
                   any(phrase in query_lower for phrase in tv_phrases):
                    continue
                # Also skip if "tvs" appears after a number (likely screen size like "55 tvs")
                if re.search(r'\b\d{2,3}\s*tvs?\b', query_lower):
                    continue
                return "two wheeler"
        
        # Priority 0: Check for scooter models
        # This handles "activa", "scooty" etc. for two-wheeler detection
        SCOOTER_MODELS = [
            "activa", "honda activa", "activa 5g", "activa 6g", "honda activa 6g","honda activa 5g",
            "honda scooter", "scooter", "scooty", "scooty pep", "scootyteo","scooty zesty",
        ]
        for keyword in SCOOTER_MODELS:
            if keyword in query_lower:
                return "two wheeler"
            
            
        # 
        # Priority 0.5: Check for car models (before two-wheeler check)
        car_models = [
            # Maruti
            "swift", "dzire", "baleno", "brezza", "ertiga", "xl6", "celerio",
            "alto", "wagonr", "wagon r", "ignis", "ciaz", "spresso", "fronx", "jimny",
            "invicto", "grand vitara", "e vitara", "evitara",
            # Hyundai
            "creta", "venue", "i10", "i20", "verna", "alcazar", "tucson", "exter",
            "aura", "xcent", "santro", "ioniq", "kona",
            # Tata
            "nexon", "punch", "harrier", "safari", "altroz", "tiago", "tigor", "curvv",
            # Mahindra
            "xuv700", "xuv 700", "xuv500", "xuv 500", "xuv400", "xuv 400",
            "xuv300", "xuv 300", "thar", "scorpio", "bolero", "xuv3xo", "xuv 3xo",
            "be 6e", "be6e", "xev 9e", "xev9e",
            # Kia
            "seltos", "sonet", "carens", "carnival", "ev6", "ev9",
            # MG
            "hector", "astor", "gloster", "zs ev", "comet", "windsor",
            # VW
            "polo", "vento", "taigun", "virtus", "tiguan",
            # Toyota
            "fortuner", "innova", "crysta", "hycross", "legender", "urban cruiser",
            # Honda Cars
            "city", "amaze", "elevate", "wr-v", "wrv"
        ]
        for model in car_models:
            if model in query_lower:
                return "car"
        
        # =================== HIGH PRIORITY: TV CABINET/STAND (FURNITURE) ===================
        # MUST come BEFORE television detection to handle "tv cabinet", "tv stand" etc.
        # These are furniture items, NOT televisions
        tv_furniture_keywords = [
            "tv cabinet", "tv cabinate", "tv stand", "tv unit", "tv table",
            "television cabinet", "television stand", "television unit",
            "table tv cabinet", "table tv cabinate"
        ]
        for keyword in tv_furniture_keywords:
            if keyword in query_lower:
                return "furniture"
        
        # =================== HIGHEST PRIORITY: GAMING MONITOR DETECTION ===================
        # "gaming monitor", "gaming monitors" should return monitor category NOT gaming consoles
        # MUST come BEFORE gaming console detection
        _gaming_monitor_keywords = ["gaming monitor", "gaming monitors", "monitor gaming"]
        for gmk in _gaming_monitor_keywords:
            if gmk in query_lower:
                return "monitor"
        
        # =================== HIGHEST PRIORITY: PS5/PLAYSTATION/GAMING CONSOLE DETECTION ===================
        # "ps5", "ps 5", "playstation", "play station", "xbox" should return Gaming Consoles
        _console_keywords = [
            "ps5", "ps 5", "ps4", "ps 4", "playstation", "play station", 
            "playstation 5", "playstation 4", "xbox", "nintendo", "switch",
            "gaming console", "game console"
        ]
        for ck in _console_keywords:
            if ck in query_lower:
                return "gaming and accessories"  # Gaming consoles category
        
        # =================== HIGHEST PRIORITY: LAPTOP/COMPUTER DETECTION ===================
        # MUST come BEFORE two-wheeler detection to prevent "computer" fuzzy matching "scouter"
        _laptop_early_keywords = ["computer", "computers", "laptop", "laptops", "notebook", "desktop", "pc"]
        for kw in _laptop_early_keywords:
            if kw in query_words:
                return "laptops"
        
        # =================== HIGH PRIORITY: BRAVIA TV DETECTION ===================
        # "Bravia" is Sony's TV brand - force television category
        # MUST come BEFORE generic brand detection as Bravia exists in camera accessories too
        if "bravia" in query_lower:
            return "television"
        
        # =================== HIGH PRIORITY: DEEP FREEZER DETECTION ===================
        # "Deep freezer" should return refrigerators category (deep freezers are in refrigerators)
        if "deep freezer" in query_lower or "deepfreezer" in query_lower or "deep freeze" in query_lower:
            return "refrigerators"
        
        # =================== HIGH PRIORITY: QLED/OLED TV DETECTION ===================
        # QLED and OLED are TV display technologies - force television category
        if "qled" in query_lower or "oled" in query_lower:
            return "television"
        
        # =================== HIGH PRIORITY: AIR COOLER DETECTION (BRAND + COOLER) ===================
        # "Bajaj coolers" should return air coolers, not two-wheelers
        # Check if any brand (including two-wheeler brands) is combined with "cooler/coolers"
        _cooler_keywords = ["cooler", "coolers", "air cooler", "air coolers", "desert cooler"]
        if any(ck in query_lower for ck in _cooler_keywords):
            return "air cooler"
        
        # =================== HIGH PRIORITY: RO / WATER PURIFIER ===================
        # MUST come BEFORE other category detection to handle "ro", "ro water", "water ro"
        ro_keywords = ["ro water", "water ro", "ro purifier", "ro filter", "water purifier",
                       "waterpurifier", "aquaguard", "kent ro", "pureit", "livpure"]
        for keyword in ro_keywords:
            if keyword in query_lower:
                return "kitchen appliances"  # RO/Water purifiers are in kitchen appliances
        
        # Also check for standalone "ro" (must be exact word match)
        if "ro" in query_words and len(query_words) <= 3:  # "ro", "ro water", "water ro"
            return "kitchen appliances"
        
        # =================== HIGH PRIORITY: AIR CONDITIONER DETECTION ===================
        # MUST come BEFORE inverter detection to handle "inverter ac", "inverter acs" correctly
        # These are air conditioners, not inverter batteries!
        # AC brands that should trigger air conditioner category
        AC_BRAND_NAMES = {"carrier", "daikin", "voltas", "blue star", "bluestar", "hitachi", 
                         "lloyd", "onida", "panasonic", "mitsubishi", "haier", "godrej",
                         "ifb", "whirlpool", "lg", "samsung", "vestar", "croma", "toshiba"}
        # AC keywords (short forms, typos)
        AC_KEYWORDS = {"ac", "acs", "a.c", "a.c.", "aircon", "airconditioner", "airconditioners",
                      "air conditioner", "air conditioners", "split ac", "window ac", "inverter ac",
                      "non inverter ac", "portable ac", "cassette ac", "ducted ac"}
        
        # Check if query contains AC keyword
        has_ac_keyword = any(kw in query_words for kw in {"ac", "acs"}) or \
                        any(phrase in query_lower for phrase in ["air conditioner", "airconditioner", 
                            "split ac", "window ac", "inverter ac", "non inverter", "inverter acs",
                            "non inverter acs", "ducted ac", "ducted aircon"])
        
        # Check if query has AC brand + "ac" pattern (e.g., "carrier ac", "daikin ac")
        has_ac_brand_with_ac = any(brand in query_lower for brand in AC_BRAND_NAMES) and \
                              ("ac" in query_words or "acs" in query_words)
        
        if has_ac_keyword or has_ac_brand_with_ac:
            return "air conditioner"
            
        television_keywords = [
            "television", "tv", "led tv", "smart tv", "4k tv", "8k tv",
            "lcd tv", "oled tv", "qled tv", "led"
        ]
        for keyword in television_keywords:
            if keyword in query_lower or keyword in query_words:
                return "television"

        # =================== SAMSUNG S-SERIES / GALAXY SMARTPHONE DETECTION ===================
        # Samsung S21-S39 (with optional ultra/plus/fe suffix) are Galaxy phones
        # Matches: "samsung s26", "samsung s26 ultra", "samsung galaxy s25", "s24 ultra"
        if re.search(r'\bsamsung\s+s\d{2}\b', query_lower) or \
           re.search(r'\bsamsung\s+galaxy\s+s\d{2}\b', query_lower) or \
           re.search(r'\bgalaxy\s+s\d{2}\b', query_lower) or \
           re.search(r'\bs(2[1-9])\s*(ultra|plus|fe|lite)?\b', query_lower):
            return "smartphone"

        # Samsung A-series / M-series / F-series are also smartphones
        if re.search(r'\bsamsung\s+[amf]\d{2}\b', query_lower) or \
           re.search(r'\bgalaxy\s+[amf]\d{2}\b', query_lower):
            return "smartphone"

        # =================== AUDIO DEVICE OVERRIDE ===================
        # If query contains audio DEVICE keywords (earbuds/earphone/headphone/neckband/airdopes/tws),
        # force "audio video" BEFORE phone brand check. This ensures "realme earbuds", "oppo earphones",
        # "vivo neckband" etc. show audio products instead of phones.
        # INCLUDES all common typos/misspellings so "boat erphn", "boat earfon" etc. also work.
        _audio_device_keywords = {
            # Earbuds + typos
            "earbuds", "earbud", "ear bud", "ear buds", "earbudz", "earbudss",
            "erbud", "erbuds", "airbud", "airbuds", "earbds", "earbd",
            # Earphone + typos
            "earphone", "earphones", "ear phone", "ear phones",
            "earphn", "earphon", "earfon", "earfone", "earpone", "erphn",
            "erpone", "erphone", "erfone", "erfon", "earphne", "eaphone", "earpho",
            "raphon", "raphne", "raphone", "raphn",
            "airphon", "airphone", "airphn", "airfon", "airfone",
            # Headphone + typos
            "headphone", "headphones", "headfon", "headfone",
            "hedphone", "headphn", "haedphone", "headphne", "headphon",
            "headset", "headsets",
            # Neckband + typos
            "neckband", "neckbands", "neck band",
            "neckbnd", "nekband", "neckbad", "nckband", "necband",
            # Airdopes + typos
            "airdopes", "airdope",
            # TWS / wireless
            "tws", "truly wireless", "true wireless",
            "wireless earbuds", "wireless earphones",
            "bluetooth earbuds", "bluetooth earphones", "bt earphones",
            # AirPods
            "airpods", "airpod", "air pods", "air pod", "airpodz", "airpodss",
            "arpods", "arpod", "aiepods", "airpds", "aripods", "aripod",
        }
        for akw in _audio_device_keywords:
            if akw in query_lower:
                return "audio video"

        # =================== HIGH PRIORITY: TABLET/PAD DETECTION ===================
        # MUST come BEFORE smartphone brand detection to handle "redmi pad", "oneplus pad", etc.
        # These are tablets, NOT smartphones - "pad" should force tablets category
        # NOTE: "tab" must be checked as word boundary to avoid matching "table" (furniture)
        _tablet_brand_keywords = [
            # Brand + pad/tab combinations (these can be substring matched)
            "redmi pad", "oneplus pad", "samsung tab", "galaxy tab",
            "realme pad", "lenovo tab", "xiaomi pad", "mi pad",
            "oppo pad", "vivo pad", "honor pad",
            # Alternative spellings
            "redmipad", "onepluspad", "samsungtab", "galaxytab",
            "realmepad", "lenovotab", "xiaomipad", "mipad",
        ]
        # Check for "pad", "tablet", or "tab" as STANDALONE WORDS (not substrings)
        # This prevents "dining table" from matching "tab"
        if "pad" in query_words or "tablet" in query_words or "tab" in query_words:
            return "tablets"
        # Check for brand + pad/tab combinations (substring matching is OK for these)
        for tkw in _tablet_brand_keywords:
            if tkw in query_lower:
                return "tablets"

        # =================== HIGH PRIORITY: GAMING LAPTOP DETECTION ===================
        # MUST come BEFORE generic laptop detection to properly categorize gaming laptops
        _gaming_laptop_keywords = [
            # Generic gaming laptop queries
            "gaming laptop", "gaming laptops", "gaming notebook", "gaming pc",
            # Lenovo Legion (gaming)
            "lenovo legion", "legion", "legion 5", "legion pro", "legion slim",
            # Lenovo LOQ (gaming)
            "lenovo loq", "loq",
            # HP OMEN (gaming)
            "hp omen", "omen", "omen 15", "omen 16", "omen 17",
            # HP Victus (gaming)
            "hp victus", "victus",
            # MSI Gaming
            "msi gaming", "msi raider", "msi stealth", "msi katana",
            # ASUS ROG / TUF
            "asus rog", "rog strix", "rog zephyrus", "rog flow",
            "asus tuf", "tuf gaming",
            # Dell G Series / Alienware
            "alienware", "dell g15", "dell g16",
            # Acer Nitro / Predator
            "acer nitro", "acer predator", "nitro 5", "predator helios",
        ]
        for gkw in _gaming_laptop_keywords:
            if gkw in query_lower:
                return "laptops"

        # =================== HIGH PRIORITY: SAMSUNG GALAXY BOOK DETECTION ===================
        # "Samsung Galaxy Book" should force laptops category, NOT smartphones
        # Must come BEFORE smartphone brand detection to prevent Galaxy Book 4 showing Galaxy A23
        if "galaxy book" in query_lower or "galaxybook" in query_lower:
            return "laptops"

        # =================== HIGH PRIORITY: DESKTOP COMPUTER DETECTION ===================
        # Handle "omnidesk", "ideacentre" etc. before generic laptop detection
        _desktop_keywords = [
            "hp omnidesk", "omnidesk", "omnibook",
            "lenovo ideacentre", "ideacentre", "thinkcentre",
            "dell optiplex", "optiplex", "dell desktop",
            "hp desktop", "lenovo desktop", "asus desktop",
            "all in one pc", "all in one desktop", "aio desktop", "aio pc",
        ]
        for dkw in _desktop_keywords:
            if dkw in query_lower:
                return "desktop"

        # =================== HIGH PRIORITY: SURFACE LAPTOP DETECTION ===================
        # Microsoft Surface products are laptops/tablets
        _surface_keywords = [
            "microsoft surface", "surface pro", "surface laptop", "surface book",
            "surface go", "surface studio", "surfacepro", "surfacelaptop",
        ]
        for skw in _surface_keywords:
            if skw in query_lower:
                return "laptops"

        mobile_keyword = [
            "motorola", "motrola", "moto g", "moto g series", "moto e", "moto e series",
            # Infinix brand - primarily smartphone brand, NOT TV
            "infinix", "infnix", "infinx", "infinix hot", "infinix note", "infinix zero",
            "infinix smart", "infinix gt",
            # Other phone brands that should force smartphone category
            "oppo", "realme", "redmi", "poco", "iqoo", "tecno", "lava", "itel",
            "nothing phone", "google pixel",
            # Google Pixel - "pixel" alone or with model number should force smartphone
            "pixel", "pixel 6", "pixel 7", "pixel 8", "pixel 9", "pixel pro", "pixel fold",
            "pixle", "pixxel",  # typos
        ]
        # Brand keywords that need word boundary matching to avoid false positives
        # e.g., "vivo" should match "vivo phone" but NOT "vivobook" (laptop)
        mobile_keyword_exact = [
            "vivo", "vevo", "viovo", "vivo x", "vivo y", "vivo v", "vivo t",
        ]
        for keyword in mobile_keyword:
            if keyword in query_lower or keyword in query_words:
                return "smartphone"
        # Check brands that need word boundary matching
        for keyword in mobile_keyword_exact:
            # Use word boundary regex to avoid substring matches (e.g., "vivo" in "vivobook")
            if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                return "smartphone"  # FIXED: was "smarphone" (typo)
        
        # =================== HIGH PRIORITY: AUDIO VIDEO ===================
        # Priority 0.85: Audio/Earphones/Earbuds - MUST come before smartphone detection
        # This ensures "earphones", "earbuds" return audio products, not smartphones
        for keyword in AUDIO_VIDEO_KEYWORDS:
            if keyword in query_lower or keyword in query_words:
                return "audio video"
        
        # =================== HIGH PRIORITY: AIR COOLER BRANDS ===================
        # Priority 0.855: Symphony, Kenstar are air cooler brands
        # "symphony" → air cooler (NOT smartphone)
        # "symphony air cooler", "symphony cooler" → air cooler
        AIR_COOLER_BRAND_KEYWORDS = ["symphony", "kenstar"]
        for brand in AIR_COOLER_BRAND_KEYWORDS:
            if brand in query_lower:
                return "air cooler"
        
        # =================== HIGH PRIORITY: AIR FRYER ===================
        # Priority 0.86: Air Fryer - specific detection to avoid matching Dyson Airwrap
        for keyword in AIR_FRYER_KEYWORDS:
            if keyword in query_lower:
                return "home appliance"  # Air fryers are in home appliance category
        
        # =================== HIGH PRIORITY: MIXER GRINDER ===================
        # Priority 0.865: Mixer grinder detection - prioritize category over brand
        # If "mixer grinder" is in query, return kitchen appliances category
        for keyword in MIXER_GRINDER_KEYWORDS:
            if keyword in query_lower:
                return "kitchen appliances"  # Mixer grinders are in kitchen appliances
        
        # =================== HIGH PRIORITY: SOLAR HEATER (WATER HEATER) ===================
        # Priority 0.865: Solar heater detection - MUST come BEFORE solar panels
        # "solar heater", "solar water heater", "solar geyser" → water heaters, NOT inverters
        for keyword in SOLAR_HEATER_KEYWORDS:
            if keyword in query_lower:
                return "water heater"
            
        
        
        # =================== HIGH PRIORITY: SOLAR PANELS ===================
        # Priority 0.866: Solar detection - MUST come BEFORE two-wheeler to avoid 
        # "solar" matching bikes with "Solar Red" color
        # "solar" alone or any solar variation should show solar panels/inverters
        for keyword in SOLAR_KEYWORDS:
            if keyword in query_lower or keyword in query_words:
                return "inverter"  # Solar panels are in inverter category
        
        # =================== HIGH PRIORITY: TWO-WHEELER BRAND NAMES ===================
        # Priority 0.8665: If query contains two-wheeler brand names (KTM, Royal Enfield, etc.)
        # force two-wheeler category - these brands are primarily bike/motorcycle brands
        TWO_WHEELER_BRAND_NAMES = {"ktm", "royal enfield", "royalenfield", "enfield", "kawasaki", 
                                   "ducati", "harley", "harley davidson", "harleydavidson",
                                   "benelli", "aprilia", "mv agusta", "triumph", "bmw motorrad",
                                   "jawa", "yezdi", "bajaj", "tvs", "hero", "honda", "yamaha", "suzuki"}
        for brand in TWO_WHEELER_BRAND_NAMES:
            if brand in query_words or brand in query_lower:
                # But not if it's clearly about something else (like honda car)
                if "car" not in query_words and "cars" not in query_words and "suv" not in query_words:
                    return "two wheeler"
        
        # =================== HIGH PRIORITY: TWO-WHEELER MODEL NAMES ===================
        # Priority 0.867: If query contains known two-wheeler model names (pulsar, activa, etc.)
        # force two-wheeler category regardless of brand availability
        # IMPORTANT: Only match whole words (not substrings)
        for model in TWO_WHEELER_MODELS:
            if model in query_words:  # Only exact word match, not substring
                return "two wheeler"  # Force two wheeler category for known bike models (no hyphen)
        
        # =================== HIGH PRIORITY: GEYSER / WATER HEATER ===================
        # Priority 0.87: Geyser/Water heater detection (including typos like "gijar")
        # Also detect standalone "heater" - most heater queries mean water heater
        for keyword in GEYSER_KEYWORDS:
            if keyword in query_lower:
                return "water heater and geysers"  # Geysers are in "water heater and geysers" category
        
        # Standalone "heater" detection - users searching "heater" usually want water heaters
        if "heater" in query_words or "heaters" in query_words:
            return "water heater and geysers"
        
        # Priority 0.9: Check for vacuum cleaner/mop (BEFORE two-wheeler to avoid "mop" matching "moped")
        vacuum_mop_keywords = ["vacuum", "vacuum cleaner", "vacuumcleaner", "vaccum", "robot vacuum",
                              "floor mop", "robot mop", "vacuum mop", "mopping", "floor cleaner"]
        # Check exact word match for "mop" to avoid false positives
        if "mop" in query_words:  # exact word match only
            return "vacuum cleaner"
        for keyword in vacuum_mop_keywords:
            if keyword in query_lower:
                return "vacuum cleaner"
        
        # Priority 0.905: FAN detection (BEFORE furniture to handle "table fan", "ceiling fan" etc.)
        fan_keywords = ["fan", "fans", "ceiling fan", "table fan", "pedestal fan", "exhaust fan", 
                       "wall fan", "tower fan", "pankha", "punkha"]
        for keyword in fan_keywords:
            if keyword in query_lower:
                return "home appliance"
        
        # Priority 0.91: Tractor detection (BEFORE two-wheeler)
        tractor_keywords = ["tractor", "tractors", "tracktor", "trator", "traktor"]
        for keyword in tractor_keywords:
            if keyword in query_lower:
                return "tractor"
        
        # Priority 0.92: Electric kettle detection
        kettle_keywords = ["kettle", "kettles", "electric kettle", "water kettle", "tea kettle", "ketle", "ketl"]
        for keyword in kettle_keywords:
            if keyword in query_lower:
                return "electric kettle"
        
        # Priority 0.93: Water purifier detection (BEFORE air purifier to avoid confusion)
        water_purifier_keywords = ["water purifier", "waterpurifier", "water filter", "waterfilter", 
                                   "ro purifier", "ro filter", "water purifir", "aquaguard", "kent ro",
                                   "pureit", "livpure", "eureka forbes"]
        for keyword in water_purifier_keywords:
            if keyword in query_lower:
                return "kitchen appliances"  # water purifiers are in kitchen appliances category
        
        # Priority 0.94: Sports and fitness equipment
        sports_keywords = ["sports", "fitness", "gym", "exercise", "workout", "treadmill", "dumbbell",
                          "sports equipment", "fitness equipment", "gym equipment", "exercise bike",
                          "cross trainer", "elliptical", "rowing machine", "weight bench"]
        for keyword in sports_keywords:
            if keyword in query_lower:
                return "sports fitness equipment"
        
        # Priority 0.95: Farm and gardening equipment
        farm_keywords = ["farm", "farming", "garden", "gardening", "agriculture", "lawn mower",
                        "grass cutter", "sprayer", "pump set", "farm equipment", "garden equipment"]
        for keyword in farm_keywords:
            if keyword in query_lower:
                return "farm and gardening equipment"
        
        # Priority 0.96: Travel and accessories
        travel_keywords = ["travel", "luggage", "suitcase", "trolley bag", "travel bag", "backpack",
                          "travel accessories", "luggage bag", "cabin bag", "duffle bag", "travel kit"]
        for keyword in travel_keywords:
            if keyword in query_lower:
                return "travel and accessories"
        
        # Priority 1: Check for two-wheeler specific keywords
        # IMPORTANT: For short keywords (<=3 chars like "ct", "fz"), use word boundary matching
        # to avoid false positives (e.g., "electric" matching "ct")
        for keyword in TWO_WHEELER_KEYWORDS:
            if len(keyword) <= 3:
                # Short keywords: require exact word match
                if keyword in query_words:
                    return "two wheeler"
            else:
                # Longer keywords: can use substring matching
                if keyword in query_words or keyword in query_lower:
                    return "two wheeler"
        
        # Priority 1.5: Fuzzy match for scooter/scooty/bike typos
        # IMPORTANT: Exclude attribute-related words AND common non-vehicle words from fuzzy matching
        # e.g., "star" should NOT match "skutar" (scooter typo) - "star" is for energy rating
        # e.g., "computer" should NOT match "scouter" (80% fuzzy) - computer is laptop/desktop
        CATEGORY_FUZZY_EXCLUDED_WORDS = {
            "star", "stars",  # Energy rating attribute (5 star ac, 3 star fridge)
            "ton", "tons", "tonne",  # Capacity attribute (1.5 ton ac)
            "door", "doors",  # Door type attribute (double door fridge)
            "single", "double", "triple", "side", "french", "multi",  # Door variations
            "kg", "litre", "litres", "liter", "liters",  # Capacity units
            "inch", "inches",  # Screen size
            "watt", "watts",  # Power units
            "volt", "volts",  # Voltage
            # FIX: Exclude computer/laptop related words to avoid "computer" matching "scouter"
            "computer", "computers", "laptop", "laptops", "desktop", "desktops",
            "notebook", "monitor", "monitors", "printer", "printers",
        }
        scooter_typos = ["scooter", "scooty", "scootr", "scouter", "scoter", "skutar", "scootie", "scutty"]
        bike_typos = ["bike", "byke", "motorbike", "motorcycle"]
        for word in query_words:
            if len(word) >= 4 and word not in CATEGORY_FUZZY_EXCLUDED_WORDS:
                for typo in scooter_typos + bike_typos:
                    if fuzz.ratio(word, typo) >= 80:
                        return "two wheeler"
        
        # Priority 2: Remaining four-wheeler keywords (models etc)
        # FIX: Use word boundary matching for single-word keywords to prevent
        # substring matches like "car" matching in "carrier"
        for keyword in FOUR_WHEELER_KEYWORDS:
            if ' ' in keyword:
                # Multi-word: can use substring match
                if keyword in query_lower:
                    return "car"
            else:
                # Single-word: must be exact word match
                if keyword in query_words:
                    return "car"
        
        # Priority 2.5: Fuzzy match for car typos
        car_typos = ["car", "cars", "suv", "sedan", "hatchback"]
        for word in query_words:
            if len(word) >= 3:
                for typo in car_typos:
                    if fuzz.ratio(word, typo) >= 85:
                        return "car"
        
        # Priority 2.6: Fuzzy match for "wheeler" variations (ambiguous - default to car)
        # "wheeler", "wheelers", "wheler", "weeler" → car
        wheeler_typos = ["wheeler", "wheelers", "wheler", "weeler", "wheleer", "wheelar"]
        for word in query_words:
            if len(word) >= 6:
                for typo in wheeler_typos:
                    if fuzz.ratio(word, typo) >= 80:
                        return "car"  # Default ambiguous "wheeler" to car (four-wheeler)
        
        # Priority 0.2: Check for laptop/computer keywords BEFORE phone keywords
        for keyword in LAPTOP_KEYWORDS:
            if keyword in query_words or re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                return "laptops"
        
        # Priority 0.25: Check for kitchen appliance keywords
        for keyword in KITCHEN_KEYWORDS:
            if keyword in query_words or re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                return "kitchen appliances"
        
        # Priority 0.255: Check for DISHWASHER BEFORE washing machine
        # This prevents "dishwasher" from matching "washer" substring
        for keyword in DISHWASHER_KEYWORDS:
            if keyword in query_lower or keyword in query_words:
                return "dishwasher"
        # Fuzzy match for dishwasher typos
        for word in query_words:
            if len(word) >= 8:
                if fuzz.ratio(word, "dishwasher") >= 80:
                    return "dishwasher"
        
        # Priority 0.26: Check for washing machine keywords (including typos)
        # IMPORTANT: Skip if "dish" is in query to avoid matching "dishwasher" → "washer"
        if "dish" not in query_lower:
            for keyword in WASHING_MACHINE_KEYWORDS:
                if keyword in query_lower or keyword in query_words:
                    return "washing machine"
        # Fuzzy match for washing machine typos
        for word in query_words:
            if len(word) >= 6:
                if fuzz.ratio(word, "washing") >= 75 or fuzz.ratio(word, "washer") >= 80:
                    return "washing machine"
        # Check compound without space
        if "washingmachine" in query_no_space or "washingmachin" in query_no_space:
            return "washing machine"
        
        # Priority 0.3: Check for watch/wearable keywords (including typos)
        for keyword in WATCH_KEYWORDS:
            if keyword in query_lower or keyword in query_words:
                return "watch and wearable"
        # Fuzzy match for watch typos
        for word in query_words:
            if len(word) >= 3:
                watch_score = fuzz.ratio(word, "watch")
                watches_score = fuzz.ratio(word, "watches")
                if watch_score >= 75 or watches_score >= 75:
                    return "watch and wearable"
        
        # Priority 0.45: Check for OnePlus keywords (brand detection -> smartphone)
        for keyword in ONEPLUS_KEYWORDS:
            if keyword in query_lower:
                return "smartphone"
        # Fuzzy match for OnePlus variations
        for word in query_words:
            if len(word) >= 4:
                oneplus_score = fuzz.ratio(word, "oneplus")
                if oneplus_score >= 80:
                    return "smartphone"
        
        # Priority 0.46: Check for Realme keywords (brand detection -> smartphone)
        for keyword in REALME_KEYWORDS:
            if keyword in query_lower:
                return "smartphone"
        # Fuzzy match for Realme variations
        for word in query_words:
            if len(word) >= 4:
                realme_score = fuzz.ratio(word, "realme")
                if realme_score >= 75:
                    return "smartphone"
        
        # Priority 0.5: Check for mobile phone specific keywords (use word boundary for short ones)
        for keyword in MOBILE_PHONE_KEYWORDS:
            if len(keyword) <= 3:
                # Short keywords need word boundary to avoid false matches like "ce" in "acer"
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    return "smartphone"
            else:
                if keyword in query_lower:
                    return "smartphone"
        
        # Priority 0.6: "phone" or "phones" query -> smartphone category (all brands)
        if re.search(r'\b(phone|phones|mobile|mobiles)\b', query_lower):
            return "smartphone"
        
        # Priority 1: Exact word match for category names
        for cat_key, canonical in CATEGORY_CANONICAL.items():
            cat_words = set(cat_key.lower().split())
            # Check if all category words are present in query
            if cat_words.issubset(query_words):
                return canonical
            # Check for exact substring match (with word boundary)
            if re.search(r'\b' + re.escape(cat_key.lower()) + r'\b', query_lower):
                return canonical
        
        # Priority 2: Check synonym matches (exact word boundary)
        for cat_key, synonyms in self.synonym_map.items():
            for syn in synonyms:
                syn_lower = syn.lower()
                if syn_lower in query_words or re.search(r'\b' + re.escape(syn_lower) + r'\b', query_lower):
                    if cat_key in CATEGORY_CANONICAL:
                        return CATEGORY_CANONICAL[cat_key]
        
        # Priority 3: Fuzzy match on full category names only (high threshold)
        full_categories = [k for k in CATEGORY_CANONICAL.keys() if len(k) > 3]
        match = fuzzproc.extractOne(query_lower, full_categories, scorer=fuzz.token_set_ratio, score_cutoff=85)
        if match:
            return CATEGORY_CANONICAL[match[0]]
        
        # =================== PRIORITY 4: UNIVERSAL FUZZY FALLBACK ===================
        # Last resort: Use character-level edit distance matching for severe typos
        # This catches cases like "wheler" → "wheeler" → "car"
        # Only triggers if all other methods fail
        fuzzy_result = fuzzy_correct_query(query_lower, min_similarity=0.65)
        
        if fuzzy_result.get("detected_category"):
            logger.info(f"Fuzzy fallback detected category: '{query}' → '{fuzzy_result['detected_category']}' "
                       f"(corrections: {fuzzy_result.get('corrections', [])})")
            return fuzzy_result["detected_category"]
        
        # Try individual words with fuzzy matching
        for word in query_words:
            if len(word) >= 4:  # Only try fuzzy on words with 4+ chars
                category = get_category_from_fuzzy(word, threshold=0.65)
                if category:
                    logger.info(f"Fuzzy word match: '{word}' → category '{category}'")
                    return category
        
        return None
    
    def detect_brand(self, query: str) -> Optional[str]:
        """Detect brand from query - excludes product model names"""
        if not query:
            return None
        
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        # Priority 0: Check OnePlus keywords first (handles "one plus", "1+", typos)
        for keyword in ONEPLUS_KEYWORDS:
            if keyword in query_lower:
                return "oneplus"
        
        # Priority 0.3: Check Google Pixel keywords BEFORE Realme (handles "pixel", "pixle", typos)
        # This ensures "pixel 8", "pixle 9 pro" etc. are detected as Google brand
        for keyword in GOOGLE_PIXEL_KEYWORDS:
            if keyword in query_lower:
                return "google"
        
        # Priority 0.4: Check Redmi keywords BEFORE Realme (handles "redme", typos)
        # This ensures "redme note 13" is detected as Redmi, not Realme
        for keyword in REDMI_KEYWORDS:
            if keyword in query_lower:
                return "redmi"
        
        # Priority 0.45: Check iQOO keywords BEFORE Realme (handles "iqoo neo", typos)
        # This ensures "iqoo neo 10" is detected as iQOO, not Realme GT NEO
        for keyword in IQOO_KEYWORDS:
            if keyword in query_lower:
                return "iqoo"
        
        # Priority 0.5: Check Realme keywords (handles "relme", "ralame", typos)
        # Skip "neo" detection if "macbook" is in the query (MacBook Neo is Apple, not Realme)
        # Also skip "neo" detection if "iqoo" is in the query (iQOO Neo is iQOO brand)
        for keyword in REALME_KEYWORDS:
            if keyword in query_lower:
                # Skip "neo" keyword when it's part of MacBook query or iQOO query
                if keyword == "neo" and ("macbook" in query_lower or "mac book" in query_lower or "iqoo" in query_lower):
                    continue
                return "realme"
        
        # Sort brands by length (longer first) to match "samsung mobiles" before "samsung"
        sorted_brands = sorted(BRAND_NAMES, key=len, reverse=True)
        
        # TV context indicators - if these are in query, "tvs" means televisions, not TVS brand
        _tv_context_words = {
            "tv", "television", "televisions", "led", "lcd", "oled", "qled", "smart", "uhd", "hdr", "4k", "8k",
            "inch", "inches", "sony", "lg", "philips", "toshiba", "tcl", "vu", "hisense", "panasonic"
        }
        
        # Direct brand match with word boundary
        for brand in sorted_brands:
            brand_lower = brand.lower()
            # Skip if this is actually a product model name, not a brand
            if brand_lower in PRODUCT_MODEL_NAMES:
                continue
            # For multi-word brands, check if all words are present
            if ' ' in brand_lower:
                brand_words = set(brand_lower.split())
                if brand_words.issubset(query_words):
                    return brand_lower
            # For single-word brands
            elif re.search(r'\b' + re.escape(brand_lower) + r'\b', query_lower):
                # Skip matching "tvs" as TVS brand if query has TV context (e.g., "sony tvs", "55 inch tvs")
                if brand_lower == "tvs" and (any(w in query_words for w in _tv_context_words) or 
                                              re.search(r'\b\d{2,3}\s*tvs?\b', query_lower)):
                    continue
                return brand_lower
        
        # =================== CAR MODEL → BRAND RESOLUTION ===================
        # If query contains a known car model name, resolve the brand from CANONICAL_MODEL_BRAND_MAP
        # e.g., "venue car" → brand "hyundai", "nexon ev" → brand "tata"
        # Only match car-specific models (not generic words like "city", "polo" that overlap)
        CAR_BRAND_SET = {"hyundai", "tata", "maruti", "mahindra", "kia", "mg", "volkswagen",
                         "toyota", "honda", "skoda", "nissan", "renault", "citroen", "jeep", "bmw"}
        CAR_CONTEXT_WORDS = {"car", "cars", "suv", "sedan", "hatchback", "ev", "electric"}
        has_car_context = bool(query_words & CAR_CONTEXT_WORDS)
        for word in query_words:
            if word in CANONICAL_MODEL_BRAND_MAP:
                mapped_brand = CANONICAL_MODEL_BRAND_MAP[word]
                if mapped_brand in CAR_BRAND_SET:
                    # For ambiguous short words (city, polo, etc.), require car context
                    if len(word) <= 4 and not has_car_context:
                        continue
                    return mapped_brand

        # =================== FUZZY FALLBACK FOR BRAND ===================
        # Try fuzzy matching for brand names as a last resort
        for word in query_words:
            if len(word) >= 3:  # Only try fuzzy on words with 3+ chars
                brand = get_brand_from_fuzzy(word, threshold=0.70)
                if brand:
                    # Skip "tvs" brand if TV context is present (user likely means televisions)
                    if brand.lower() == "tvs" and (any(w in query_words for w in _tv_context_words) or
                                                    re.search(r'\b\d{2,3}\s*tvs?\b', query_lower)):
                        continue
                    logger.info(f"Fuzzy brand match: '{word}' → brand '{brand}'")
                    return brand
        
        return None
    
    def extract_attributes(self, query: str) -> Dict[str, Any]:
        """Extract product attributes like storage, RAM, color from query
        
        Handles various patterns:
        - "8gb ram" → ram=8
        - "128gb storage" → storage=128
        - "128gb" (no context) → storage=128 (most common use case)
        - "8gb ram 128gb" → ram=8, storage=128
        - "6gb ram oppo" → ram=6
        - "256gb phone" → storage=256
        """
        attrs = {}
        query_lower = query.lower()
        
        # Step 1: Extract RAM patterns FIRST (more specific - requires "ram" keyword)
        # Patterns: "8gb ram", "8 gb ram", "ram 8gb", "ram 8 gb"
        ram_patterns = [
            r'(\d+)\s*gb\s*ram',       # "8gb ram", "8 gb ram"
            r'ram\s*(\d+)\s*gb',       # "ram 8gb", "ram 8 gb"
            r'(\d+)\s*gb\s+ram',       # "8 gb  ram" (multiple spaces)
        ]
        ram_found = None
        for pattern in ram_patterns:
            ram_match = re.search(pattern, query_lower)
            if ram_match:
                ram_found = int(ram_match.group(1))
                attrs['ram'] = ram_found
                break
        
        # Step 2: Extract storage patterns
        # Match sizes that are clearly storage (128, 256, 512, 1024, etc.) or explicitly marked
        # Patterns: "128gb", "256 gb storage", "storage 128gb", "128gb phone", "1tb"
        storage_patterns = [
            r'(\d+)\s*gb\s*storage',    # "128gb storage"
            r'storage\s*(\d+)\s*gb',    # "storage 128gb"
        ]
        
        # First check for TB patterns (1tb, 2tb, etc.)
        tb_match = re.search(r'(\d+)\s*tb', query_lower)
        if tb_match:
            tb_size = int(tb_match.group(1))
            attrs['storage_tb'] = tb_size  # Store TB separately for proper formatting
            attrs['storage'] = tb_size * 1024  # Also store in GB for reference
            storage_found = tb_size * 1024
        else:
            storage_found = None
            for pattern in storage_patterns:
                storage_match = re.search(pattern, query_lower)
                if storage_match:
                    size = int(storage_match.group(1))
                    storage_found = size
                    attrs['storage'] = storage_found
                    break
        
        # Step 3: If no explicit storage found, look for standalone GB values
        # But EXCLUDE values already captured as RAM
        if not storage_found:
            # Find ALL gb values in the query
            all_gb_matches = re.findall(r'(\d+)\s*gb', query_lower)
            for gb_val in all_gb_matches:
                gb_int = int(gb_val)
                # Skip if this is the RAM value we already found
                if ram_found and gb_int == ram_found:
                    # Check if this specific occurrence is the RAM one
                    # by seeing if "ram" follows this number
                    ram_context = re.search(rf'{gb_val}\s*gb\s*ram', query_lower)
                    if ram_context:
                        continue  # This is the RAM value, skip
                # Assume larger values (>=32) or values after skipping RAM are storage
                # Common storage: 32, 64, 128, 256, 512, 1024
                # Common RAM: 2, 3, 4, 6, 8, 12, 16
                if gb_int >= 32 or (gb_int not in [2, 3, 4, 6, 8, 12, 16] and not ram_found):
                    attrs['storage'] = gb_int
                    break
                elif not ram_found and gb_int <= 16:
                    # Small value without "ram" keyword could be either
                    # In context like "16gb phone" it's likely storage
                    # In context like "16gb laptop" check for product type
                    if any(kw in query_lower for kw in ['phone', 'mobile', 'iphone', 'samsung', 'oppo', 'vivo', 'realme', 'redmi']):
                        # For phones, small GB without "ram" is ambiguous - could be storage
                        # Common phone storage: 32, 64, 128, 256
                        # If it's a phone-related query with small value, might be RAM
                        pass  # Skip ambiguous cases
                    else:
                        attrs['storage'] = gb_int
                        break
        
        # Step 4: Handle combined patterns like "8gb+128gb" or "8gb 128gb"
        combined_match = re.search(r'(\d+)\s*gb\s*[+\s]\s*(\d+)\s*gb', query_lower)
        if combined_match:
            val1 = int(combined_match.group(1))
            val2 = int(combined_match.group(2))
            # Smaller is typically RAM, larger is storage
            if val1 < val2:
                attrs['ram'] = val1
                attrs['storage'] = val2
            else:
                attrs['ram'] = val2
                attrs['storage'] = val1
        
        # Screen size (55 inch, 65")
        screen_match = re.search(r'(\d+)\s*(inch|in|")', query_lower)
        if screen_match:
            attrs['screen_size'] = int(screen_match.group(1))
        
        # AC tonnage (1.5 ton, 2 ton)
        tonnage_match = re.search(r'(\d+\.?\d*)\s*ton', query_lower)
        if tonnage_match:
            attrs['tonnage'] = float(tonnage_match.group(1))
        
        # Washing machine capacity (7kg, 8 kg)
        capacity_match = re.search(r'(\d+\.?\d*)\s*kg', query_lower)
        if capacity_match:
            attrs['capacity_kg'] = float(capacity_match.group(1))
        
        # =================== REFRIGERATOR CAPACITY (LITRES) ===================
        # Patterns: "201 to 300 L", "300L", "200 litres", "201-300l", "250 litre"
        # Valid ES values: "80 L and Below", "81 to 170 L", "171 to 200 L", 
        #                  "201 to 300 L", "301 to 400 L", "401 to 500 L", "501 L and Above"
        litre_range_patterns = [
            # Range patterns: "201 to 300 L", "171-200 l", "301 to 400 litres"
            (r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:l|litre|liter|litres|liters)\b', 'range'),
            # Single value patterns: "300L", "250 litres", "200 litre"
            (r'(\d+)\s*(?:l|litre|liter|litres|liters)\b', 'single'),
        ]
        
        for pattern, pattern_type in litre_range_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if pattern_type == 'range':
                    # Range: "201 to 300 L" -> use the range midpoint or exact match
                    low = int(match.group(1))
                    high = int(match.group(2))
                    attrs['capacity_litres_range'] = f"{low} to {high} L"
                    # Store midpoint for range matching
                    attrs['capacity_litres'] = (low + high) / 2
                else:
                    # Single value: "300L" -> map to appropriate range
                    attrs['capacity_litres'] = int(match.group(1))
                break
        
        # =================== STAR/ENERGY RATING ===================
        # Patterns: "5 star", "5star", "5-star", "five star"
        # Used for AC, refrigerator, washing machine, etc.
        star_patterns = [
            r'(\d)\s*[-]?\s*star',       # "5 star", "5star", "5-star"
            r'(\d)\s*stars',              # "5 stars"
        ]
        # Word to digit mapping for star ratings
        star_word_map = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5
        }
        for pattern in star_patterns:
            star_match = re.search(pattern, query_lower)
            if star_match:
                attrs['star_rating'] = int(star_match.group(1))
                break
        # Also check word patterns like "five star"
        if 'star_rating' not in attrs:
            for word, digit in star_word_map.items():
                if re.search(rf'\b{word}\s*[-]?\s*star', query_lower):
                    attrs['star_rating'] = digit
                    break
        
        # =================== REFRIGERATOR DOOR TYPE ===================
        # Patterns: "double door", "single door", "side by side", "french door", "triple door"
        # Note: "4 door" / "four door" maps to attribute values "Four Door" or "Multi Door"
        # Note: "3 door" / "three door" maps to "Triple Door"
        door_type_patterns = [
            (r'double\s*door', 'Double Door'),
            (r'single\s*door', 'Single Door'),
            (r'side\s*by\s*side', 'Side By Side Door'),
            (r'french\s*door', 'French Door'),
            (r'(?:triple|three|3)\s*doors?', 'Triple Door'),  # "3 door", "triple door", "three doors"
            (r'multi\s*door', 'Multi Door'),
            (r'(?:four|4)\s*doors?', 'Four Door'),  # "4 door", "four doors" -> Four Door
        ]
        for pattern, door_value in door_type_patterns:
            if re.search(pattern, query_lower):
                attrs['door_type'] = door_value
                break

        # =================== WASHING MACHINE FUNCTION TYPE ===================
        # Patterns: "front load", "top load", "semi automatic", "fully automatic"
        # ES values: "Front Load", "Top Load", "Fully Automatic Front Load",
        #            "Fully Automatic Top Load", "Semi Automatic Top Load", etc.
        wm_type_patterns = [
            (r'front\s*load', 'Front Load'),
            (r'top\s*load', 'Top Load'),
            (r'semi\s*[-\s]?auto(?:matic)?', 'Semi Automatic'),
            (r'fully\s*[-\s]?auto(?:matic)?', 'Fully Automatic'),
        ]
        for pattern, wm_value in wm_type_patterns:
            if re.search(pattern, query_lower):
                attrs['wm_type'] = wm_value
                break

        # Color patterns
        colors = ['black', 'white', 'silver', 'gold', 'blue', 'red', 'green', 'grey', 'gray', 
                  'purple', 'pink', 'orange', 'yellow', 'brown', 'titanium', 'graphite', 'midnight']
        for color in colors:
            if re.search(rf'\b{color}\b', query_lower):
                attrs['color'] = color
                break
        
        return attrs
    
    def detect_product_boost(self, query: str) -> Dict[str, Any]:
        """
        Detect specific product types that need boosting.
        Returns boost info for bed, fan, sofa, etc.
        """
        if not query:
            return {"boost_type": None, "boost_terms": [], "category_hint": None}
        
        query_lower = query.lower().strip()
        
        # PRIORITY: Check for FAN keywords FIRST (before furniture)
        # This ensures "table fan", "ceiling fan" etc. return fans, not furniture
        fan_keywords = HOME_APPLIANCE_KEYWORDS.get("fan", [])
        for keyword in fan_keywords:
            if keyword in query_lower:
                return {
                    "boost_type": "fan",
                    "boost_terms": fan_keywords,
                    "category_hint": None
                }
        
        # PRIORITY: Specific table types detection (dining, study, coffee, center)
        # This ensures "dining table" / "dinner table" returns dining tables, not generic furniture
        if "dining" in query_lower or "dinner" in query_lower or "diner" in query_lower:
            return {
                "boost_type": "dining_table",
                "boost_terms": ["dining table", "dining", "dinner table", "dining set"],
                "category_hint": "furniture"
            }
        if "study" in query_lower:
            return {
                "boost_type": "study_table",
                "boost_terms": ["study table", "study desk", "computer table", "office table"],
                "category_hint": "furniture"
            }
        if "coffee" in query_lower:
            return {
                "boost_type": "coffee_table",
                "boost_terms": ["coffee table", "center table"],
                "category_hint": "furniture"
            }
        if "center" in query_lower:
            return {
                "boost_type": "center_table",
                "boost_terms": ["center table", "coffee table"],
                "category_hint": "furniture"
            }
        
        # EXCLUSION: Skip furniture table detection for "tablet" queries
        # "tablet", "android tablet", "samsung tablet" should NOT match furniture "table"
        if "tablet" in query_lower or "ipad" in query_lower or "tab" in query_lower.split():
            # Skip table/furniture detection for electronic tablets
            pass
        else:
            # Check furniture products
            for product_type, keywords in FURNITURE_PRODUCT_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        return {
                            "boost_type": product_type,
                            "boost_terms": keywords,
                            "category_hint": "furniture"
                        }
        
        # Check home appliance products (excluding fan, already checked above)
        for product_type, keywords in HOME_APPLIANCE_KEYWORDS.items():
            if product_type == "fan":
                continue  # Already checked above
            for keyword in keywords:
                if keyword in query_lower:
                    return {
                        "boost_type": product_type,
                        "boost_terms": keywords,
                        "category_hint": None
                    }
        
        return {"boost_type": None, "boost_terms": [], "category_hint": None}
    
    def detect_phone_brand_boost(self, query: str) -> List[str]:
        """
        Detect if this is a generic phone query that needs brand boosting.
        Returns list of brands to boost (vivo, oppo, samsung) for 'phone' queries.
        """
        if not query:
            return []
        
        query_lower = query.lower().strip()
        
        # Plain phone queries should boost these brands
        phone_patterns = [r'^phone$', r'^phones$', r'^mobile$', r'^mobiles$', 
                         r'^smartphone$', r'^smartphones$', r'^mobile phone$']
        
        for pattern in phone_patterns:
            if re.match(pattern, query_lower):
                return ["vivo", "oppo", "samsung"]
        
        return []
    
    def process(self, query: str) -> Dict[str, Any]:
        """Process query and extract all relevant information"""
        if not query:
            return {"original": "", "processed": "", "category": None, "brand": None, 
                    "attributes": {}, "is_apple_query": False, "product_boost": {}, "phone_brand_boost": []}
        
        original = query
        
        # Check for phone brand boost FIRST on original query (before any processing)
        phone_brand_boost = self.detect_phone_brand_boost(original)
        
        # =================== APPLE PRODUCT DETECTION (ALL PRODUCTS) ===================
        # Check for ANY Apple product (iPhone, iPad, MacBook, AirPods, Apple Watch)
        # When Apple is detected, show ONLY Apple products (no other brands)
        # NOTE: We now have ALL Apple products in our data!
        is_apple, apple_category_hint = is_apple_product_query(query)
        
        if is_apple:
            # Normalize Apple query for better matching
            processed = normalize_apple_query(query)
            # Use category hint if specific product detected, otherwise show all Apple products
            category = apple_category_hint  # Can be "smartphone", "tablets", "laptops", etc. or None
            brand = "apple"          # Force Apple brand
            product_boost = {}
            phone_brand_boost = []  # Override - no phone boost for Apple queries
            bike_only_filter = False  # Not applicable for Apple queries
            scooter_only_filter = False  # Not applicable for Apple queries
            is_electric_scooter_query = False  # Not applicable for Apple queries
            is_brand_only_phone_query = False  # Not applicable for Apple queries
            logger.info(f"Apple product detected: '{query}' → brand=apple, category_hint={category}")
        else:
            # =================== PRE-CLEAN FLAG KEYWORDS ===================
            # Clean flag keywords (one emi off, zero dp, etc.) from query BEFORE typo correction
            # This prevents fuzzy matching from corrupting them (e.g., "one" → "fone")
            query_for_typo_correction = query.lower()
            
            # Flag keywords to clean before typo correction
            flag_keywords_to_preclean = [
                # One EMI Off keywords
                "one emi off", "1 emi off", "emi off", "emi free",
                "one month emi off", "1 month emi off", "first emi off",
                "no first emi", "skip first emi",
                # Zero DP keywords
                "zero dp", "zero down payment", "zero downpayment", "0dp",
                "0 dp", "no down payment", "no downpayment", "no dp", "zero down",
                "0 down", "nodp", "zerodp", "without down payment", "without dp",
                # New launch keywords
                "new launch", "newlaunch", "newly launched", "latest launch",
                "latest launched", "new arrival", "new arrivals", "just launched",
                "recently launched", "brand new", "latest model", "latest models",
                "newest", "new model", "new models", "latest",
                # Best selling keywords
                "best selling", "bestselling", "best seller", "bestseller",
                "top selling", "topselling", "top seller", "topseller",
                "most sold", "most popular", "popular products", "trending",
                "hot selling", "fast selling", "highest selling"
            ]
            # Sort by length descending to match longer phrases first
            flag_keywords_to_preclean = sorted(flag_keywords_to_preclean, key=len, reverse=True)
            
            for kw in flag_keywords_to_preclean:
                query_for_typo_correction = re.sub(
                    rf'\b{re.escape(kw)}\b', '', query_for_typo_correction, flags=re.IGNORECASE
                ).strip()
            query_for_typo_correction = re.sub(r'\s+', ' ', query_for_typo_correction).strip()
            
            # Now apply typo correction to the cleaned query
            processed = self.correct_typos(query_for_typo_correction) if query_for_typo_correction else ""
            category = self.detect_category(processed)
            brand = self.detect_brand(processed)
            product_boost = self.detect_product_boost(original)  # Use original for product detection
            
            # If product boost has a category hint, use it
            if product_boost.get("category_hint") and not category:
                category = product_boost["category_hint"]
            
            # =================== TWO-WHEELER MODEL HANDLING ===================
            # Handle two-wheeler queries differently based on what's detected:
            # 1. Brand + generic keyword (honda bike) → keep brand, show that brand's bikes
            # 2. Specific model name (pulsar, bullet) → use brand from model mapping
            # 3. Scooter models (activa, jupiter) → apply scooter-only filter
            bike_only_filter = False  # Flag to show only bikes (not scooters)
            scooter_only_filter = False  # Flag to show only scooters (not bikes)
            is_electric_scooter_query = False  # Flag for electric two-wheeler queries
            
            # Generic two-wheeler keywords that should NOT trigger brand clearing
            GENERIC_TWO_WHEELER_KEYWORDS = {
                "bike", "bikes", "motorcycle", "motorbike", "motor bike", "motor cycle",
                "scooter", "scooty", "moped", "two wheeler", "twowheeler", "2 wheeler", "2wheeler"
            }
            
            # Use the comprehensive CANONICAL_MODEL_BRAND_MAP defined at module level
            # For SPECIFIC_MODEL_NAMES, use union of all model mappings
            SPECIFIC_MODEL_NAMES = set(CANONICAL_MODEL_BRAND_MAP.keys())
            
            if category == "two wheeler":
                processed_words = processed.lower().split()
                
                # Check if query has a specific model name (not generic keyword)
                specific_model_found = None
                for word in processed_words:
                    if word in SPECIFIC_MODEL_NAMES:
                        specific_model_found = word
                        break
                
                # Check if query has generic bike/scooter keyword
                has_generic_keyword = any(kw in processed_words or kw in processed.lower() for kw in GENERIC_TWO_WHEELER_KEYWORDS)
                has_scooter_keyword = any(kw in processed.lower() for kw in ["scooter", "scooty", "moped", "gearless"])
                has_bike_keyword = any(kw in processed.lower() for kw in ["bike", "bikes", "motorcycle", "motorbike"])
                
                if specific_model_found:
                    # Specific model found (pulsar, bullet, activa, etc.)
                    # KEEP search text as model name for text matching
                    processed = specific_model_found  # Keep model name for search relevance
                    
                    # SET brand from CANONICAL mapping (comprehensive with typo handling)
                    if specific_model_found in CANONICAL_MODEL_BRAND_MAP:
                        brand = CANONICAL_MODEL_BRAND_MAP[specific_model_found]
                        logger.info(f"Model '{specific_model_found}' mapped to brand '{brand}' (canonical)")
                    else:
                        brand = None  # Unknown model - show all brands
                    
                    # =================== SUBCATEGORY FILTER ===================
                    # Apply scooter-only or bike-only filter based on model type
                    if specific_model_found in SCOOTER_ONLY_MODELS:
                        scooter_only_filter = True
                        bike_only_filter = False
                        logger.info(f"Scooter model '{specific_model_found}' detected - filtering to Scooters only, brand={brand}")
                    elif specific_model_found in BIKE_ONLY_MODELS:
                        bike_only_filter = True
                        scooter_only_filter = False
                        logger.info(f"Bike model '{specific_model_found}' detected - filtering to exclude Scooters, brand={brand}")
                    else:
                        logger.info(f"Two-wheeler model '{specific_model_found}' detected - showing all subcategories, brand={brand}")
                        
                elif has_generic_keyword and brand:
                    # Generic keyword with brand (honda bike) - KEEP brand, show that brand's vehicles
                    # Just remove generic keyword from processed text
                    processed_words = [w for w in processed_words if w not in GENERIC_TWO_WHEELER_KEYWORDS]
                    processed = " ".join(processed_words) if processed_words else brand
                    
                    # Apply subcategory filter based on keyword type
                    if has_bike_keyword:
                        bike_only_filter = True
                        logger.info(f"Brand+bike keyword: brand={brand}, filtering to bikes only")
                    elif has_scooter_keyword:
                        scooter_only_filter = True
                        logger.info(f"Brand+scooter keyword: brand={brand}, filtering to scooters only")
                    else:
                        logger.info(f"Brand+generic keyword: brand={brand}, showing all subcategories")
                        
                elif has_scooter_keyword and not brand and not specific_model_found:
                    # Generic scooter query without brand (just "scooter" or "scooty")
                    scooter_only_filter = True
                    logger.info("Generic scooter query - filtering to Scooters subcategory")
                    
                elif has_bike_keyword and not brand and not specific_model_found:
                    # Generic bike query without brand (just "bike" or "motorcycle")
                    bike_only_filter = True
                    logger.info("Generic bike query - filtering to exclude Scooters")
            
            # =================== ELECTRIC SCOOTER BOOST ===================
            # Check if this is an electric scooter query and set a flag
            electric_scooter_keywords = {"electric scooter", "electric scooty", "ev scooter", "e scooter",
                                         "electric two wheeler", "electric bike", "e bike", "ev bike"}
            is_electric_scooter_query = any(kw in processed.lower() or kw in original.lower() for kw in electric_scooter_keywords)
            
            # =================== TWO-WHEELER BRAND NOT IN DATA ===================
            # If user searches a two-wheeler brand that doesn't exist in our data (e.g., KTM bikes)
            # Redirect to show all bikes from other brands
            # NOTE: Removed "bajaj" - Bajaj bikes (Pulsar, Chetak, etc.) ARE in our data!
            TWO_WHEELER_BRANDS_NOT_IN_DATA = {"ktm", "kawasaki", "ducati", "harley", "harley davidson", 
                                               "benelli", "aprilia", "mv agusta", "triumph", "bmw motorrad"}
            if category == "two wheeler" and brand and brand.lower() in TWO_WHEELER_BRANDS_NOT_IN_DATA:
                logger.info(f"Two-wheeler brand '{brand}' not in data - showing all bikes from other brands")
                brand = None  # Clear brand to show all
                processed = ""  # Clear text
                bike_only_filter = True  # Show bikes only
            
            # =================== BLOCKED BRAND HANDLING ===================
            # Check if query contains a blocked brand (e.g., Vestar)
            # If so, remove the brand and keep only the category to show other brands
            query_lower = processed.lower()
            query_words = set(query_lower.split())
            blocked_brand_found = None
            for blocked_kw in BLOCKED_BRAND_KEYWORDS:
                if blocked_kw in query_words or blocked_kw in query_lower:
                    blocked_brand_found = blocked_kw
                    break
            
            if blocked_brand_found:
                logger.info(f"Blocked brand detected: '{blocked_brand_found}' - removing brand filter")
                # Remove the blocked brand from the processed query
                processed_words = processed.lower().split()
                processed_words = [w for w in processed_words if w not in BLOCKED_BRAND_KEYWORDS]
                processed = " ".join(processed_words)
                # Clear the brand so other brands are shown
                brand = None
                # Re-detect category from remaining words if not already set
                if not category and processed.strip():
                    category = self.detect_category(processed)
                # If query is now empty (user only searched blocked brand), use default category
                if not processed.strip() and not category:
                    default_cat = BLOCKED_BRAND_DEFAULT_CATEGORY.get(blocked_brand_found)
                    if default_cat:
                        category = default_cat
                        logger.info(f"Using default category for blocked brand '{blocked_brand_found}': {category}")
            
            # =================== PHONE BRAND PRIMARY CATEGORY ===================
            # UPDATED: For brand-only queries, show ALL categories from that brand
            # instead of forcing smartphone category
            # This allows "oppo" to show watches, tablets, etc. along with phones
            # BUT we will BOOST smartphones to appear first
            is_brand_only_phone_query = False
            if brand and brand.lower() in PHONE_BRAND_PRIMARY_CATEGORY:
                query_words = set(processed.lower().split())
                brand_words = set(brand.lower().split())
                # Check if query is ONLY the brand name (no other significant words)
                non_brand_words = query_words - brand_words - {"mobiles", "mobile", "phones", "phone"}
                if not non_brand_words or len(non_brand_words) == 0:
                    # Query is just brand name - DON'T force smartphone category
                    # Let search return all products from this brand, sorted by relevance
                    # BUT mark it for smartphone boosting
                    is_brand_only_phone_query = True
                    if category in ["watch and wearable"]:
                        # Don't force smartphone - let it show all categories
                        category = None  # Remove category filter to show all products
                    logger.info(f"Phone brand only query - will boost smartphones for '{brand}'")
        
        attributes = self.extract_attributes(processed)
        
        # Remove color attribute if the color word is part of the brand name
        # e.g., "blue star ac" → "blue" is part of brand "blue star", not a color filter
        if attributes.get("color") and brand:
            brand_lower = brand.lower()
            color_val = attributes["color"].lower()
            if color_val in brand_lower.split() or color_val in brand_lower:
                del attributes["color"]
                logger.info(f"Removed color '{color_val}' - it's part of brand name '{brand}'")
        
        # =================== INFER CATEGORY FROM ATTRIBUTES ===================
        # When user searches only by attribute (e.g., "201 to 300 L"), infer the category
        # This allows attribute-only searches to work without explicit category keywords
        if not category and attributes:
            # Litre capacity → Refrigerator
            if attributes.get('capacity_litres') or attributes.get('capacity_litres_range'):
                category = "refrigerators"
                processed = "refrigerator"  # Add search term for better matching
                logger.info(f"Inferred category 'refrigerators' from litre capacity attribute")
            # Door type → Refrigerator  
            elif attributes.get('door_type'):
                category = "refrigerators"
                processed = "refrigerator"
                logger.info(f"Inferred category 'refrigerators' from door_type attribute")
            # WM function type (front/top load) → Washing Machine
            elif attributes.get('wm_type'):
                category = "washing machines"
                processed = "washing machine"
                logger.info(f"Inferred category 'washing machines' from wm_type attribute")
            # Tonnage → Air Conditioner
            elif attributes.get('tonnage') and not attributes.get('capacity_kg'):
                category = "air conditioner"
                processed = "air conditioner"
                logger.info(f"Inferred category 'air conditioner' from tonnage attribute")
            # Capacity kg → Washing Machine
            elif attributes.get('capacity_kg'):
                category = "washing machines"
                processed = "washing machine"
                logger.info(f"Inferred category 'washing machines' from capacity_kg attribute")
        
        # =================== CATEGORY-ONLY QUERY NORMALIZATION ===================
        # For pure category queries like "4 wheeler", "four wheeler", "television", etc.
        # Replace the search text with the category name to ensure ES matches
        # This prevents fallback from removing category filter when text doesn't match
        CATEGORY_ONLY_KEYWORDS = {
            # Four-wheeler / Car keywords and typos → search for "car"
            "4 wheeler": "car", "4wheeler": "car", "four wheeler": "car", 
            "fourwheeler": "car", "4-wheeler": "car", "four-wheeler": "car",
            "wheeler": "car", "wheelers": "car",
            # Wheeler typos
            "wheler": "car", "whelers": "car", "weeler": "car", "weelers": "car",
            "wheelar": "car", "wheelr": "car", "wheleer": "car", "wheelerr": "car",
            "4 wheler": "car", "4wheler": "car", "four wheler": "car",
            "for wheeler": "car", "for wheler": "car",  # "for" instead of "four"
            # Two-wheeler keywords and typos → search for "bike scooter"
            "2 wheeler": "bike scooter", "2wheeler": "bike scooter", 
            "two wheeler": "bike scooter", "twowheeler": "bike scooter",
            "2-wheeler": "bike scooter", "two-wheeler": "bike scooter",
            "2 wheler": "bike scooter", "two wheler": "bike scooter",
            "to wheeler": "bike scooter", "to wheler": "bike scooter",  # "to" instead of "two"
            # Television typos → search for "television"
            "televisn": "television", "televison": "television", "telivision": "television",
            "telvision": "television", "televisoin": "television", "televsion": "television",
            # Refrigerator typos
            "refridgerator": "refrigerator", "refrigerater": "refrigerator", 
            "refrgrator": "refrigerator", "refregerator": "refrigerator",
            # Washing machine typos
            "washng machine": "washing machine", "washin machine": "washing machine",
            "washing machin": "washing machine", "washingmachin": "washing machine",
        }
        
        processed_lower = processed.lower().strip()
        if processed_lower in CATEGORY_ONLY_KEYWORDS and category:
            # This is a pure category query - replace text for better ES matching
            new_text = CATEGORY_ONLY_KEYWORDS[processed_lower]
            logger.info(f"Category-only query '{processed}' → search text '{new_text}' (category: {category})")
            processed = new_text
        
        # =================== SPECIAL KEYWORD DETECTION (BRD Requirement 2) ===================
        # Detect special keywords that affect filtering/sorting:
        # 1. "zero down payment" / "zero dp" / "no down payment" → filter by zero_dp_flag
        # 2. "new launch" / "latest" / "newly launched" → filter by new_launch_flag
        
        original_lower = original.lower()
        
        # Zero down payment keyword detection
        zero_dp_keywords = [
            "zero down payment", "zero downpayment", "zero dp", "0 down payment",
            "0 dp", "no down payment", "no downpayment", "no dp", "zero down",
            "0 down", "nodp", "zerodp", "without down payment", "without dp"
        ]
        detect_zero_dp = any(kw in original_lower for kw in zero_dp_keywords)
        
        # New launch keyword detection  
        new_launch_keywords = [
            "new launch", "newlaunch", "newly launched", "latest launch",
            "latest launched", "new arrival", "new arrivals", "just launched",
            "recently launched", "brand new", "latest model", "latest models",
            "newest", "new model", "new models", "latest"  # Added "latest" as standalone
        ]
        detect_new_launch = any(kw in original_lower for kw in new_launch_keywords)
        
        # Best selling / Top seller keyword detection
        best_selling_keywords = [
            "best selling", "bestselling", "best seller", "bestseller",
            "top selling", "topselling", "top seller", "topseller",
            "most sold", "most popular", "popular products", "trending",
            "hot selling", "fast selling", "highest selling"
        ]
        detect_best_selling = any(kw in original_lower for kw in best_selling_keywords)
        
        # One EMI Off keyword detection
        one_emi_off_keywords = [
            "one emi off", "1 emi off", "emi off", "emi free",
            "one month emi off", "1 month emi off", "first emi off",
            "no first emi", "skip first emi"
        ]
        detect_one_emi_off = any(kw in original_lower for kw in one_emi_off_keywords)
        
        if detect_zero_dp:
            logger.info(f"Detected 'zero down payment' keyword in query: '{original}'")
            # Clean the keyword from processed text to improve search
            for kw in zero_dp_keywords:
                processed = re.sub(rf'\b{re.escape(kw)}\b', '', processed, flags=re.IGNORECASE).strip()
            processed = re.sub(r'\s+', ' ', processed).strip()  # Clean extra spaces
        
        if detect_new_launch:
            logger.info(f"Detected 'new launch/latest' keyword in query: '{original}'")
            # Clean the keyword from processed text
            for kw in new_launch_keywords:
                processed = re.sub(rf'\b{re.escape(kw)}\b', '', processed, flags=re.IGNORECASE).strip()
            processed = re.sub(r'\s+', ' ', processed).strip()
        
        if detect_best_selling:
            logger.info(f"Detected 'best selling/top seller' keyword in query: '{original}'")
            # Clean the keyword from processed text
            for kw in best_selling_keywords:
                processed = re.sub(rf'\b{re.escape(kw)}\b', '', processed, flags=re.IGNORECASE).strip()
            processed = re.sub(r'\s+', ' ', processed).strip()
        
        if detect_one_emi_off:
            logger.info(f"Detected 'one emi off' keyword in query: '{original}'")
            # Clean the keyword from processed text
            for kw in one_emi_off_keywords:
                processed = re.sub(rf'\b{re.escape(kw)}\b', '', processed, flags=re.IGNORECASE).strip()
            processed = re.sub(r'\s+', ' ', processed).strip()
        
        # =================== AC TYPE DETECTION (window ac, split ac, etc.) ===================
        # When searching for specific AC types, filter by attribute_ac_type_value
        ac_type_filter = None
        ac_type_mappings = {
            "window ac": "Window",
            "window air conditioner": "Window",
            "window a/c": "Window",
            "split ac": "Split",
            "split air conditioner": "Split",
            "split a/c": "Split",
            "inverter ac": "Inverter Split",
            "inverter split ac": "Inverter Split",
            "inverter air conditioner": "Inverter Split",
            "portable ac": "Portable",
            "portable air conditioner": "Portable",
            "cassette ac": "Cassette",
            "cassette air conditioner": "Cassette",
            "tower ac": "Tower",
            "tower air conditioner": "Tower",
            "inverter window ac": "Inverter Window",
            "inverter window": "Inverter Window",
        }
        for ac_phrase, ac_value in ac_type_mappings.items():
            if ac_phrase in original_lower:
                ac_type_filter = ac_value
                logger.info(f"Detected AC type '{ac_phrase}' → filter: attribute_ac_type_value='{ac_value}'")
                # Keep 'ac' or 'air conditioner' in processed for category matching
                # but remove the type qualifier
                for word in ["window", "split", "portable", "cassette", "tower", "inverter"]:
                    processed = re.sub(rf'\b{word}\b', '', processed, flags=re.IGNORECASE).strip()
                processed = re.sub(r'\s+', ' ', processed).strip()
                break
        
        # =================== TV/TVS DISAMBIGUATION ===================
        # When category is "television" but processed text contains "tvs" (plural of TV),
        # replace "tvs" with "television" in search text to avoid matching TVS two-wheeler brand.
        # This fix is needed because text search for "55 inch tvs" would otherwise match
        # TVS brand products when no television products have "tvs" in their name.
        if category == "television" and re.search(r'\btvs\b', processed, re.IGNORECASE):
            processed = re.sub(r'\btvs\b', 'television', processed, flags=re.IGNORECASE)
            logger.info(f"TV/TVS disambiguation: replaced 'tvs' with 'television' in search text")
        
        return {
            "original": original,
            "processed": processed,
            "category": category,
            "brand": brand,
            "attributes": attributes,
            "is_apple_query": is_apple,
            "product_boost": product_boost,
            "phone_brand_boost": phone_brand_boost,
            "bike_only_filter": bike_only_filter,
            "scooter_only_filter": scooter_only_filter,
            "is_brand_only_phone_query": is_brand_only_phone_query,
            "is_electric_scooter_query": is_electric_scooter_query,
            "detect_zero_dp": detect_zero_dp,  # BRD: filter by zero_dp_flag
            "detect_new_launch": detect_new_launch,  # BRD: filter by new_launch_flag
            "detect_best_selling": detect_best_selling,  # BRD: filter by top_seller_flag
            "detect_one_emi_off": detect_one_emi_off,  # BRD: filter by one_emi_off
            "ac_type_filter": ac_type_filter  # AC type filter (Window, Split, etc.)
        }


# =================== SEARCH ENGINE ===================
class SearchEngine:
    """Handles Elasticsearch query building and execution with BM25"""
    
    def __init__(self, es_client, index_name: str):
        self.es = es_client
        self.index = index_name
    
    def build_query(self, query_info: Dict, city_id: str = None, filters: Dict = None,
                    page: int = 1, page_size: int = 26, emi_range: Dict = None,
                    sort_by: str = None, from_offset: int = None, skip_aggregations: bool = False,
                    skip_expensive_sorts: bool = False) -> Dict:
        """Build Elasticsearch query with BM25 multi-match
        
        Args:
            sort_by: Sorting option - 'relevance' (default), 'low_to_high', 'high_to_low',
                     'latest_launch', 'best_selling', 'most_viewed', 'discounts', 'newest'.
                     City priority: user's city_id products appear first, then city_id=0.
                     Fallback: if sort returns 0 results, fallback to relevance.
            from_offset: Direct offset for pagination (overrides page calculation if provided)
            skip_aggregations: Skip building aggregations (for pagination) - PERFORMANCE optimization
            skip_expensive_sorts: Skip expensive script-based sorts (for pagination pages 2+)
        """
        
        query = query_info.get("processed", "")
        category = query_info.get("category")
        brand = query_info.get("brand")
        attributes = query_info.get("attributes", {})
        is_apple_query = query_info.get("is_apple_query", False)
        product_boost = query_info.get("product_boost", {})
        phone_brand_boost = query_info.get("phone_brand_boost", [])
        is_brand_only_phone_query = query_info.get("is_brand_only_phone_query", False)
        
        # =================== SCORING CONTROL ===================
        # Only these categories use complex scoring/boosting.
        # For furniture, mattress, etc. we use simple exact-match ranking.
        SCORING_ENABLED_CATEGORIES = {
            "smartphone", "mobile", "mobiles",
            "watch and wearable", "watch", "wearable",
            "laptops", "laptop",
            "washing machine", "washing machines",
            "television", "tv", "led tv",
            "refrigerators", "refrigerator", "fridge",
            "air conditioner", "ac",
            "audio video", "audio",
            "two wheeler", "car",
        }
        category_lower = (category or "").lower()
        use_scoring = category_lower in SCORING_ENABLED_CATEGORIES or is_apple_query or is_brand_only_phone_query
        
        # Detect Google Pixel queries for special handling
        original_lower = query_info.get("original", "").lower()
        is_google_pixel_query = brand == "google" and any(kw in original_lower for kw in ["pixel", "pixle", "pixxel", "pixal", "pxel", "piksel"])
        
        # Extract Pixel model number if present (e.g., "pixel 8" → "8", "pixel 9 pro" → "9")
        import re
        pixel_model_match = re.search(r'(?:pixel|pixle|pixxel|pixal|pxel|piksel)\s*(\d+)', original_lower)
        pixel_model_number = pixel_model_match.group(1) if pixel_model_match else None
        
        # Build must clauses
        must_clauses = []
        should_clauses = []
        filter_clauses = []
        
        # =================== GOOGLE PIXEL PRODUCTS BUSINESS RULE ===================
        # If Google Pixel query detected, use brand+category filter ONLY (skip text search)
        # This is needed because ES analyzer doesn't tokenize "pixel" properly in product_name
        # Even "pixel 8" text search returns 0 for Google Pixel phones
        # Solution: Use only manufacturer_desc + actual_category filters
        # Model number boosting is done via function_score later (see GOOGLE PIXEL MODEL BOOST section)
        if is_google_pixel_query:
            # Add HARD FILTER for Google brand
            filter_clauses.append({
                "term": {"manufacturer_desc": "Google"}
            })
            # Force smartphone category
            filter_clauses.append({
                "term": {"actual_category": "smartphone"}
            })
            
            if pixel_model_number:
                logger.info(f"Google Pixel query detected - will boost Pixel {pixel_model_number} models via function_score")
            else:
                logger.info(f"Google Pixel query detected - applying Google+smartphone filter only")
            
            # Skip text search entirely - filters are sufficient
            query = ""
        
        # =================== APPLE PRODUCTS BUSINESS RULE ===================
        # If Apple query detected, ONLY show Apple products (hard filter)
        # Show ALL Apple product categories unless a specific category was detected
        if is_apple_query:
            # Add HARD FILTER for Apple brand - this ensures ONLY Apple products are returned
            filter_clauses.append({
                "bool": {
                    "should": [
                        {"match": {"manufacturer_desc": {"query": "Apple", "operator": "and"}}},
                        {"match": {"attribute_brand_new_value": {"query": "Apple", "operator": "and"}}},
                        {"term": {"attribute_brand_new": "4936"}},  # Apple brand ID
                    ],
                    "minimum_should_match": 1
                }
            })
            # Only apply category filter if a specific Apple product was detected
            # e.g., "iphone" → smartphone, "macbook" → laptops, "airpods" → audio video
            # Generic "apple" queries (category=None) will show ALL Apple products
            # NOTE: Some newer products may have actual_category="unknown", so include that for iPhones
            if category:
                if category == "smartphone":
                    # For iPhone queries, also include "unknown" category (some new iPhones)
                    filter_clauses.append({
                        "bool": {
                            "should": [
                                {"term": {"actual_category": category}},
                                {"term": {"actual_category": "unknown"}}
                            ],
                            "minimum_should_match": 1
                        }
                    })
                else:
                    filter_clauses.append({
                        "term": {"actual_category": category}
                    })
                logger.info(f"Apple query detected - applying Apple-only filter with category '{category}'")
            else:
                logger.info(f"Apple query detected - applying Apple-only filter (all categories)")
        
        # Main search query with BM25 multi-match
        if query:
            must_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        "product_name^4",
                        "product_name.ngram^2",
                        "product_name.edge^3",
                        "sku_name^3",           # Added for SKU matching
                        "search_field^3",       # Added for search field matching
                        "actual_category^2",
                        "manufacturer_desc^3",
                        "search_keywords^2",
                        "attribute_brand_new_value^2",
                        "attribute_color_value",
                        "attribute_internal_storage_value",
                        "attribute_ram_value"
                    ],
                    "type": "best_fields",
                    "operator": "or",
                    "minimum_should_match": "50%",
                    "fuzziness": "AUTO"
                }
            })
            
            # =================== EXACT PHRASE BOOST ===================
            # Add high-weight phrase match boost to prioritize exact model matches
            # E.g., "mac mini" should rank higher than "MacBook Air" for query "mac mini"
            # E.g., "redmi pad" should rank higher than "Redmi 8" for query "redmi pad"
            # ONLY for scoring-enabled categories (smartphones, TV, etc.)
            if use_scoring:
                should_clauses.append({
                    "match_phrase": {
                        "product_name": {
                            "query": query,
                            "boost": 50  # Very high boost for exact phrase match
                        }
                    }
                })
                should_clauses.append({
                    "match_phrase": {
                        "search_field": {
                            "query": query,
                            "boost": 45  # High boost for search_field phrase match
                        }
                    }
                })
                should_clauses.append({
                    "match_phrase": {
                        "sku_name": {
                            "query": query,
                            "boost": 40  # High boost for sku_name phrase match
                        }
                    }
                })
                
                # =================== EXACT QUERY BOOST (ALWAYS APPLIED) ===================
                # Boost products that contain the EXACT query phrase in product name
                # This ensures "ThinkPad" ranks ThinkPad above IdeaPad, "Reno 15" ranks above "Reno 12"
                # Uses product_name.keyword for exact text matching (no tokenization)
                if len(query) >= 3:  # Only for queries with 3+ characters
                    should_clauses.append({
                        "wildcard": {
                            "product_name.keyword": {
                                "value": f"*{query}*",
                                "case_insensitive": True,
                                "boost": 300  # High boost for exact phrase match
                            }
                        }
                    })
                    should_clauses.append({
                        "wildcard": {
                            "sku_name.keyword": {
                                "value": f"*{query}*",
                                "case_insensitive": True,
                                "boost": 280
                            }
                        }
                    })
                    logger.info(f"Exact query boost added for '{query}'")
                
                # =================== MODEL NUMBER BOOST ===================
                # Detect model numbers in query (like "5", "CE5", "5G") and boost exact matches
                # This ensures "Nord 5" ranks above "Nord" when user searches "nord 5"
                # IMPORTANT: Uses product_name.keyword field for exact matching (not analyzed)
                model_number_pattern = re.search(r'\b(ce\s*\d+|[a-z]*\d+[a-z]*)\b', query, re.IGNORECASE)
                if model_number_pattern:
                    model_num = model_number_pattern.group(1).strip()
                    # Use .keyword field for wildcard matching (exact text, no tokenization)
                    # "nord 5" -> "*Nord 5*" to match "OnePlus Nord 5 256 GB Storage..."
                    should_clauses.append({
                        "wildcard": {
                            "product_name.keyword": {
                                "value": f"*{query}*",  # Exact query pattern
                                "case_insensitive": True,
                                "boost": 500  # Extremely high boost for exact match
                            }
                        }
                    })
                    # Also try sku_name.keyword for nested product matching
                    should_clauses.append({
                        "wildcard": {
                            "sku_name.keyword": {
                                "value": f"*{query}*",
                                "case_insensitive": True,
                                "boost": 450
                            }
                        }
                    })
                    # For numeric model numbers like "5", "5g", "4a", ensure space boundaries
                    if model_num.isdigit() or re.match(r'^\d+[a-z]*$', model_num, re.IGNORECASE):
                        should_clauses.append({
                            "wildcard": {
                                "product_name.keyword": {
                                    "value": f"* {model_num} *",  # Space-bounded number
                                    "case_insensitive": True,
                                    "boost": 400  # Very high boost
                                }
                            }
                        })
                    logger.info(f"Model number boost added for '{model_num}' from query '{query}'")
                
                # Add constant_score for ALL words matching (stricter match)
                # This gives a fixed score boost when all query words appear
                query_words_list = query.split()
                if len(query_words_list) >= 2:
                    # Build a term query that requires ALL words to appear
                    should_clauses.append({
                        "bool": {
                            "must": [
                                {"match": {"product_name": {"query": word, "operator": "and"}}}
                                for word in query_words_list if len(word) >= 2
                            ],
                            "boost": 100  # Very high boost when ALL query words match
                        }
                    })
            else:
                # For furniture and other non-scoring categories:
                # Use simple exact match with higher priority, no complex boosting
                # Exact phrase match gets top priority
                should_clauses.append({
                    "match_phrase": {
                        "product_name": {
                            "query": query,
                            "boost": 100  # Highest boost for exact phrase match
                        }
                    }
                })
                logger.info(f"Simple scoring mode for category '{category}' - exact phrase match priority only")
        
        # Category anchor - use filter for exact match (skip if Apple query already set it)
        # =================== STRICT ASSET_CATEGORY_NAME FILTER FOR FURNITURE SUBCATEGORIES ===================
        # For specific furniture subcategories (mattress), use asset_category_name for STRICT filtering
        # This prevents "Sofa Cum Bed" (asset_category_name: Furniture) from appearing in mattress results
        # even if its actual_category is "mattress"
        # Also: For furniture items like "sofa", "bed", exclude Mattress category (sofa cum beds)
        STRICT_ASSET_CATEGORY_MAP = {
            "mattress": "Mattress",      # Only show products with asset_category_name: Mattress
            "mattresses": "Mattress",    # Plural variant
        }
        # Furniture product types that should EXCLUDE Mattress category
        FURNITURE_EXCLUDE_MATTRESS = {"sofa", "bed", "table", "chair", "wardrobe", "cabinet", "bookshelf", "dining_table", "study_table", "center_table", "coffee_table"}
        
        if category and not is_apple_query:
            category_lower = category.lower() if category else ""
            boost_type = product_boost.get("boost_type", "") or ""
            
            if category_lower in STRICT_ASSET_CATEGORY_MAP:
                # Use asset_category_name for strict furniture subcategory filtering
                asset_cat_value = STRICT_ASSET_CATEGORY_MAP[category_lower]
                filter_clauses.append({
                    "term": {"asset_category_name": asset_cat_value}
                })
                logger.info(f"Strict category filter: asset_category_name='{asset_cat_value}' (detected: '{category}')")
            elif category_lower == "furniture" and boost_type in FURNITURE_EXCLUDE_MATTRESS:
                # For furniture items (sofa, bed, etc.), exclude Mattress to avoid "sofa cum bed" appearing
                # Sorting fix (text relevance first) ensures exact matches appear at top
                filter_clauses.append({
                    "term": {"actual_category": category}
                })
                filter_clauses.append({
                    "term": {"asset_category_name": "Furniture"}  # Exclude Mattress category
                })
                logger.info(f"Furniture filter with Mattress exclusion for boost_type='{boost_type}'")
            else:
                # Default: use actual_category for other categories
                filter_clauses.append({
                    "term": {"actual_category": category}
                })
        
        # =================== BIKE-ONLY FILTER ===================
        # If bike model was detected (pulsar, bullet, etc.), show only bikes, not scooters
        bike_only_filter = query_info.get("bike_only_filter", False)
        if bike_only_filter and category == "two wheeler":
            # Filter to show only bikes/motorcycles, exclude scooters
            # asset_category_name in ES: "Mid range premium bikes", "High range premium bikes", 
            # "Super Bikes", "Executive", "Economy", "Electric" (bikes), "Mopeds"
            # Exclude: "Scooters"
            filter_clauses.append({
                "bool": {
                    "must_not": [
                        {"term": {"asset_category_name": "Scooters"}}
                    ]
                }
            })
            logger.info("Bike-only filter applied - excluding scooters")
        
        # =================== ELECTRIC TWO-WHEELER FILTER ===================
        # If this is an electric scooter/bike query, filter to "Electric" subcategory
        # This ensures "electric scooter" shows only electric two-wheelers, not cars
        is_electric_scooter_query = query_info.get("is_electric_scooter_query", False)
        if is_electric_scooter_query and category == "two wheeler":
            # Filter to show only electric two-wheelers
            # asset_category_name in ES: "Electric"
            filter_clauses.append({
                "term": {"asset_category_name": "Electric"}
            })
            
            # Check if query specifically mentions "scooter" - if so, exclude e-bike brands
            original_lower = query_info.get("original", "").lower()
            if "scooter" in original_lower or "scooty" in original_lower:
                # Exclude e-bike brands (Wardwizard = Joy E-bike)
                filter_clauses.append({
                    "bool": {
                        "must_not": [
                            {"term": {"manufacturer_desc": "Wardwizard"}}
                        ]
                    }
                })
                # Boost known electric scooter brands (Ola, Ather, TVS, Hero, Bajaj)
                should_clauses.extend([
                    {"term": {"manufacturer_desc": {"value": "Ola", "boost": 15}}},
                    {"term": {"manufacturer_desc": {"value": "Ather", "boost": 15}}},
                    {"term": {"manufacturer_desc": {"value": "TVS", "boost": 12}}},
                    {"term": {"manufacturer_desc": {"value": "Hero", "boost": 12}}},
                    {"term": {"manufacturer_desc": {"value": "Bajaj", "boost": 12}}},
                    {"term": {"manufacturer_desc": {"value": "Ampere", "boost": 10}}},
                    {"term": {"manufacturer_desc": {"value": "Okinawa", "boost": 10}}},
                    {"term": {"manufacturer_desc": {"value": "Hero Electric", "boost": 10}}},
                    {"term": {"manufacturer_desc": {"value": "Bounce Infinity", "boost": 8}}}
                ])
                logger.info("Electric SCOOTER filter applied - excluding Wardwizard (Joy E-bike), boosting scooter brands")
            
            logger.info("Electric two-wheeler filter applied - showing only Electric subcategory")
        
        # =================== SCOOTER-ONLY FILTER ===================
        # If scooter model was detected (activa, jupiter, chetak, etc.), show only scooters
        # Skip this if electric scooter query (already filtered above)
        scooter_only_filter = query_info.get("scooter_only_filter", False)
        if scooter_only_filter and category == "two wheeler" and not is_electric_scooter_query:
            # Filter to show only scooters
            # asset_category_name in ES: "Scooters"
            filter_clauses.append({
                "term": {"asset_category_name": "Scooters"}
            })
            logger.info("Scooter-only filter applied - showing only Scooters subcategory")
        
        # =================== BRAND DIRECT FILTER ===================
        # If brand is detected, use as HARD FILTER (not boost)
        # This implements "samsung phone" → show ONLY Samsung phones
        # EXCEPTION: When specific product type (headphones, earbuds, etc.) is mentioned,
        # use boost instead of filter. This allows "Boat Headphones" to show headphones
        # from other brands if Boat has no headphones (Boat has only speakers/soundbars).
        
        # Check if query has explicit product type keywords that may not exist for the brand
        _audio_device_keywords = {"headphone", "headphones", "earbuds", "earbud", "earphones", "earphone",
                                  "neckband", "neckbands", "tws", "airdopes", "airdope", "headset", "headsets"}
        _query_words = set(query.lower().split())
        _has_audio_device_type = bool(_query_words & _audio_device_keywords)
        
        # Use boost instead of filter when audio device type is specified
        _use_brand_boost_instead = _has_audio_device_type
        
        if brand and not is_apple_query:
            # When audio device type is mentioned (headphones, earbuds, etc.), skip brand filter
            # This allows "Boat Headphones" to show headphones from ALL brands
            # if Boat doesn't have headphones (Boat only has speakers/soundbars)
            if _use_brand_boost_instead:
                logger.info(f"Brand boost (not filter) will be used for: {brand} - audio device type detected")
                # Brand boost will be applied via function_score later
            else:
                # Check for brand aliases (e.g., iqoo → also search vivo)
                brand_lower = brand.lower()
                if brand_lower in BRAND_ALIAS_MAP:
                    # Use alias mapping for sub-brands
                    alias_brands = BRAND_ALIAS_MAP[brand_lower]
                    brand_should_clauses = []
                    for alias in alias_brands:
                        # For short brands (2-3 chars like "mi", "lg"), use exact term match
                        # For longer brands, use fuzziness and wildcard for flexibility
                        # Note: manufacturer_desc is a keyword field, so use term queries
                        if len(alias) <= 3:
                            # Short brand - use exact term match with different casings
                            brand_should_clauses.append({"term": {"manufacturer_desc": alias.title()}})  # Mi
                            brand_should_clauses.append({"term": {"manufacturer_desc": alias.upper()}})  # MI
                            brand_should_clauses.append({"term": {"manufacturer_desc": alias.capitalize()}})  # Mi
                        else:
                            # Longer brand - use term match with title case + wildcard for variations
                            brand_should_clauses.append({"term": {"manufacturer_desc": alias.title()}})  # Xiaomi
                            brand_should_clauses.append({"term": {"manufacturer_desc": alias.capitalize()}})
                            brand_should_clauses.append({"wildcard": {"manufacturer_desc": {"value": f"*{alias}*", "case_insensitive": True}}})
                    filter_clauses.append({
                        "bool": {
                            "should": brand_should_clauses,
                            "minimum_should_match": 1
                        }
                    })
                    logger.info(f"Brand alias filter applied for '{brand}' → {alias_brands}")
                else:
                    # For short brands (2-3 chars like "lg", "hp"), use exact term match
                    # For longer brands, use wildcard for variations like "Samsung" → "Samsung Mobiles"
                    # Note: manufacturer_desc is a keyword field
                    if len(brand_lower) <= 3:
                        filter_clauses.append({
                            "bool": {
                                "should": [
                                    {"term": {"manufacturer_desc": brand.title()}},
                                    {"term": {"manufacturer_desc": brand.upper()}},
                                    {"term": {"manufacturer_desc": brand.capitalize()}}
                                ],
                                "minimum_should_match": 1
                            }
                        })
                    else:
                        # Longer brand - use term match with wildcard for variations
                        filter_clauses.append({
                            "bool": {
                                "should": [
                                    {"term": {"manufacturer_desc": brand.title()}},
                                    {"wildcard": {"manufacturer_desc": {"value": f"*{brand}*", "case_insensitive": True}}}
                                ],
                                "minimum_should_match": 1
                            }
                        })
                    logger.info(f"Brand direct filter applied for: {brand}")
        
        # =================== PHONE BRAND BOOST ===================
        # For generic "phone" queries, boosting is handled via function_score at the end
        # Just log for now
        if phone_brand_boost and not is_apple_query:
            logger.info(f"Phone brand boost will be applied via function_score for: {phone_brand_boost}")
        
        # =================== PRODUCT TYPE BOOST ===================
        # For specific products like bed, fan, sofa - boosting is handled via function_score
        if product_boost.get("boost_type") and not is_apple_query:
            logger.info(f"Product boost will be applied via function_score for: {product_boost.get('boost_type')}")
            
            # For "fan" queries, filter by asset_category_name = "Fan" to exclude heaters/coolers with "fan" in name
            if product_boost.get("boost_type") == "fan":
                filter_clauses.append({
                    "term": {"asset_category_name": "Fan"}
                })
                logger.info("Fan query detected - applying asset_category_name = Fan filter")
        
        # =================== ATTRIBUTE FILTERS FROM QUERY ===================
        # When user specifies attributes like "8gb ram" or "128gb storage", apply as FILTER
        # This ensures results MUST have the specified attribute
        # Use should_clauses for boost (secondary match) and filter for strict matching
        
        if attributes.get("storage") or attributes.get("storage_tb"):
            # For storage, use term match (exact) as values are "128 GB", "256 GB", "1 TB", etc.
            # Check if this is a TB value or GB value
            # Handle both storage_tb (from QueryProcessor) and large GB values (from enhanced parser)
            storage_gb = attributes.get("storage", 0)
            storage_tb = attributes.get("storage_tb")
            
            # Convert large GB values to TB (1024 GB = 1 TB, 2048 GB = 2 TB)
            if not storage_tb and storage_gb >= 1024 and storage_gb % 1024 == 0:
                storage_tb = storage_gb // 1024
            
            if storage_tb:
                storage_value = str(storage_tb) + " TB"  # Format as "1 TB", "2 TB"
            else:
                storage_value = str(storage_gb) + " GB"  # Format as "128 GB", "256 GB"
            
            query_lower_check = query.lower() if query else ""
            is_laptop_query = any(word in query_lower_check for word in ['laptop', 'macbook', 'notebook', 'thinkpad', 'ideapad'])
            
            if is_laptop_query:
                # For laptops, use should with SSD field (not strict filter)
                should_clauses.append({
                    "term": {"attribute_ssd_value": storage_value}
                })
                logger.info(f"Laptop storage boost applied (SSD): {storage_value}")
            else:
                # For mobiles, use strict filter on internal storage
                filter_clauses.append({
                    "term": {"attribute_internal_storage_value": storage_value}
                })
                logger.info(f"Storage filter applied: {storage_value}")
        
        if attributes.get("ram"):
            # For RAM, use term match (exact) as values are "8 GB", "6 GB", etc.
            ram_value = str(attributes["ram"]) + " GB"
            filter_clauses.append({
                "term": {"attribute_ram_value": ram_value}
            })
            logger.info(f"RAM filter applied: {ram_value}")
        
        # =================== STRICT ATTRIBUTE FILTERS ===================
        # When user specifies exact values (32 inch, 1.5 ton, 8kg), filter strictly
        # This ensures "lg 32 inch tv" only shows 32 inch TVs, not 65 inch
        
        # Define available attribute values in index for fallback logic
        AVAILABLE_SCREEN_SIZES = ['32', '43', '55', '65', '50', '40', '75', '49', '24', '39', '85', '48', '42', 
                                  '58', '70', '77', '86', '46', '98', '60', '28', '22', '31.5', '83', '88', '100',
                                  '23', '38.5', '64', '78', '20', '23.6', '42.5', '45', '51', '79', '82', '19', 
                                  '29', '31', '38', '105', '115', '12', '120', '18.5', '19.5', '27', '30', '31.8']
        
        # Max available tonnage is ~2 tons, max capacity is ~11-12 kg
        MAX_TONNAGE = 2.0
        MAX_CAPACITY = 12.0
        
        if attributes.get("screen_size"):
            requested_size = str(attributes["screen_size"])
            
            if requested_size in AVAILABLE_SCREEN_SIZES:
                # Exact match - use strict filter
                filter_clauses.append({
                    "term": {"attribute_screen_size_in_inches_value": requested_size}
                })
                logger.info(f"Screen size filter applied: {requested_size} inch (exact match)")
            else:
                # Find nearest available size
                try:
                    req_size_num = float(requested_size)
                    available_nums = [(s, abs(float(s) - req_size_num)) for s in AVAILABLE_SCREEN_SIZES 
                                      if s.replace('.', '').isdigit()]
                    available_nums.sort(key=lambda x: x[1])
                    
                    if available_nums:
                        # Get nearest size(s) within 5 inches
                        nearest_sizes = [s for s, diff in available_nums if diff <= 5][:2]
                        if nearest_sizes:
                            filter_clauses.append({
                                "terms": {"attribute_screen_size_in_inches_value": nearest_sizes}
                            })
                            logger.info(f"Screen size {requested_size} not found, using nearest: {nearest_sizes}")
                        else:
                            # No close match - still apply filter (will return 0 results)
                            filter_clauses.append({
                                "term": {"attribute_screen_size_in_inches_value": requested_size}
                            })
                            logger.info(f"Screen size {requested_size} not available, no close match")
                except ValueError:
                    # Non-numeric size, try exact match
                    filter_clauses.append({
                        "term": {"attribute_screen_size_in_inches_value": requested_size}
                    })
        
        if attributes.get("tonnage"):
            # AC tonnage is stored as ranges like "1.3 to 1.7 Tons", "Upto 1.2 Tons"
            
            # ===== CHECK FOR EXACT RANGE MATCH FIRST =====
            # If user query contains an exact ES range value, use it directly
            # Use user_original_query (actual user input) not "original" (may be modified to "air conditioner")
            user_orig = query_info.get("user_original_query", "")
            orig_val = query_info.get("original", "")
            original_query_lower = (user_orig or orig_val).lower()
            
            # ES values for tonnage ranges - must match exactly (case-sensitive)
            ES_TONNAGE_RANGES = {
                "1.5 to 1.9 tons": "1.5 to 1.9 Tons",
                "1.3 to 1.7 tons": "1.3 to 1.7 Tons", 
                "upto 1 tons": "Upto 1 Tons",
                "upto 1.2 tons": "Upto 1.2 Tons",
                "1.8 tons and up": "1.8 Tons and Up",
                "upto 2 tons": "Upto 2 Tons",
                "0.8 tons": "0.8 Tons",
                "1 tons": "1 Tons"
            }
            
            exact_range_match = None
            for query_pattern, es_value in ES_TONNAGE_RANGES.items():
                if query_pattern in original_query_lower:
                    exact_range_match = es_value
                    logger.info(f"Tonnage exact range: '{query_pattern}' → '{es_value}'")
                    break
            
            if exact_range_match:
                # User query contains exact range - use it directly
                filter_clauses.append({
                    "terms": {"attribute_capacity_tons_ac_value": [exact_range_match]}
                })
                logger.info(f"Tonnage filter added: '{exact_range_match}'")
            else:
                # Map user input to the correct range bucket
                try:
                    tonnage = float(attributes["tonnage"])
                    
                    # Cap tonnage to max available (2 tons)
                    if tonnage > MAX_TONNAGE:
                        logger.info(f"Requested {tonnage} ton exceeds max available ({MAX_TONNAGE}), using max range")
                        tonnage = MAX_TONNAGE
                    
                    tonnage_ranges = []
                    if tonnage <= 0.8:
                        tonnage_ranges = ["0.8 Tons", "Upto 1 Tons"]
                    elif tonnage <= 1.0:
                        tonnage_ranges = ["Upto 1 Tons", "1 Tons", "Upto 1.2 Tons"]
                    elif tonnage <= 1.2:
                        tonnage_ranges = ["Upto 1.2 Tons", "1 Tons", "1.3 to 1.7 Tons"]
                    elif tonnage <= 1.5:
                        tonnage_ranges = ["1.3 to 1.7 Tons", "1.5 to 1.9 Tons"]
                    elif tonnage <= 1.7:
                        tonnage_ranges = ["1.3 to 1.7 Tons", "1.5 to 1.9 Tons"]
                    elif tonnage <= 2.0:
                        tonnage_ranges = ["1.8 Tons and Up", "Upto 2 Tons", "1.5 to 1.9 Tons"]
                    else:
                        tonnage_ranges = ["1.8 Tons and Up", "Upto 2 Tons"]
                    
                    # Use correct field name: attribute_capacity_tons_ac_value
                    filter_clauses.append({
                        "terms": {"attribute_capacity_tons_ac_value": tonnage_ranges}
                    })
                    logger.info(f"Tonnage filter applied: {attributes['tonnage']} ton → ranges {tonnage_ranges}")
                except ValueError:
                    logger.warning(f"Invalid tonnage value: {attributes['tonnage']}")
        
        if attributes.get("capacity_kg"):
            # Washing machine capacity is stored as ranges like "7.1 to 8 kg", "6.1 to 7 kg"
            # Map user input to the correct range bucket
            try:
                capacity = float(attributes["capacity_kg"])
                
                # Cap capacity to max available (~12 kg)
                if capacity > MAX_CAPACITY:
                    logger.info(f"Requested {capacity}kg exceeds max available ({MAX_CAPACITY}kg), using max range")
                    capacity = MAX_CAPACITY
                
                capacity_ranges = []
                if capacity <= 6:
                    capacity_ranges = ["6 kg and Below"]
                elif capacity <= 7:
                    capacity_ranges = ["6.1 to 7 kg"]
                elif capacity <= 8:
                    capacity_ranges = ["7.1 to 8 kg"]
                elif capacity <= 9:
                    capacity_ranges = ["8.1 to 9 kg"]
                elif capacity <= 10:
                    capacity_ranges = ["9.1 to 10 kg"]
                elif capacity <= 11:
                    capacity_ranges = ["10.1 kg and Above", "11 kg"]
                else:
                    capacity_ranges = ["10.1 kg and Above", "11 kg"]
                
                filter_clauses.append({
                    "terms": {"attribute_capacity_wm_value": capacity_ranges}
                })
                logger.info(f"Capacity filter applied: {attributes['capacity_kg']}kg → ranges {capacity_ranges}")
            except ValueError:
                logger.warning(f"Invalid capacity value: {attributes['capacity_kg']}")
        
        # =================== REFRIGERATOR CAPACITY (LITRES) FILTER ===================
        # For queries like "201 to 300 L", "300L refrigerator"
        # ES values: "80 L and Below", "81 to 170 L", "171 to 200 L", 
        #            "201 to 300 L", "301 to 400 L", "401 to 500 L", "501 L and Above"
        if attributes.get("capacity_litres_range"):
            # User specified an exact range like "201 to 300 L" - use it directly
            litres_range = attributes["capacity_litres_range"]
            filter_clauses.append({
                "term": {"attribute_capacity_litres_value": litres_range}
            })
            logger.info(f"Litre capacity filter applied (exact range): {litres_range}")
        elif attributes.get("capacity_litres"):
            # User specified a single value like "300L" - map to appropriate range
            try:
                litres = float(attributes["capacity_litres"])
                litres_ranges = []
                if litres <= 80:
                    litres_ranges = ["80 L and Below"]
                elif litres <= 170:
                    litres_ranges = ["81 to 170 L"]
                elif litres <= 200:
                    litres_ranges = ["171 to 200 L"]
                elif litres <= 300:
                    litres_ranges = ["201 to 300 L"]
                elif litres <= 400:
                    litres_ranges = ["301 to 400 L"]
                elif litres <= 500:
                    litres_ranges = ["401 to 500 L"]
                else:
                    litres_ranges = ["501 L and Above"]
                
                filter_clauses.append({
                    "terms": {"attribute_capacity_litres_value": litres_ranges}
                })
                logger.info(f"Litre capacity filter applied: {litres}L → ranges {litres_ranges}")
            except ValueError:
                logger.warning(f"Invalid litre capacity value: {attributes['capacity_litres']}")
        
        if attributes.get("color"):
            # Color from NL query (e.g. "red fridge", "5 star red refrigerator"):
            # Use wildcard post_filter for hard filtering — matches "Red", "Burgundy Red", "Red Hilton" etc.
            # Also keep a should-clause boost so relevance ranking still prefers exact matches.
            color_val = attributes["color"]
            attributes["_color_post_filter"] = {
                "wildcard": {
                    "attribute_color_value": {
                        "value": f"*{color_val}*",
                        "case_insensitive": True
                    }
                }
            }
            should_clauses.append({
                "match": {
                    "attribute_color_value": color_val
                }
            })
        
        # =================== STAR/ENERGY RATING FILTER ===================
        # For queries like "5 star fridge", "3 star AC", "5 star washing machine"
        # Each category uses a DIFFERENT ES attribute field and a different ID per star level.
        #
        # Category → (ES field, {star_number: attribute_id})
        # IDs verified from live index via aggregation queries (March 2026)
        STAR_RATING_MAP = {
            # Refrigerator: attribute_energy_rating_ref
            "refrigerators": (
                "attribute_energy_rating_ref",
                {1: "4822", 2: "4823", 3: "4824", 4: "4825", 5: "4826"}
            ),
            # Washing Machine: attribute_energy_efficiency
            "washing machines": (
                "attribute_energy_efficiency",
                {1: "27520", 2: "18059", 3: "18060", 4: "18061", 5: "18063"}
            ),
            # Air Conditioner: attribute_energy_rating
            "air conditioner": (
                "attribute_energy_rating",
                {1: "4751", 2: "4752", 3: "4757", 4: "4758", 5: "4759"}
            ),
            # Water Heater / Geyser: attribute_energy_rating
            "water heater and geysers": (
                "attribute_energy_rating",
                {1: "4751", 2: "4752", 3: "4757", 4: "4758", 5: "4759"}
            ),
            "geysers": (
                "attribute_energy_rating",
                {1: "4751", 2: "4752", 3: "4757", 4: "4758", 5: "4759"}
            ),
        }

        if attributes.get("star_rating"):
            star_num = int(attributes["star_rating"])
            cat_key = (category or "").lower()

            # Find matching category entry (normalise plural/singular)
            star_entry = STAR_RATING_MAP.get(cat_key)
            if star_entry is None:
                # Try common aliases
                _alias = {
                    "refrigerator": "refrigerators",
                    "fridge": "refrigerators",
                    "washing machine": "washing machines",
                    "ac": "air conditioner",
                    "air conditioners": "air conditioner",
                    "water heater": "water heater and geysers",
                    "geyser": "geysers",
                }
                star_entry = STAR_RATING_MAP.get(_alias.get(cat_key, ""))

            if star_entry:
                es_field, id_map = star_entry
                attr_id = id_map.get(star_num)
                if attr_id:
                    # Add to post_filter_clauses so it works alongside user-selected filters
                    # and respects the post_filter architecture (aggregations stay unaffected)
                    # We inject directly into post_filter_clauses below — store for now
                    attributes["_star_post_filter"] = {"terms": {es_field: [attr_id]}}
                    logger.info(
                        f"Star rating filter queued: {star_num} Star → {es_field}={attr_id} "
                        f"for category '{cat_key}'"
                    )
                else:
                    logger.warning(f"Star rating {star_num} not found in map for '{cat_key}'")
            else:
                logger.warning(
                    f"Star rating filter SKIPPED: no attribute mapping for category '{cat_key}'. "
                    f"Query asked for {star_num} Star but category has no known star field."
                )
        
        # =================== DOOR TYPE FILTER (REFRIGERATOR) ===================
        # For queries like "double door fridge", "single door refrigerator"
        # CHANGED: Use BOOST instead of filter so exact matches rank first but other refrigerators still show
        # This handles cases like "Haier 3 door fridge" where only 2 exact matches exist
        if attributes.get("door_type"):
            door_type_val = attributes["door_type"]
            # Store for boosting (not filtering)
            attributes["_door_type_boost"] = door_type_val
            # Special case: "Four Door" can be mapped as "Four Door" OR "Multi Door" in index
            if door_type_val == "Four Door":
                attributes["_door_type_boost_values"] = ["Four Door", "Multi Door"]
            elif door_type_val == "Triple Door":
                attributes["_door_type_boost_values"] = ["Triple Door", "Three Door"]
            else:
                attributes["_door_type_boost_values"] = [door_type_val]
            logger.info(f"Door type BOOST queued (not filter): {door_type_val}")

        # =================== WM FUNCTION TYPE FILTER ===================
        # For queries like "front load washing machine", "top load washer", "semi automatic WM"
        # Wildcard matches all ES variants:
        #   "Front Load" → "Front Load", "Fully Automatic Front Load", "Semi Automatic Front Load"
        #   "Top Load"   → "Top Load",   "Fully Automatic Top Load",   "Semi Automatic Top Load"
        #   "Semi Automatic" → "Semi Automatic", "Semi Automatic Top Load", "Semi Automatic Front Load"
        #   "Fully Automatic" → "Fully Automatic Front Load", "Fully Automatic Top Load"
        if attributes.get("wm_type"):
            wm_type_val = attributes["wm_type"].lower()  # e.g. "front load", "top load"
            attributes["_wm_type_post_filter"] = {
                "wildcard": {
                    "attribute_function_type_wm_value": {
                        "value": f"*{wm_type_val}*",
                        "case_insensitive": True
                    }
                }
            }
            logger.info(f"WM function type filter queued for post_filter: {attributes['wm_type']}")
        
        # NOTE: User filters are now applied via post_filter for multi-selection support
        # This allows aggregations to show ALL options, not just filtered ones
        # See build_query() where post_filter is added to es_query
        
        # CITY FILTER REMOVED - User requested to remove city-level filtering
        # Products will now be returned from global catalog regardless of city
        # City-specific pricing can still be selected from city_offers if available
        # if city_id:
        #     filter_clauses.append({
        #         "nested": {
        #             "path": "city_offers",
        #             "query": {
        #                 "bool": {
        #                     "should": [
        #                         {"term": {"city_offers.cityid": city_id}},
        #                         {"term": {"city_offers.cityid": "citi_id_0"}}
        #                     ],
        #                     "minimum_should_match": 1
        #                 }
        #             },
        #             # Return more inner_hits to get both city-specific and global offers
        #             "inner_hits": {
        #                 "size": 10,
        #                 "sort": [
        #                     # Sort so exact city_id match comes first
        #                     {
        #                         "_script": {
        #                             "type": "number",
        #                             "script": {
        #                                 "source": f"doc['city_offers.cityid'].value == params.target_city ? 0 : 1",
        #                                 "params": {"target_city": city_id}
        #                             },
        #                             "order": "asc"
        #                         }
        #                     }
        #                 ]
        #             }
        #         }
        #     })
        
        # EMI range filter
        if emi_range:
            emi_filter = {"nested": {
                "path": "city_offers",
                "query": {"range": {"city_offers.lowest_emi": {}}}
            }}
            if "gte" in emi_range:
                emi_filter["nested"]["query"]["range"]["city_offers.lowest_emi"]["gte"] = emi_range["gte"]
            if "lte" in emi_range:
                emi_filter["nested"]["query"]["range"]["city_offers.lowest_emi"]["lte"] = emi_range["lte"]
            filter_clauses.append(emi_filter)
        
        # =================== PRICE RANGE FILTER (NESTED) ===================
        # Price filter on city_offers.offer_price for city-specific pricing
        # Similar to EMI filter, this uses nested query for accurate city-based filtering
        if filters and ('price_max' in filters or 'price_range' in filters):
            price_range = filters.get('price_range', {})
            if 'price_max' in filters:
                price_range = {'lte': filters['price_max']}
            
            if price_range:
                price_filter = {
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "bool": {
                                "must": [
                                    {"range": {"city_offers.offer_price": {}}}
                                ]
                            }
                        }
                    }
                }
                # Build range query
                range_query = price_filter["nested"]["query"]["bool"]["must"][0]["range"]["city_offers.offer_price"]
                if "gte" in price_range:
                    range_query["gte"] = price_range["gte"]
                if "lte" in price_range:
                    range_query["lte"] = price_range["lte"]
                
                # Add city filter if city_id is provided
                if city_id:
                    price_filter["nested"]["query"]["bool"]["should"] = [
                        {"term": {"city_offers.cityid": city_id}},
                        {"term": {"city_offers.cityid": "citi_id_0"}}
                    ]
                    price_filter["nested"]["query"]["bool"]["minimum_should_match"] = 1
                
                filter_clauses.append(price_filter)
                logger.info(f"Added nested price filter: {price_range}")
        
        # =================== ZERO DOWN PAYMENT KEYWORD FILTER (BRD Requirement 2) ===================
        # If query contains "zero down payment" keyword, filter to show ONLY products with zero_dp_flag=1
        # Filter on user's city_id OR citi_id_0 to ensure displayed city_offer has zero_dp_flag=1
        detect_zero_dp = query_info.get("detect_zero_dp", False)
        if detect_zero_dp:
            if city_id:
                filter_clauses.append({
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"city_offers.zero_dp_flag": 1}}
                                ],
                                "should": [
                                    {"term": {"city_offers.cityid": city_id}},
                                    {"term": {"city_offers.cityid": "citi_id_0"}}
                                ],
                                "minimum_should_match": 1
                            }
                        }
                    }
                })
            else:
                filter_clauses.append({
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "term": {"city_offers.zero_dp_flag": 1}
                        }
                    }
                })
            logger.info("Applied zero_dp_flag=1 filter (BRD: zero down payment keyword detected)")
        
        # =================== NEW LAUNCH KEYWORD FILTER (BRD Requirement 2) ===================
        # If query contains "new launch" / "latest" keyword, filter to show ONLY new_launch products
        # Filter on user's city_id OR citi_id_0 to ensure displayed city_offer has new_launch_flag=1
        detect_new_launch = query_info.get("detect_new_launch", False)
        if detect_new_launch:
            if city_id:
                filter_clauses.append({
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"city_offers.new_launch_flag": 1}}
                                ],
                                "should": [
                                    {"term": {"city_offers.cityid": city_id}},
                                    {"term": {"city_offers.cityid": "citi_id_0"}}
                                ],
                                "minimum_should_match": 1
                            }
                        }
                    }
                })
            else:
                filter_clauses.append({
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "term": {"city_offers.new_launch_flag": 1}
                        }
                    }
                })
            logger.info("Applied new_launch_flag=1 filter (BRD: new launch keyword detected)")
        
        # =================== BEST SELLING KEYWORD FILTER (BRD Requirement) ===================
        # If query contains "best selling" / "top seller" keyword, filter to show ONLY top_seller products
        # IMPORTANT: We need to filter on top_seller_flag=1 for the user's city_id OR citi_id_0
        # This ensures the displayed city_offer (which is user's city or fallback to citi_id_0) has top_seller_flag=1
        detect_best_selling = query_info.get("detect_best_selling", False)
        if detect_best_selling:
            if city_id:
                # Filter on top_seller_flag=1 for user's city OR global (citi_id_0)
                filter_clauses.append({
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"city_offers.top_seller_flag": 1}}
                                ],
                                "should": [
                                    {"term": {"city_offers.cityid": city_id}},
                                    {"term": {"city_offers.cityid": "citi_id_0"}}
                                ],
                                "minimum_should_match": 1
                            }
                        }
                    }
                })
            else:
                # No city_id - just filter on top_seller_flag=1
                filter_clauses.append({
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "term": {"city_offers.top_seller_flag": 1}
                        }
                    }
                })
            logger.info("Applied top_seller_flag=1 filter (BRD: best selling keyword detected)")
        
        # =================== ONE EMI OFF KEYWORD FILTER (BRD Requirement) ===================
        # If query contains "one emi off" / "emi free" keyword, filter to show ONLY one_emi_off products
        detect_one_emi_off = query_info.get("detect_one_emi_off", False)
        if detect_one_emi_off:
            if city_id:
                # Filter on one_emi_off=1 for user's city OR global (citi_id_0)
                filter_clauses.append({
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"city_offers.one_emi_off": 1}}
                                ],
                                "should": [
                                    {"term": {"city_offers.cityid": city_id}},
                                    {"term": {"city_offers.cityid": "citi_id_0"}}
                                ],
                                "minimum_should_match": 1
                            }
                        }
                    }
                })
            else:
                # No city_id - just filter on one_emi_off=1
                filter_clauses.append({
                    "nested": {
                        "path": "city_offers",
                        "query": {
                            "term": {"city_offers.one_emi_off": 1}
                        }
                    }
                })
            logger.info("Applied one_emi_off=1 filter (BRD: one emi off keyword detected)")
        
        # =================== AC TYPE FILTER (window ac, split ac, etc.) ===================
        # If user searches for specific AC type, filter by attribute_ac_type_value
        ac_type_filter = query_info.get("ac_type_filter")
        if ac_type_filter:
            # Filter to show only ACs of the specified type
            # Use should clause to match exact type OR inverter variant
            if ac_type_filter == "Window":
                # Window AC: match "Window" or "Inverter Window"
                filter_clauses.append({
                    "bool": {
                        "should": [
                            {"term": {"attribute_ac_type_value": "Window"}},
                            {"term": {"attribute_ac_type_value": "Inverter Window"}}
                        ],
                        "minimum_should_match": 1
                    }
                })
            elif ac_type_filter == "Split":
                # Split AC: match "Split" or "Inverter Split" or "Dual Inverter Split"
                filter_clauses.append({
                    "bool": {
                        "should": [
                            {"term": {"attribute_ac_type_value": "Split"}},
                            {"term": {"attribute_ac_type_value": "Inverter Split"}},
                            {"term": {"attribute_ac_type_value": "Dual Inverter Split"}}
                        ],
                        "minimum_should_match": 1
                    }
                })
            else:
                # Exact match for other types
                filter_clauses.append({"term": {"attribute_ac_type_value": ac_type_filter}})
            logger.info(f"Applied AC type filter: attribute_ac_type_value='{ac_type_filter}'")
        
        # =================== ANDROID TABLET: EXCLUDE APPLE ===================
        # When user searches "android tablet", "android tab", exclude Apple products (iPad)
        # Apple uses iPadOS, not Android - so showing iPads is incorrect
        original_lower_for_android = query_info.get("original", "").lower()
        if "android" in original_lower_for_android and category == "tablets":
            filter_clauses.append({
                "bool": {
                    "must_not": [
                        {"match": {"manufacturer_desc": "Apple"}},
                        {"match": {"attribute_brand_new_value": "Apple"}}
                    ]
                }
            })
            logger.info("Android tablet query - excluding Apple products (iPad)")
        
        # Build final query
        bool_query = {}
        if must_clauses:
            bool_query["must"] = must_clauses
        if should_clauses:
            bool_query["should"] = should_clauses
        if filter_clauses:
            bool_query["filter"] = filter_clauses
        
        if not bool_query:
            bool_query = {"must": [{"match_all": {}}]}
        
        # Wrap in function_score for phone brand boosting
        # ONLY for scoring-enabled categories
        final_query = {"bool": bool_query}
        
        if use_scoring and phone_brand_boost and not is_apple_query:
            # Use function_score to multiply scores for target brands
            functions = []
            for boost_brand in phone_brand_boost:
                functions.append({
                    "filter": {"match": {"manufacturer_desc": boost_brand}},
                    "weight": 50  # High weight for target brands
                })
            # Penalize Apple for generic phone queries
            functions.append({
                "filter": {"match": {"manufacturer_desc": "Apple"}},
                "weight": 0.1  # Very low weight for Apple
            })
            
            final_query = {
                "function_score": {
                    "query": {"bool": bool_query},
                    "functions": functions,
                    "score_mode": "multiply",
                    "boost_mode": "multiply"
                }
            }
        
        # =================== BRAND-ONLY PHONE QUERY SMARTPHONE BOOST ===================
        # For queries like "samsung", "oppo", "vivo" etc. boost smartphones to appear first
        # but still show other products (watches, tablets, TVs) after phones
        # ONLY for scoring-enabled categories
        if use_scoring and is_brand_only_phone_query and not is_apple_query and not phone_brand_boost:
            # Boost smartphones/mobiles category and penalize watches
            functions = [
                # Boost smartphones category (highest priority)
                {
                    "filter": {"bool": {
                        "should": [
                            {"term": {"categoryName.keyword": "Smartphones"}},
                            {"match": {"categoryName": "smartphone"}},
                            {"match": {"sub_category_name": "mobile"}},
                            {"match": {"sub_category_name": "smartphone"}},
                            {"match": {"product_name": "galaxy phone"}},
                            {"match": {"product_name": "mobile phone"}},
                            {"match": {"product_name": "smartphone"}},
                        ],
                        "minimum_should_match": 1
                    }},
                    "weight": 500  # Very high weight for smartphones
                },
                # Penalize watches/wearables - they should appear after phones
                {
                    "filter": {"bool": {
                        "should": [
                            {"match": {"categoryName": "watch"}},
                            {"match": {"categoryName": "wearable"}},
                            {"match": {"sub_category_name": "watch"}},
                            {"match": {"sub_category_name": "wearable"}},
                            {"match": {"product_name": "watch"}},
                            {"match": {"product_name": "smartwatch"}},
                        ],
                        "minimum_should_match": 1
                    }},
                    "weight": 0.1  # Low weight for watches
                },
            ]
            
            final_query = {
                "function_score": {
                    "query": final_query,
                    "functions": functions,
                    "score_mode": "multiply",
                    "boost_mode": "multiply"
                }
            }
            logger.info(f"Applied smartphone boost for brand-only phone query")
        
        # Also apply function_score for product boost (bed, fan, etc)
        # SKIP for non-scoring categories (furniture, mattress, etc.)
        if use_scoring and product_boost.get("boost_type") and not is_apple_query and not phone_brand_boost:
            boost_terms = product_boost.get("boost_terms", [])
            functions = []
            for term in boost_terms:
                functions.append({
                    "filter": {"match": {"product_name": term}},
                    "weight": 10
                })
            if functions:
                final_query = {
                    "function_score": {
                        "query": final_query,
                        "functions": functions,
                        "score_mode": "sum",
                        "boost_mode": "multiply"
                    }
                }
        
        # =================== AUDIO DEVICE TYPE BOOST ===================
        # For audio device queries (earbuds/earphone/headphone/neckband/airdopes/tws),
        # boost matching device type products and strongly demote soundbars/speakers.
        # This prevents soundbars from dominating when user explicitly wants earbuds.
        # Includes typo variants so "boat erphn", "samsung earfon" etc. are also boosted.
        original_query = query_info.get("original", "").lower()
        _oq_words = set(original_query.split())
        
        # Define audio device type -> boost/demote mapping
        _audio_device_boost = None
        _earbuds_kw = {"earbuds", "earbud", "earbudz", "earbudss", "erbud", "erbuds", "airbud", "airbuds", "earbds", "earbd",
                        "airdopes", "airdope", "tws"}
        _earphone_kw = {"earphone", "earphones", "earphn", "earphon", "earfon", "earfone", "earpone",
                        "erphn", "erpone", "erphone", "erfone", "erfon", "earphne", "eaphone", "earpho",
                        "raphon", "raphne", "raphone", "raphn", "airphon", "airphone", "airphn", "airfon", "airfone"}
        _headphone_kw = {"headphone", "headphones", "headfon", "headfone", "hedphone", "headphn",
                         "haedphone", "headphne", "headphon", "headset", "headsets"}
        _neckband_kw = {"neckband", "neckbands", "neckbnd", "nekband", "neckbad", "nckband", "necband"}
        
        if _oq_words & _earbuds_kw or "ear bud" in original_query:
            _audio_device_boost = {
                "boost": [
                    ("earbuds", 20), ("buds", 15), ("tws", 15), ("truly wireless", 10), ("wireless earbuds", 10),
                ],
                "demote": [
                    ("soundbar", 0.05), ("sound bar", 0.05), ("speaker", 0.1), ("party", 0.1),
                    ("home theater", 0.05), ("home theatre", 0.05),
                    ("headphone", 0.3), ("over ear", 0.2), ("on ear", 0.2),
                ],
            }
        elif _oq_words & _earphone_kw or "ear phone" in original_query:
            _audio_device_boost = {
                "boost": [
                    ("earbuds", 15), ("earphone", 20), ("buds", 10), ("tws", 15),
                    ("truly wireless", 10), ("wireless earphone", 15), ("neckband", 8),
                ],
                "demote": [
                    ("soundbar", 0.05), ("sound bar", 0.05), ("speaker", 0.1), ("party", 0.1),
                    ("home theater", 0.05), ("home theatre", 0.05),
                ],
            }
        elif _oq_words & _headphone_kw:
            _audio_device_boost = {
                "boost": [
                    ("headphone", 20), ("headset", 15), ("over ear", 10), ("on ear", 10),
                    ("earbuds", 5), ("buds", 5),
                ],
                "demote": [
                    ("soundbar", 0.05), ("sound bar", 0.05), ("speaker", 0.1), ("party", 0.1),
                    ("home theater", 0.05), ("home theatre", 0.05),
                ],
            }
        elif _oq_words & _neckband_kw or "neck band" in original_query:
            _audio_device_boost = {
                "boost": [
                    ("neckband", 20), ("wireless earphone", 15), ("bluetooth earphone", 10),
                    ("earbuds", 5), ("tws", 5),
                ],
                "demote": [
                    ("soundbar", 0.05), ("sound bar", 0.05), ("speaker", 0.1), ("party", 0.1),
                    ("home theater", 0.05), ("home theatre", 0.05),
                    ("over ear", 0.3),
                ],
            }
        
        if _audio_device_boost:
            audio_functions = []
            for term, weight in _audio_device_boost["boost"]:
                audio_functions.append({"filter": {"match": {"product_name": term}}, "weight": weight})
            for term, weight in _audio_device_boost["demote"]:
                audio_functions.append({"filter": {"match": {"product_name": term}}, "weight": weight})
            
            # If brand boost is requested (audio device + brand query like "Boat Headphones"),
            # add brand boost to audio functions so Boat products rank higher but don't exclude others
            if _use_brand_boost_instead and brand:
                audio_functions.append({
                    "filter": {"match": {"manufacturer_desc": brand.title()}},
                    "weight": 50  # Strong boost for matching brand
                })
                logger.info(f"Brand boost added for '{brand}' in audio device query")
            
            final_query = {
                "function_score": {
                    "query": final_query,
                    "functions": audio_functions,
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            }
            logger.info("Audio device boost applied - prioritizing device type")
        
        # =================== MODEL SUFFIX BOOST (Pro/Plus/Ultra/FE) ===================
        # When user searches for a specific model variant like "X200 Pro", boost exact suffix
        # match and penalize other suffixes (e.g., FE, Lite) to push them down
        # Skip for MacBook queries — MacBook product lines (Neo, Air, Pro) are distinct products
        # But DO apply for iPhone/iPad queries where Pro/Plus/Ultra ARE model variants
        # REDUCED: When specific model number is present (V60, X200, etc.), reduce suffix boost
        # to prevent V40 Pro ranking higher than V60 when user searches "V60 Pro"
        is_macbook_query = is_apple_query and any(kw in original_query for kw in ["macbook", "mac book", "mackbook"])
        
        # Detect if query has specific model number (V60, X200, A55, etc.)
        has_model_number = bool(re.search(r'\b[vxyt]\d{2,3}[a-z]?\b', original_query, re.IGNORECASE))
        
        if not is_macbook_query:
            model_suffix_map = {
                "pro": {"boost": ["pro"], "penalize": ["fe", "lite"]},
                "ultra": {"boost": ["ultra"], "penalize": ["fe", "lite"]},
                "plus": {"boost": ["plus"], "penalize": ["fe", "lite"]},
                "fe": {"boost": ["fe"], "penalize": ["pro", "ultra", "plus"]},
                "lite": {"boost": ["lite"], "penalize": ["pro", "ultra"]},
            }
            query_lower_words = original_query.split()
            for suffix, config in model_suffix_map.items():
                if suffix in query_lower_words:
                    suffix_functions = []
                    # REDUCED weight when model number present - model number should be primary match
                    suffix_weight = 5 if has_model_number else 50
                    for boost_term in config["boost"]:
                        suffix_functions.append({
                            "filter": {"match_phrase": {"product_name": boost_term}},
                            "weight": suffix_weight
                        })
                    for penalize_term in config["penalize"]:
                        suffix_functions.append({
                            "filter": {"match_phrase": {"product_name": penalize_term}},
                            "weight": 0.05
                        })
                    final_query = {
                        "function_score": {
                            "query": final_query,
                            "functions": suffix_functions,
                            "score_mode": "multiply",
                            "boost_mode": "multiply"
                        }
                    }
                    logger.info(f"Model suffix boost applied for '{suffix}' (weight={suffix_weight}) - boosting {config['boost']}, penalizing {config['penalize']}")
                    break  # Only apply one suffix boost
            
            # Add stronger model number boost when specific model is in query
            # This ensures V60 ranks above V40 Pro when searching "V60 Pro"
            if has_model_number:
                model_match = re.search(r'\b([vxyt])(\d{2,3}[a-z]?)\b', original_query, re.IGNORECASE)
                if model_match:
                    model_num = f"{model_match.group(1)}{model_match.group(2)}".lower()
                    final_query = {
                        "function_score": {
                            "query": final_query,
                            "functions": [
                                {"filter": {"match": {"product_name": model_num}}, "weight": 100},
                            ],
                            "score_mode": "sum",
                            "boost_mode": "multiply"
                        }
                    }
                    logger.info(f"Model number function_score boost for '{model_num}' (weight=100)")
        
        # NOTE: Google Pixel model boosting is done via sort script, not function_score
        # See GOOGLE PIXEL MODEL SORT section below in the sorting logic
        
        # =================== ELECTRIC SCOOTER BOOST ===================
        # For electric scooter queries, boost electric vehicles
        electric_keywords = ["electric scooter", "electric scooty", "ev scooter", "e scooter",
                            "electric two wheeler", "electric bike", "e bike", "ev bike", "electric"]
        is_electric_query = any(kw in original_query for kw in electric_keywords)
        if is_electric_query and category == "two wheeler":
            electric_functions = [
                # Boost electric vehicles
                {"filter": {"match": {"product_name": "electric"}}, "weight": 30},
                {"filter": {"match": {"asset_category_name": "Electric"}}, "weight": 50},
                {"filter": {"match": {"product_name": "ev"}}, "weight": 25},
                {"filter": {"match": {"product_name": "ather"}}, "weight": 20},
                {"filter": {"match": {"product_name": "ola"}}, "weight": 20},
                {"filter": {"match": {"product_name": "revolt"}}, "weight": 20},
                {"filter": {"match": {"product_name": "okinawa"}}, "weight": 20},
                {"filter": {"match": {"product_name": "chetak"}}, "weight": 20},
                # Demote petrol vehicles
                {"filter": {"match": {"product_name": "cc"}}, "weight": 0.3},
                {"filter": {"match": {"product_name": "petrol"}}, "weight": 0.2},
            ]
            final_query = {
                "function_score": {
                    "query": final_query,
                    "functions": electric_functions,
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            }
            logger.info("Electric scooter boost applied - prioritizing EVs")
        
        # =================== WINDOW AC / SPLIT AC TYPE BOOST ===================
        # For "window ac" queries, boost window ACs over split ACs
        # For "split ac" queries, boost split ACs over window ACs
        if category == "air conditioner":
            ac_type_functions = []
            if "window" in original_query:
                ac_type_functions = [
                    {"filter": {"match": {"product_name": "window"}}, "weight": 50},
                    {"filter": {"match": {"product_name": "Window"}}, "weight": 50},
                    # Demote split ACs
                    {"filter": {"match": {"product_name": "split"}}, "weight": 0.1},
                ]
            elif "split" in original_query:
                ac_type_functions = [
                    {"filter": {"match": {"product_name": "split"}}, "weight": 50},
                    {"filter": {"match": {"product_name": "Split"}}, "weight": 50},
                    # Demote window ACs
                    {"filter": {"match": {"product_name": "window"}}, "weight": 0.1},
                ]
            if ac_type_functions:
                final_query = {
                    "function_score": {
                        "query": final_query,
                        "functions": ac_type_functions,
                        "score_mode": "sum",
                        "boost_mode": "multiply"
                    }
                }
                logger.info(f"AC type boost applied for: {original_query}")
        
        # =================== JUICER / MIXER BOOST ===================
        # For "juicer mixer", "mixer juicer", "mixer grinder" queries,
        # boost mixer/juicer/grinder products and demote water purifiers
        if category == "kitchen appliances":
            mixer_keywords = {"juicer", "mixer", "grinder", "blender", "mixi", "mixie"}
            if mixer_keywords & set(original_query.split()):
                mixer_functions = [
                    {"filter": {"match": {"product_name": "mixer"}}, "weight": 40},
                    {"filter": {"match": {"product_name": "grinder"}}, "weight": 40},
                    {"filter": {"match": {"product_name": "juicer"}}, "weight": 40},
                    {"filter": {"match": {"product_name": "blender"}}, "weight": 30},
                    {"filter": {"match": {"product_name": "food processor"}}, "weight": 20},
                    # Demote water purifiers (they shouldn't appear for mixer queries)
                    {"filter": {"match": {"product_name": "purifier"}}, "weight": 0.05},
                    {"filter": {"match": {"product_name": "water purifier"}}, "weight": 0.01},
                    {"filter": {"match": {"product_name": "RO"}}, "weight": 0.05},
                ]
                final_query = {
                    "function_score": {
                        "query": final_query,
                        "functions": mixer_functions,
                        "score_mode": "sum",
                        "boost_mode": "multiply"
                    }
                }
                logger.info("Mixer/juicer boost applied - demoting water purifiers")
        
        # =================== NEO PHONE BOOST ===================
        # For "neo" queries (without "macbook"), boost realme GT NEO phones
        # "neo" is commonly searched for realme GT NEO series, but MacBook Neo also exists
        # When user searches just "neo" or "gt neo", they likely want phones
        # When they search "macbook neo", they want the laptop
        # IMPORTANT: Do NOT apply for TV queries - "Neo QLED TV" is Samsung's TV series, not phones!
        query_words = set(original_query.split())
        tv_keywords_in_query = {"tv", "television", "qled", "oled", "led", "smart tv", "inch", "inches"}
        is_tv_query = bool(tv_keywords_in_query & query_words) or "tv" in original_query
        if "neo" in query_words and "macbook" not in original_query and "mac book" not in original_query and not is_tv_query:
            neo_functions = [
                # Boost realme GT NEO phones
                {"filter": {"match": {"product_name": "GT NEO"}}, "weight": 100},
                {"filter": {"match": {"product_name": "gt neo"}}, "weight": 100},
                {"filter": {"match": {"manufacturer_desc": "Realme"}}, "weight": 50},
                {"filter": {"match": {"manufacturer_desc": "realme"}}, "weight": 50},
                {"filter": {"term": {"asset_category_name": "Mobiles"}}, "weight": 30},
                # Don't completely demote MacBook Neo, just lower its priority
                {"filter": {"match": {"product_name": "MacBook"}}, "weight": 0.5},
            ]
            final_query = {
                "function_score": {
                    "query": final_query,
                    "functions": neo_functions,
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            }
            logger.info("Neo phone boost applied - prioritizing realme GT NEO over MacBook Neo")
        
        # =================== DOOR TYPE BOOST (REFRIGERATOR) ===================
        # For "3 door fridge", "double door refrigerator", etc.
        # Use BOOST instead of filter so exact matches rank first but other refrigerators still appear
        # Handles cases like "Haier 3 door fridge" where only 2 exact matches exist
        if attributes.get("_door_type_boost"):
            door_type_values = attributes.get("_door_type_boost_values", [attributes["_door_type_boost"]])
            door_type_functions = []
            for dt_val in door_type_values:
                door_type_functions.append({
                    "filter": {"term": {"attribute_door_type_value": dt_val}},
                    "weight": 80  # High weight to ensure door type matches rank first
                })
            if door_type_functions:
                final_query = {
                    "function_score": {
                        "query": final_query,
                        "functions": door_type_functions,
                        "score_mode": "sum",  # Sum all matching door types
                        "boost_mode": "multiply"  # Multiply with base score
                    }
                }
                logger.info(f"Door type BOOST applied for: {door_type_values} (weight=80)")
        
        # Build post_filter for user-selected filters (multi-selection support)
        # post_filter applies AFTER aggregations, so all filter options remain visible
        post_filter_clauses = []

        # Inject query-intent derived filters (star rating, door type) as post_filter
        # so they work consistently with explicit user attribute filters
        if attributes.get("_star_post_filter"):
            post_filter_clauses.append(attributes["_star_post_filter"])
            logger.info(f"Star rating injected into post_filter: {attributes['_star_post_filter']}")
        # CHANGED: Door type now uses BOOST (not filter) so other products still appear
        # if attributes.get("_door_type_post_filter"):
        #     post_filter_clauses.append(attributes["_door_type_post_filter"])
        #     logger.info(f"Door type injected into post_filter: {attributes['_door_type_post_filter']}")
        if attributes.get("_color_post_filter"):
            post_filter_clauses.append(attributes["_color_post_filter"])
            logger.info(f"Color injected into post_filter: {attributes['_color_post_filter']}")
        if attributes.get("_wm_type_post_filter"):
            post_filter_clauses.append(attributes["_wm_type_post_filter"])
            logger.info(f"WM function type injected into post_filter: {attributes['_wm_type_post_filter']}")

        if filters:
            for field, value in filters.items():
                if not value:
                    continue
                # Skip EMI filter - handled separately
                if field == "emi":
                    continue
                # Skip price_range filter - already applied as range filter in main query
                if field == "price_range":
                    continue
                # Skip price_max filter - already applied as range filter in main query
                if field == "price_max":
                    continue
                # Handle comma-separated values
                if isinstance(value, str):
                    values = [v.strip() for v in value.split(",") if v.strip()]
                elif isinstance(value, list):
                    values = value
                else:
                    values = [str(value)]
                
                if values:
                    post_filter_clauses.append({
                        "terms": {field: values}
                    })
                    logger.info(f"Applied post_filter: {field} = {values}")
        
        # Calculate "from" offset - use from_offset if provided, otherwise calculate from page
        calculated_from = from_offset if from_offset is not None else (page - 1) * page_size
        
        # Build post_filter dict for passing to aggregations (correct cardinality count)
        # When attribute filters are active, cardinality must be wrapped in matching filter agg
        pf_for_agg = None
        if post_filter_clauses:
            if len(post_filter_clauses) == 1:
                pf_for_agg = post_filter_clauses[0]
            else:
                pf_for_agg = {"bool": {"must": post_filter_clauses}}
        
        es_query = {
            "query": final_query,
            "from": calculated_from,
            "size": page_size,
            "track_total_hits": True,  # Get accurate total count after collapse
            "collapse": {
                "field": "modelid",
                "inner_hits": {
                    "name": "sku_variants",
                    "size": 5  # Reduced from 10 for performance (UI rarely shows >5 variants)
                }
            }
        }
        
        # PERFORMANCE: Only add aggregations for page 1 (filter counts needed)
        # Skip for pagination requests to reduce query complexity
        if not skip_aggregations:
            es_query["aggs"] = self._build_aggregations(city_id, pf_for_agg)
        
        # =================== SORTING ===================
        # Supported sort options (in order):
        # 1. "relevance" (default) - Best match based on query
        # 2. "low_to_high" - Price/EMI lowest to highest
        # 3. "high_to_low" - Price/EMI highest to lowest
        # 4. "latest_launch" - Newest products first (new_launch_flag)
        # 5. "best_selling" - Top sellers first (top_seller_flag, transaction_count)
        # 6. "most_viewed" - Most viewed first (most_viewed_flag, pdp_view_count)
        # 7. "discounts" - Highest discount first (off_percentage)
        #
        # CITY PRIORITY: In all sort cases, products for user's city_id are shown first,
        # then products with city_id=0 (fallback/generic)
        sort_by_lower = (sort_by or "").lower().strip()
        
        # =================== FAST PAGINATION SORT (page 2+) ===================
        # For pagination, skip expensive script-based sorts that iterate city_offers via params._source
        # The city filtering is already done in the query filter, so we need only a stable ordering
        # This saves significant CPU time per document (avoids multiple city_offers iterations)
        if skip_expensive_sorts:
            # Use simpler nested sort for EMI-based sorts, skip city priority script entirely
            if sort_by_lower in ("low_to_high", "lowtohigh", "asc", "ascending", "emi_low_high", "price_low_high"):
                es_query["sort"] = [
                    {
                        "city_offers.lowest_emi": {
                            "order": "asc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "min"
                        }
                    },
                    {"modelid": {"order": "asc"}}
                ]
            elif sort_by_lower in ("high_to_low", "hightolow", "desc", "descending", "emi_high_low", "price_high_low"):
                es_query["sort"] = [
                    {
                        "city_offers.lowest_emi": {
                            "order": "desc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "max"
                        }
                    },
                    {"modelid": {"order": "asc"}}
                ]
            else:
                # For relevance and other sorts, use simple score + model sort
                es_query["sort"] = [
                    {"_score": {"order": "desc"}},
                    {"model_launch_date": {"order": "desc"}},
                    {"modelid": {"order": "asc"}}
                ]
            logger.debug(f"Fast pagination sort applied (skip_expensive_sorts=True, sort_by={sort_by_lower})")
            # Skip rest of sort logic - we've already set the sort
        
        else:
            # =================== NORMAL SORTING (page 1 or explicit sort requests) ===================
            # Helper function to create city-prioritized sort
            def _build_city_priority_sort(city_id_val):
                """
                Creates a script sort that prioritizes products matching user's city_id.
                Products matching user's city first (0), then city_id=0 products (1), then others (2)
                """
                clean_city = city_id_val.replace('citi_id_', '') if city_id_val else '0'
                return {
                    "_script": {
                        "type": "number",
                        "script": {
                            "lang": "painless",
                            "source": f"""
                                // Priority: user's city (0) > city_id_0 (1) > others (2)
                                def userCity = 'citi_id_{clean_city}';
                                def fallbackCity = 'citi_id_0';
                                
                                if (params._source != null && params._source.containsKey('city_offers')) {{
                                    def offers = params._source.get('city_offers');
                                    if (offers != null) {{
                                        for (def offer : offers) {{
                                            if (offer.containsKey('cityid') && offer.get('cityid').toString() == userCity) {{
                                                return 0;  // User's city - highest priority
                                            }}
                                        }}
                                        for (def offer : offers) {{
                                            if (offer.containsKey('cityid') && offer.get('cityid').toString() == fallbackCity) {{
                                                return 1;  // Fallback city
                                            }}
                                        }}
                                    }}
                                }}
                                return 2;  // No matching city
                            """
                        },
                        "order": "asc"
                    }
                }
            
            if sort_by_lower in ("low_to_high", "lowtohigh", "asc", "ascending", "emi_low_high", "price_low_high"):
                # Sort by lowest EMI ascending (requires nested sort for city_offers)
                # City priority: user's city first, then city_id=0
                es_query["sort"] = [
                    _build_city_priority_sort(city_id),  # City priority first
                    {
                        "city_offers.lowest_emi": {
                            "order": "asc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "min"
                        }
                    },
                    {"modelid": {"order": "asc"}}
                ]
                logger.info("Sorting: Low to High (EMI ascending) with city priority")
                
            elif sort_by_lower in ("high_to_low", "hightolow", "desc", "descending", "emi_high_low", "price_high_low"):
                # Sort by lowest EMI descending
                # City priority: user's city first, then city_id=0
                es_query["sort"] = [
                    _build_city_priority_sort(city_id),  # City priority first
                    {
                        "city_offers.lowest_emi": {
                            "order": "desc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "max"
                        }
                    },
                    {"modelid": {"order": "asc"}}
                ]
                logger.info("Sorting: High to Low (EMI descending) with city priority")
            
            elif sort_by_lower in ("latest_launch", "latestlaunch", "latest", "newest", "new_launch"):
                # Sort by latest launch - products with new_launch_flag first, then by launch date
                # City priority: user's city first, then city_id=0
                es_query["sort"] = [
                    _build_city_priority_sort(city_id),  # City priority first
                    {
                        "city_offers.new_launch_flag": {
                            "order": "desc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "max"
                        }
                    },
                    {"model_launch_date": {"order": "desc"}},  # Then by launch date
                    {"modelid": {"order": "asc"}}
                ]
                logger.info("Sorting: Latest Launch (new_launch_flag + launch_date) with city priority")
            
            elif sort_by_lower in ("best_selling", "bestselling", "best_seller", "top_seller", "topseller"):
                # Sort by best selling - products with top_seller_flag first, then by transaction_count
                # City priority: user's city first, then city_id=0
                es_query["sort"] = [
                    _build_city_priority_sort(city_id),  # City priority first
                    {
                        "city_offers.top_seller_flag": {
                            "order": "desc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "max"
                        }
                    },
                    {
                        "city_offers.transaction_count": {
                            "order": "desc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "max",
                            "missing": 0
                        }
                    },
                    {"modelid": {"order": "asc"}}
                ]
                logger.info("Sorting: Best Selling (top_seller_flag + transaction_count) with city priority")
            
            elif sort_by_lower in ("most_viewed", "mostviewed", "popular", "trending"):
                # Sort by most viewed - products with most_viewed_flag first, then by pdp_view_count
                # City priority: user's city first, then city_id=0
                es_query["sort"] = [
                    _build_city_priority_sort(city_id),  # City priority first
                    {
                        "city_offers.most_viewed_flag": {
                            "order": "desc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "max"
                        }
                    },
                    {
                        "city_offers.pdp_view_count": {
                            "order": "desc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "max",
                            "missing": 0
                        }
                    },
                    {"modelid": {"order": "asc"}}
                ]
                logger.info("Sorting: Most Viewed (most_viewed_flag + pdp_view_count) with city priority")
            
            elif sort_by_lower in ("discounts", "discount", "offers", "deal", "deals"):
                # Sort by discount percentage - highest discount first
                # City priority: user's city first, then city_id=0
                es_query["sort"] = [
                    _build_city_priority_sort(city_id),  # City priority first
                    {
                        "city_offers.off_percentage": {
                            "order": "desc",
                            "nested": {
                                "path": "city_offers",
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id or "citi_id_0"}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            },
                            "mode": "max",
                            "missing": 0
                        }
                    },
                    {"modelid": {"order": "asc"}}
                ]
                logger.info("Sorting: Discounts (off_percentage descending) with city priority")
                
            else:
                # Default: relevance (score) with secondary priority for flagged products
                # Applies when sort_by is "relevance", None, or any unrecognized value
                # CHANGE: Score is now TOP priority, then city priority, then flags
                # E.g., "oppo k13" should return "K13" before "K13x" even if K13x has new_launch_flag
                # City priority: user's city products first, then city_id=0 products
                
                # =================== GOOGLE PIXEL MODEL SORT ===================
                # For Google Pixel queries with model number (e.g., "pixel 8"), use custom sort
                # since function_score/wildcard don't work due to ES analyzer converting "pixel" to "smartphone"
                if is_google_pixel_query and pixel_model_number:
                    es_query["sort"] = [
                        _build_city_priority_sort(city_id),  # City priority first
                        {
                            "_script": {
                                "type": "number",
                                "script": {
                                    "lang": "painless",
                                    "source": f"""
                                        // Check if search_field contains the target model (e.g., "pixel 8")
                                        String searchField = '';
                                        if (params._source != null && params._source.containsKey('search_field')) {{
                                            searchField = params._source.get('search_field').toString().toLowerCase();
                                        }}
                                        if (searchField.contains('pixel {pixel_model_number}')) {{
                                            return 0;  // Sort first (matching model)
                                        }}
                                        return 1;  // Sort second (non-matching)
                                    """
                                },
                                "order": "asc"
                            }
                        },
                        {"model_launch_date": {"order": "desc"}},
                        {"modelid": {"order": "asc"}}
                    ]
                    logger.info(f"Sorting: Google Pixel model {pixel_model_number} priority (script sort) with city priority")
                
                else:
                    # =================== MODEL-SPECIFIC QUERY SORT ===================
                    # For queries like "vivo t4x", "samsung s24", "galaxy z fold", prioritize text relevance over city/score
                    # This ensures exact model matches (only in citi_id_0) appear above generic matches (in user's city)
                    # Also handles model patterns without digits like "z fold", "z flip", "mac mini", "ideacentre"
                    MODEL_PATTERNS_NO_DIGITS = ['z fold', 'z flip', 'mac mini', 'mac studio', 'ideacentre', 'ideapad', 'omen', 'legion', 'omnidesk', 'surface']
                    processed_query_lower = query_info.get("processed", "").lower()
                    has_model_digits = re.search(r'\d', re.sub(r'\b5g\b|\b4g\b|\b3g\b', '', processed_query_lower))
                    has_model_pattern = any(pattern in processed_query_lower for pattern in MODEL_PATTERNS_NO_DIGITS)
                    
                    # =================== NON-SCORING CATEGORIES: TEXT RELEVANCE FIRST ===================
                    # For furniture, mattress, and other non-scoring categories:
                    # Text relevance (exact match) should rank ABOVE city priority
                    # This ensures "bed" query shows beds first, not sofas from user's city
                    if not use_scoring:
                        es_query["sort"] = [
                            {"_score": {"order": "desc"}},  # Text relevance FIRST for furniture etc.
                            _build_city_priority_sort(city_id),  # Then city priority
                            {"model_launch_date": {"order": "desc"}},
                            {"modelid": {"order": "asc"}}
                        ]
                        logger.info(f"Sorting: Non-scoring category '{category}' (text relevance first, then city priority)")
                    
                    elif query_info.get("processed") and (has_model_digits or has_model_pattern):
                        # Model-specific query detected (contains alphanumeric model like t4x, s24, v70, z fold)
                        es_query["sort"] = [
                            {"_score": {"order": "desc"}},  # Text relevance FIRST for model queries
                            _build_city_priority_sort(city_id),  # Then city priority
                            {"model_launch_date": {"order": "desc"}},
                            {"modelid": {"order": "asc"}}
                        ]
                        logger.info(f"Sorting: Model-specific query (text relevance first, then city priority)")
                
                    else:
                        # =================== BRD-COMPLIANT RELEVANCE SORT ===================
                        # Per Business Requirements:
                        # 1. First 2 SKUs should have new_launch_flag=true, ordered by score (feed score)
                        # 2. After first 2, order by score (highest to lowest)
                        # 3. City priority: user's city_id first, then city_id=0
                        # 4. One SKU per model_id (handled by collapse)
                        #
                        # Implementation: ES sorts by city_priority + new_launch_flag + business_score
                        # Post-processing in search() ensures exactly first 2 are new_launch products
                        es_query["sort"] = [
                            _build_city_priority_sort(city_id),  # City priority: user's city first
                            {
                                "_script": {
                                    "type": "number",
                                    "script": {
                                        "lang": "painless",
                                        "source": """
                                            // BRD: new_launch products should appear first (within city priority)
                                            // Check if product has new_launch_flag=true in city_offers
                                            def hasNewLaunch = false;
                                            if (doc.containsKey('city_offers.new_launch_flag') && 
                                                doc['city_offers.new_launch_flag'].size() > 0) {
                                                for (int i = 0; i < doc['city_offers.new_launch_flag'].length; i++) {
                                                    if (doc['city_offers.new_launch_flag'][i] == true) {
                                                        hasNewLaunch = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            return hasNewLaunch ? 1 : 0;
                                        """
                                    },
                                    "order": "desc"  # new_launch products first
                                }
                            },
                            {
                                # BRD: Sort by business score - use transaction_count as primary metric
                                # NOTE: The 'score' field in data is always 0, so we use transaction_count
                                # which represents actual sales/popularity
                                # Use script sort to prefer user's city transaction_count, fallback to citi_id_0
                                "_script": {
                                    "type": "number",
                                    "script": {
                                        "lang": "painless",
                                        "source": f"""
                                            def userCity = 'citi_id_{city_id.replace("citi_id_","") if city_id else "0"}';
                                            def fallbackCity = 'citi_id_0';
                                            def userTx = -1.0;
                                            def fallbackTx = -1.0;
                                            if (params._source != null && params._source.containsKey('city_offers')) {{
                                                def offers = params._source.get('city_offers');
                                                if (offers != null) {{
                                                    for (def offer : offers) {{
                                                        def cid = offer.containsKey('cityid') ? offer.get('cityid').toString() : '';
                                                        // Use transaction_count as the business score
                                                        def tx = offer.containsKey('transaction_count') ? ((Number) offer.get('transaction_count')).doubleValue() : 0.0;
                                                        if (cid == userCity) {{ userTx = tx; }}
                                                        if (cid == fallbackCity) {{ fallbackTx = tx; }}
                                                    }}
                                                }}
                                            }}
                                            // Prefer user's city transaction_count; fallback to citi_id_0
                                            return userTx >= 0 ? userTx : (fallbackTx >= 0 ? fallbackTx : 0.0);
                                        """
                                    },
                                    "order": "desc"
                                }
                            },
                            {"_score": {"order": "desc"}},  # ES text relevance as tiebreaker
                            {"model_launch_date": {"order": "desc"}},
                            {"modelid": {"order": "asc"}}
                        ]
                        logger.info("Sorting: BRD Relevance (city_priority → new_launch_flag → transaction_count → ES_score)")
        
        # Add post_filter if user filters exist
        if post_filter_clauses:
            if len(post_filter_clauses) == 1:
                es_query["post_filter"] = post_filter_clauses[0]
            else:
                es_query["post_filter"] = {
                    "bool": {
                        "must": post_filter_clauses
                    }
                }
        
        return es_query
    
    def _build_aggregations(self, city_id: str = None, post_filter: dict = None) -> Dict:
        """Build aggregations for filters, EMI range, and unique model count
        
        Args:
            city_id: City ID for EMI aggregation
            post_filter: If provided (user attribute filters active), wraps the unique_models
                         cardinality aggregation inside a filter agg so the count reflects
                         only products matching the post_filter (fixes wrong totalrecords bug).
        """
        aggs = {}
        
        # =================== UNIQUE MODEL COUNT ===================
        # When post_filter is active (user selected attribute filters), ES cardinality
        # aggregations do NOT see post_filter by design — they run BEFORE post_filter.
        # Fix: wrap cardinality in a filter agg matching the post_filter so totalrecords
        # reflects the filtered count, not the full unfiltered count.
        if post_filter:
            aggs["unique_models"] = {
                "filter": post_filter,
                "aggs": {
                    "filtered_count": {
                        "cardinality": {
                            "field": "modelid",
                            "precision_threshold": 40000
                        }
                    }
                }
            }
        else:
            aggs["unique_models"] = {
                "cardinality": {
                    "field": "modelid",
                    "precision_threshold": 40000  # High precision for accuracy
                }
            }
        
        # Attribute aggregations
        # PERFORMANCE: Reduced from 1000 to 100 - UI rarely shows more than 50 filter values
        for option in ALL_ATTRIBUTE_OPTIONS:
            es_key = option['es_key']
            aggs[es_key] = {
                "terms": {
                    "field": es_key,
                    "size": 100,  # Reduced from 1000 for performance
                    "missing": "__missing__"
                }
            }
        
        # EMI aggregation (nested)
        if city_id:
            aggs["city_offers"] = {
                "nested": {"path": "city_offers"},
                "aggs": {
                    "filtered_offers": {
                        "filter": {
                            "bool": {
                                "should": [
                                    {"term": {"city_offers.cityid": city_id}},
                                    {"term": {"city_offers.cityid": "citi_id_0"}}
                                ],
                                "minimum_should_match": 1
                            }
                        },
                        "aggs": {
                            "min_emi": {"min": {"field": "city_offers.lowest_emi"}},
                            "max_emi": {"max": {"field": "city_offers.lowest_emi"}}
                        }
                    }
                }
            }
        
        return aggs
    
    def get_global_emi_slider_range(self, query_info: Dict, city_id: str = None, 
                                     filters: Dict = None) -> Dict:
        """
        Calculate global EMI slider min/max BEFORE applying EMI filter.
        This ensures the slider shows the full available range.
        (Reference: mall_search_api lines 4697-4711)
        """
        try:
            # Build query WITHOUT emi_range to get global min/max
            filters_for_emi = dict(filters) if filters else {}
            filters_for_emi.pop("emi", None)  # Remove EMI filter for aggregation
            
            es_query = self.build_query(
                query_info=query_info,
                city_id=city_id,
                filters=filters_for_emi,
                page=1,
                page_size=0,  # No results needed, just aggregation
                emi_range=None  # No EMI filter
            )
            
            # Replace size with 0 for aggregation only
            es_query["size"] = 0
            
            # Build city-specific EMI aggregation
            if city_id:
                es_query["aggs"] = {
                    "city_offers": {
                        "nested": {"path": "city_offers"},
                        "aggs": {
                            "filtered_offers": {
                                "filter": {
                                    "bool": {
                                        "should": [
                                            {"term": {"city_offers.cityid": city_id}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ],
                                        "minimum_should_match": 1
                                    }
                                },
                                "aggs": {
                                    "min_emi": {"min": {"field": "city_offers.lowest_emi"}},
                                    "max_emi": {"max": {"field": "city_offers.lowest_emi"}}
                                }
                            }
                        }
                    }
                }
            
            response = self.es.search(index=self.index, body=es_query, request_timeout=3)
            
            # Extract slider values
            if city_id and "aggregations" in response:
                aggs = response["aggregations"]
                if "city_offers" in aggs and "filtered_offers" in aggs["city_offers"]:
                    filtered = aggs["city_offers"]["filtered_offers"]
                    slider_min = int(filtered.get("min_emi", {}).get("value") or 0)
                    slider_max = int(filtered.get("max_emi", {}).get("value") or 0)
                    return {"min": slider_min, "max": slider_max}
            
            return {"min": 0, "max": 0}
        except Exception as e:
            logger.warning(f"EMI slider aggregation failed: {e}")
            return {"min": 0, "max": 0}
    
    def search(self, query_info: Dict, city_id: str = None, filters: Dict = None,
               page: int = 1, page_size: int = 26, emi_range: Dict = None,
               sort_by: str = None, from_offset: int = None, start_time: float = None) -> Dict:
        """Execute search and return results with fallback for zero results
        
        Args:
            sort_by: Sorting option - 'relevance' (default), 'low_to_high', 'high_to_low',
                     'latest_launch', 'best_selling', 'most_viewed', 'discounts', 'newest'.
                     City priority: user's city_id products appear first, then city_id=0.
                     Fallback: if sort returns 0 results, fallback to relevance.
            from_offset: Direct offset for pagination (if provided, overrides page calculation)
            start_time: Request start time for timeout tracking (skip fallbacks if running low)
        """
        try:
            # Helper function to check if we should skip expensive operations
            def should_skip_fallback():
                """Returns True if we're running low on time and should skip fallbacks"""
                if start_time is None:
                    return False
                elapsed = time.time() - start_time
                remaining = MAX_REQUEST_TIME_SECONDS - elapsed
                if remaining < 1.5:  # Less than 1.5s remaining, skip fallbacks
                    logger.warning(f"Time budget low ({remaining:.2f}s remaining), skipping fallback")
                    return True
                return False
            # =================== BLOCKED RESULTS CHECK ===================
            # If block_results flag is set (e.g., for Apple non-phone products),
            # return empty results immediately without searching
            if query_info.get("block_results"):
                logger.info(f"Results blocked for query: '{query_info.get('original', '')}' - returning empty")
                return {
                    "success": True,
                    "hits": [],
                    "total": 0,
                    "aggregations": {},
                    "blocked": True  # Flag to indicate intentionally blocked
                }
            
            # =================== HAND-CRAFTED FIX: CHETAK/PULSAR → TWO-WHEELER ===================
            # Force two-wheeler category for Chetak (scooter) and Pulsar (bike) queries
            original_lower = query_info.get("original", "").lower()
            
            # Chetak and typos → force scooter (two-wheeler)
            chetak_keywords = ["chetak", "chetk", "chetek", "chatak", "cheetek", "chetaak"]
            for kw in chetak_keywords:
                if kw in original_lower:
                    query_info["category"] = "two wheeler"
                    query_info["processed"] = original_lower.replace(kw, "scooter")
                    query_info["scooter_only_filter"] = True
                    logger.info(f"Hand-crafted fix: '{kw}' → two wheeler scooter")
                    break
            
            # Pulsar and typos → force bike (two-wheeler)
            pulsar_keywords = ["pulsar", "pulser", "plsar", "plser", "pulsr", "pulsaar", "pulsur", "pulzar"]
            for kw in pulsar_keywords:
                if kw in original_lower:
                    query_info["category"] = "two wheeler"
                    query_info["processed"] = original_lower.replace(kw, "motorcycle bike")
                    query_info["bike_only_filter"] = True
                    logger.info(f"Hand-crafted fix: '{kw}' → two wheeler bike")
                    break
            
            # BRD Req3: For brand-only queries on page 1, over-fetch from ES to ensure
            # all categories are represented for multi-category reranking.
            # We'll trim back to the requested page_size after reranking.
            original_page_size = page_size
            _is_brand_only_prefetch = (
                page == 1 and
                (query_info.get("is_brand_only_phone_query", False) or
                 (query_info.get("brand") and not query_info.get("category") and not query_info.get("is_apple_query")))
            )
            
            # =================== MODEL-SPECIFIC QUERY OVER-FETCH ===================
            # For model-specific queries (e.g., "Vivo V70", "moto g96"), over-fetch to ensure
            # we get exact matches even if they're only available in citi_id_0 (not user's city)
            # City priority sort may push citi_id_0 products beyond page_size; over-fetching compensates.
            _processed = query_info.get("processed", "")
            _has_model_number = bool(re.search(r'\d', re.sub(r'\b5g\b|\b4g\b|\b3g\b', '', _processed)))
            _is_model_query_prefetch = (
                page == 1 and 
                _has_model_number and 
                not query_info.get("is_brand_only_phone_query", False)
            )
            
            if _is_brand_only_prefetch:
                page_size = max(page_size, 50)  # Fetch up to 50 to cover all brand categories
                logger.info(f"BRD Req3: Over-fetching {page_size} hits for brand-only multi-category (original: {original_page_size})")
            elif _is_model_query_prefetch:
                page_size = max(page_size, 50)  # Fetch up to 50 to ensure exact model matches (reduced from 100 for performance)
                logger.info(f"Model query: Over-fetching {page_size} hits to find exact match for '{_processed}' (original: {original_page_size})")
            
            # =================== MULTI-WORD QUERY OVER-FETCH ===================
            # For multi-word queries without numbers (e.g., "mac mini", "redmi pad", "oneplus pad"),
            # over-fetch to ensure exact phrase matches appear even if they have lower scores initially.
            # This is needed because partial matches may rank higher without the phrase boost working.
            _query_words = _processed.split() if _processed else []
            _is_multi_word_query = (
                page == 1 and
                len(_query_words) >= 2 and
                not _is_brand_only_prefetch and
                not _is_model_query_prefetch
            )
            if _is_multi_word_query:
                page_size = max(page_size, 50)  # Fetch up to 50 for reranking
                logger.info(f"Multi-word query: Over-fetching {page_size} hits for exact phrase reranking (original: {original_page_size})")
            
            # =================== FLAG-BASED QUERY OVER-FETCH ===================
            # For flag-based queries (best_selling, one_emi_off, zero_dp, new_launch), over-fetch
            # to ensure products with flags only in citi_id_0 aren't pushed beyond page_size
            # by city priority sort (which ranks citi_id_0 lower than user's city)
            _is_flag_query = (
                page == 1 and
                (query_info.get("detect_best_selling", False) or
                 query_info.get("detect_one_emi_off", False) or
                 query_info.get("detect_zero_dp", False) or
                 query_info.get("detect_new_launch", False))
            )
            if _is_flag_query and not _is_brand_only_prefetch and not _is_model_query_prefetch:
                page_size = max(page_size, 50)  # Fetch up to 50 for flag-based queries (reduced from 100 for performance)
                logger.info(f"Flag-based query: Over-fetching {page_size} hits to include citi_id_0 products (original: {original_page_size})")
            
            # PERFORMANCE: Keep aggregations on all pages (filters required by frontend)
            # Only skip expensive script-based sorts for pagination (page > 1)
            _skip_aggs = False  # Always include aggregations - frontend needs filters on all pages
            _skip_expensive_sorts = (page > 1)  # Skip script-based sorts for pagination
            
            es_query = self.build_query(query_info, city_id, filters, page, page_size, emi_range, sort_by, from_offset, 
                                        skip_aggregations=_skip_aggs, skip_expensive_sorts=_skip_expensive_sorts)
            
            # Add timeout to ES query for production safety (3 seconds max)
            es_query["timeout"] = "3s"
            
            # PERFORMANCE: Only log query size, not full query (JSON dump is expensive)
            logger.debug(f"ES Query size: {len(str(es_query))} chars")
            
            response = self.es.search(
                index=self.index, 
                body=es_query, 
                request_timeout=4,
                request_cache=True  # Cache repeated queries for better performance
            )
            total = response["hits"]["total"]["value"]
            
            # =================== SORT FALLBACK TO RELEVANCE ===================
            # If a non-relevance sort returns 0 results, fallback to relevance sort
            # This ensures users always see results if products exist, regardless of sort option
            sort_by_lower = (sort_by or "").lower().strip()
            non_relevance_sorts = ("low_to_high", "lowtohigh", "price_low_high", "emi_low_high", "asc",
                                   "high_to_low", "hightolow", "price_high_low", "emi_high_low", "desc",
                                   "latest_launch", "latestlaunch", "newest", "new_arrivals",
                                   "best_selling", "bestselling", "best_seller", "top_seller", "topseller",
                                   "most_viewed", "mostviewed", "popular", "trending",
                                   "discounts", "discount", "offers", "deal", "deals")
            
            if total == 0 and sort_by_lower in non_relevance_sorts:
                logger.info(f"Zero results with sort_by='{sort_by}', falling back to relevance sort...")
                fallback_es_query = self.build_query(query_info, city_id, filters, page, page_size, emi_range, "relevance", from_offset)
                fallback_response = self.es.search(index=self.index, body=fallback_es_query, request_timeout=2)
                fallback_total = fallback_response["hits"]["total"]["value"]
                
                if fallback_total > 0:
                    logger.info(f"Sort fallback to relevance returned {fallback_total} results")
                    response = fallback_response
                    total = fallback_total
                else:
                    logger.info(f"Sort fallback to relevance also returned 0 results")
            
            # =================== CITY FILTER FALLBACK (GLOBAL CATALOG SEARCH) ===================
            # If zero results with city filter, search global catalog (no city restriction)
            # This ensures products not available in user's city still appear in search results
            # Products from global search are marked for potential "not available in your city" display
            if total == 0 and city_id and city_id not in ['citi_id_0', '0', 'citi_id_']:
                logger.info(f"Zero results with city_id='{city_id}', falling back to global catalog search (no city filter)...")
                # Search without city filter - pass None to skip city_offers nested filter
                global_fallback_es_query = self.build_query(
                    query_info, 
                    city_id=None,  # No city filter
                    filters=filters, 
                    page=page, 
                    page_size=page_size, 
                    emi_range=None,  # Also skip EMI filter (city-dependent)
                    sort_by=sort_by, 
                    from_offset=from_offset
                )
                global_fallback_response = self.es.search(index=self.index, body=global_fallback_es_query, request_timeout=2)
                global_fallback_total = global_fallback_response["hits"]["total"]["value"]
                
                if global_fallback_total > 0:
                    logger.info(f"Global catalog fallback returned {global_fallback_total} results (products may not be available in city {city_id})")
                    response = global_fallback_response
                    total = global_fallback_total
                    # Mark for frontend that these are global results (not city-specific)
                    query_info["_global_catalog_fallback"] = True
                    query_info["_original_city_id"] = city_id
                else:
                    logger.info(f"Global catalog fallback also returned 0 results")
            
            # =================== BRD KEYWORD FILTER FALLBACKS ===================
            # If new_launch_flag filter returns 0 results (no flag data yet in feed),
            # fall back to "latest_launch" sort (model_launch_date desc) — semantically equivalent
            if total == 0 and query_info.get("detect_new_launch"):
                logger.info("Zero results with new_launch_flag filter — falling back to latest_launch sort (model_launch_date)...")
                nl_fallback_info = query_info.copy()
                nl_fallback_info["detect_new_launch"] = False  # Remove the strict flag filter
                nl_fallback_query = self.build_query(nl_fallback_info, city_id, filters, page, page_size, emi_range, "latest_launch", from_offset)
                nl_fallback_resp = self.es.search(index=self.index, body=nl_fallback_query)
                nl_fallback_total = nl_fallback_resp["hits"]["total"]["value"]
                if nl_fallback_total > 0:
                    logger.info(f"new_launch fallback to latest_launch sort returned {nl_fallback_total} results")
                    response = nl_fallback_resp
                    total = nl_fallback_total

            # =================== PROGRESSIVE FILTER RELAXATION ENGINE ===================
            # When multi-attribute NL queries return 0 results (e.g. "5 star double door red fridge"),
            # progressively drop the weakest filter and retry until results appear.
            #
            # Relaxation priority (weakest dropped first → strongest kept longest):
            #   1. color         — most subjective, user often flexible
            #   2. wm_type       — function type preference (front/top load)
            #   3. door_type     — physical form factor
            #   4. star_rating   — strongest purchase intent signal, dropped last
            #
            # Each level strips the filter AND its text token from the query so ES text
            # matching doesn't inadvertently exclude valid products.
            #
            # Metadata: query_info["_relaxed_filters"] tracks which filters were relaxed
            # so the response can tell the frontend (e.g. "Showing 5★ double-door fridges, all colors").

            _nl_attrs = query_info.get("attributes") or {}

            # Ordered list of droppable NL filters: (filter_key, text_key, label)
            # filter_key  = the "_*_post_filter" key in attributes
            # text_key    = the plain-text key in attributes (word to strip from query)
            # label       = human-readable name for relaxed_filters metadata
            _RELAXATION_ORDER = [
                ("_color_post_filter",    "color",    "color"),
                ("_wm_type_post_filter",  "wm_type",  "washing machine type"),
                ("_door_type_post_filter", "door_type", "door type"),
                ("_star_post_filter",     "star_rating", "star rating"),
            ]

            # Collect which NL filters are actually active
            _active_nl_filters = [
                (fk, tk, lbl) for fk, tk, lbl in _RELAXATION_ORDER
                if _nl_attrs.get(fk)
            ]

            if total == 0 and len(_active_nl_filters) >= 2:
                logger.info(
                    f"Progressive filter relaxation: {len(_active_nl_filters)} NL filters active, "
                    f"0 results. Filters: {[lbl for _, _, lbl in _active_nl_filters]}"
                )
                _relaxed = []  # track which filters we dropped

                for _fk, _tk, _lbl in _active_nl_filters:
                    if total > 0:
                        break  # found results — stop relaxing

                    _text_word = _nl_attrs.get(_tk, "")
                    logger.info(f"  Relaxation: dropping '{_lbl}' (word='{_text_word}')...")

                    # Remove filter + text key from attributes
                    _fb_attrs = dict(query_info.get("attributes") or {})
                    _fb_attrs.pop(_fk, None)
                    _fb_attrs.pop(_tk, None)

                    _fb_query_info = dict(query_info)
                    _fb_query_info["attributes"] = _fb_attrs

                    # Strip the text word from processed/original so ES text match
                    # doesn't exclude products that lack that word
                    if _text_word:
                        _strip_re = re.compile(rf'\b{re.escape(str(_text_word))}\b', re.IGNORECASE)
                        for _tf in ("processed", "original"):
                            if _fb_query_info.get(_tf):
                                _fb_query_info[_tf] = _strip_re.sub("", _fb_query_info[_tf]).strip()

                        # Guard against false brand detection caused by stripped word
                        # e.g., "red" in "red fridge" → brand falsely detected as "redmi"
                        _detected_brand = (_fb_query_info.get("brand") or "").lower()
                        _stripped_lower = str(_text_word).lower()
                        if _detected_brand and (
                            _detected_brand.startswith(_stripped_lower) or
                            _stripped_lower.startswith(_detected_brand)
                        ):
                            logger.info(
                                f"  Relaxation: clearing false brand '{_detected_brand}' "
                                f"(triggered by stripped word '{_text_word}')"
                            )
                            _fb_query_info["brand"] = None

                    try:
                        _fb_es_query = self.build_query(
                            _fb_query_info, city_id, filters, page, page_size,
                            emi_range, sort_by, from_offset
                        )
                        _fb_resp = self.es.search(index=self.index, body=_fb_es_query)
                        _fb_total = _fb_resp["hits"]["total"]["value"]

                        _relaxed.append({"filter": _lbl, "value": str(_text_word)})

                        if _fb_total > 0:
                            logger.info(
                                f"  Relaxation success: dropping '{_lbl}' → {_fb_total} results. "
                                f"Kept: {[l for f, t, l in _active_nl_filters if f in _fb_attrs]}"
                            )
                            response = _fb_resp
                            total = _fb_total
                            query_info = _fb_query_info  # propagate to downstream guards
                        else:
                            logger.info(f"  Relaxation: still 0 after dropping '{_lbl}', continuing...")
                            # Propagate the dropped filter for next iteration
                            query_info = _fb_query_info
                    except Exception as _relax_err:
                        logger.warning(f"  Relaxation retry failed for '{_lbl}': {_relax_err}")
                        _relaxed.append({"filter": _lbl, "value": str(_text_word), "error": True})
                        # Still propagate the drop so next iteration doesn't include it
                        query_info = _fb_query_info

                # Store relaxation metadata for the response
                if _relaxed:
                    query_info["_relaxed_filters"] = _relaxed
                    _kept = [
                        lbl for fk, tk, lbl in _active_nl_filters
                        if (query_info.get("attributes") or {}).get(fk)
                    ]
                    if total > 0:
                        logger.info(
                            f"Progressive relaxation resolved: {total} results. "
                            f"Dropped: {[r['filter'] for r in _relaxed]}, Kept: {_kept}"
                        )
                    else:
                        logger.info(
                            f"Progressive relaxation exhausted all {len(_relaxed)} filters, "
                            f"still 0 results. Fallbacks 5 & 6 will be attempted."
                        )

            # If zero_dp_flag filter returns 0 results for this city/query,
            # broaden to show zero_dp products across all cities (citi_id_0)
            if total == 0 and query_info.get("detect_zero_dp"):
                logger.info("Zero results with zero_dp_flag filter — trying without city restriction...")
                zdp_fallback_info = query_info.copy()
                zdp_fallback_info["detect_zero_dp"] = False  # Remove flag filter; keep as sort hint
                zdp_fallback_query = self.build_query(zdp_fallback_info, city_id, filters, page, page_size, emi_range, "relevance", from_offset)
                zdp_fallback_resp = self.es.search(index=self.index, body=zdp_fallback_query)
                zdp_fallback_total = zdp_fallback_resp["hits"]["total"]["value"]
                if zdp_fallback_total > 0:
                    logger.info(f"zero_dp fallback returned {zdp_fallback_total} results")
                    response = zdp_fallback_resp
                    total = zdp_fallback_total

            # =================== EXACT MODEL MATCH RERANKING ===================
            # Re-rank results to prioritize exact model matches over partial matches
            # E.g., "oppo k13" query should show "K13" before "K13x"
            # This compensates for ES edge ngram tokenizer matching both equally
            
            # =================== SERIES KEYWORD DETECTION (Shared by reranking + BRD Req1) ===================
            # Detect series keywords EARLY so they can be used both for reranking boost
            # and for filtering new_launch products in BRD Req1
            _GLOBAL_SERIES_KEYWORDS = {
                # Samsung phone series
                "fold": "z fold", "flip": "z flip",
                # Samsung laptop series
                "galaxy book": "galaxy book",
                # HP laptop series
                "victus": "victus", "omen": "omen", "envy": "envy", "spectre": "spectre",
                "elitebook": "elitebook", "probook": "probook", "pavilion": "pavilion",
                # Lenovo laptop series  
                "thinkpad": "thinkpad", "ideapad": "ideapad", "legion": "legion", "yoga": "yoga",
                "loq": "loq",
                # Asus laptop series
                "tuf": "tuf", "rog": "rog", "vivobook": "vivobook", "zenbook": "zenbook",
                # Dell laptop series
                "inspiron": "inspiron", "latitude": "latitude", "xps": "xps", "alienware": "alienware",
                # Acer laptop series
                "nitro": "nitro", "predator": "predator", "aspire": "aspire", "swift": "swift",
            }
            _series_boost_term = None  # Will be set if query contains a series keyword
            processed_for_series = query_info.get("processed", "").lower()
            for kw, series in _GLOBAL_SERIES_KEYWORDS.items():
                if kw in processed_for_series:
                    _series_boost_term = series
                    break
            if _series_boost_term:
                logger.info(f"Series keyword detected: '{_series_boost_term}' from query '{processed_for_series}'")
            
            if total > 0 and query_info.get("processed"):
                hits = response["hits"]["hits"]
                processed_query = query_info.get("processed", "").lower()
                query_words = processed_query.split()
                
                # Find alphanumeric model numbers in query (e.g., "k13", "reno14", "a55", "ps5")
                model_patterns = []
                for word in query_words:
                    if len(word) >= 2 and word.isalnum() and re.search(r'\d', word):
                        model_patterns.append(word.lower())
                
                # FIX: Add concatenated patterns for "brand number" → "brandnumber"
                # E.g., "reno 15" → also check for "reno15", "nord 5" → "nord5", "v70" etc.
                # This handles products named like "Reno15 Pro" when user searches "Reno 15 Pro"
                for i, word in enumerate(query_words[:-1]):
                    next_word = query_words[i + 1]
                    # If current word is letters-only and next word starts with digit
                    if word.isalpha() and next_word[0].isdigit():
                        concat = (word + next_word).lower()
                        if concat not in model_patterns:
                            model_patterns.append(concat)
                
                # =================== HIGH PRECISION EXACT MATCH RERANKING ===================
                # ALWAYS apply exact match reranking for better precision
                # This ensures products with ALL query terms in name appear first
                
                def calc_precision_score(hit):
                    """Calculate precision score - prioritize exact query matches in product name"""
                    source = hit.get("_source", {})
                    product_name = source.get("product_name", "").lower()
                    search_field = source.get("search_field", "").lower()
                    sku_name = source.get("sku_name", "").lower()
                    combined_text = f"{product_name} {search_field} {sku_name}"
                    
                    score = 0
                    stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'for', 'and', 'or', 'with', 'gb', 'ram', 'storage'}
                    significant_words = [w for w in query_words if len(w) >= 2 and w not in stop_words]
                    
                    # ===== MODEL SERIES MATCH (highest priority) =====
                    if _series_boost_term and _series_boost_term in product_name:
                        score += 10000
                    
                    # ===== EXACT MODEL NUMBER MATCH =====
                    # For model numbers like "ps5", "ce4", "v60", boost exact match over partial
                    for pattern in model_patterns:
                        # Exact match in product name (not substring)
                        pattern_regex = rf'\b{re.escape(pattern)}\b'
                        if re.search(pattern_regex, product_name):
                            score += 8000  # Very high for exact model in product name
                        elif re.search(pattern_regex, combined_text):
                            score += 4000  # Good for exact model in other fields
                    
                    # ===== ALL QUERY WORDS IN PRODUCT NAME =====
                    words_in_name = 0
                    words_in_combined = 0
                    for word in significant_words:
                        pattern = rf'\b{re.escape(word)}\b'
                        if re.search(pattern, product_name):
                            words_in_name += 1
                        elif re.search(pattern, combined_text):
                            words_in_combined += 1
                    
                    # Huge bonus if ALL significant words are in product name
                    if significant_words and words_in_name == len(significant_words):
                        score += 6000
                    elif significant_words and words_in_name > 0:
                        # Partial match in name
                        score += int(3000 * (words_in_name / len(significant_words)))
                    
                    # Smaller bonus for words in other fields
                    if words_in_combined > 0 and significant_words:
                        score += int(1000 * (words_in_combined / len(significant_words)))
                    
                    # ===== SPECIAL KEYWORDS BOOST =====
                    # "gaming" in query should boost products with "gaming" in name
                    if "gaming" in query_words and "gaming" in product_name:
                        score += 8000  # High boost for gaming keyword match
                    
                    # Penalize MacBook when user searches "gaming" (MacBook Neo is NOT gaming laptop)
                    if "gaming" in query_words and "macbook" in product_name:
                        score -= 10000  # Strong penalty
                    
                    # PS5/PS4 specific boost - prioritize exact version match
                    if "ps5" in processed_query or "ps 5" in processed_query:
                        if "ps5" in product_name.replace(" ", ""):
                            score += 10000
                        elif "ps4" in product_name.replace(" ", ""):
                            score -= 5000  # Penalize PS4 when searching for PS5
                    if "ps4" in processed_query or "ps 4" in processed_query:
                        if "ps4" in product_name.replace(" ", ""):
                            score += 10000
                    
                    # V60 vs X60 disambiguation - exact model number priority
                    # Also handle "V60 Pro" query when V60 Pro doesn't exist - should show V60 not V40 Pro
                    if "v60" in query_words:
                        if "v60" in product_name:
                            score += 10000  # Strong boost for exact V60 match
                        elif re.search(r'\bv\d{2}\b', product_name):
                            # Penalize OTHER V-series models (V20, V30, V40) when searching for V60
                            score -= 6000
                        elif "x60" in product_name:
                            score -= 5000  # Penalize X60 when searching for V60
                    
                    # Generic V-series model disambiguation (V20, V30, V40, V50, V70, etc.)
                    v_model_match = re.search(r'\bv(\d{2})\b', processed_query)
                    if v_model_match:
                        model_num = v_model_match.group(1)  # e.g., "60" from "v60"
                        target_model = f"v{model_num}"
                        if target_model in product_name:
                            score += 8000  # Boost exact model
                        elif re.search(r'\bv\d{2}\b', product_name):
                            # Penalize other V-series when specific model requested
                            score -= 5000
                    
                    # Nord series boost
                    if "nord" in query_words and "nord" in product_name:
                        score += 5000
                    
                    # CE4 specific - boost CE4 over other CE models
                    if "ce4" in processed_query.replace(" ", "") or "ce 4" in processed_query:
                        if "ce 4" in product_name or "ce4" in product_name.replace(" ", ""):
                            score += 8000
                    
                    # Neo series boost (for iQOO Neo)
                    if "neo" in query_words and "neo" in product_name:
                        score += 5000
                    
                    # Reno series boost (for OPPO Reno)
                    if "reno" in query_words and "reno" in product_name:
                        score += 5000
                    
                    # ===== LAPTOP SERIES DISAMBIGUATION =====
                    # ThinkPad vs IdeaPad - boost exact series match, penalize wrong series
                    if "thinkpad" in query_words or "think pad" in processed_query:
                        if "thinkpad" in product_name.replace(" ", "").lower():
                            score += 10000  # Strong boost for ThinkPad match
                        elif "ideapad" in product_name.replace(" ", "").lower():
                            score -= 8000  # Penalize IdeaPad when searching for ThinkPad
                    if "ideapad" in query_words or "idea pad" in processed_query:
                        if "ideapad" in product_name.replace(" ", "").lower():
                            score += 10000  # Strong boost for IdeaPad match
                        elif "thinkpad" in product_name.replace(" ", "").lower():
                            score -= 8000  # Penalize ThinkPad when searching for IdeaPad
                    
                    # EliteBook vs ProBook disambiguation
                    if "elitebook" in query_words or "elite book" in processed_query:
                        if "elitebook" in product_name.replace(" ", "").lower():
                            score += 10000
                        elif "probook" in product_name.replace(" ", "").lower():
                            score -= 8000
                    if "probook" in query_words or "pro book" in processed_query:
                        if "probook" in product_name.replace(" ", "").lower():
                            score += 10000
                        elif "elitebook" in product_name.replace(" ", "").lower():
                            score -= 8000
                    
                    return score
                
                # Apply precision reranking
                indexed_hits = [(i, hit, calc_precision_score(hit)) for i, hit in enumerate(hits)]
                sorted_hits = sorted(indexed_hits, key=lambda x: (-x[2], x[0]))
                response["hits"]["hits"] = [hit for _, hit, _ in sorted_hits]
                
                # Log top scores for debugging
                top_scores = [(h.get("_source", {}).get("product_name", "")[:40], s) for _, h, s in sorted_hits[:3]]
                logger.info(f"Precision reranked {len(hits)} hits for '{processed_query}'. Top: {top_scores}")
            
            # =================== BRD: FIRST 2 NEW LAUNCH RERANKING ===================
            # Per BRD Requirement 1: First 2 SKUs should have new_launch_flag=true
            # Priority: user's city_id new_launch first, then citi_id_0 (global) new_launch
            # Only apply for relevance sort (default) and page 1
            # SKIP for model-specific queries (e.g., "moto g96 5g", "vivo t4x") where
            # user wants THAT exact product, not a random new_launch product.
            # SKIP for flag-based queries (zero_dp, new_launch, best_selling, one_emi_off)
            # where user has explicit intent and we're already filtering by that flag.
            is_relevance_sort = sort_by_lower in ("", "relevance", None) or not sort_by_lower
            _processed = query_info.get("processed", "")
            _brand = query_info.get("brand", "") or ""
            _has_model_number = bool(re.search(r'\d', re.sub(r'\b5g\b|\b4g\b|\b3g\b', '', _processed)))
            # Also check for model patterns without digits (z fold, mac mini, etc.)
            # Include "gaming" so "gaming laptops" doesn't get random new_launch products
            _MODEL_PATTERNS_NO_DIGITS = ['z fold', 'z flip', 'mac mini', 'mac studio', 'ideacentre', 'ideapad', 'omen', 'legion', 'omnidesk', 'surface', 'gaming', 'loq', 'victus', 'tuf', 'rog', 'predator', 'nitro', 'vivobook', 'zenbook', 'thinkpad', 'elitebook', 'probook', 'galaxy book']
            _has_model_pattern = any(p in _processed.lower() for p in _MODEL_PATTERNS_NO_DIGITS)
            _is_model_query = (_has_model_number or _has_model_pattern) and not query_info.get("is_brand_only_phone_query", False)
            
            # Check all flag-based queries - skip new_launch reranking for any of them
            _is_best_selling = query_info.get("detect_best_selling", False)
            _is_zero_dp = query_info.get("detect_zero_dp", False)
            _is_new_launch_filter = query_info.get("detect_new_launch", False)
            _is_one_emi_off = query_info.get("detect_one_emi_off", False)
            _has_flag_filter = _is_best_selling or _is_zero_dp or _is_new_launch_filter or _is_one_emi_off
            
            if _is_model_query:
                logger.info(f"BRD Req1: Skipping new_launch reranking for model-specific query: '{_processed}'")
            if _has_flag_filter:
                logger.info(f"BRD Req1: Skipping new_launch reranking for flag-based query (user has explicit intent)")

            def _normalize_city(c):
                """Normalize city_id to citi_id_X format for comparison"""
                if not c:
                    return "citi_id_0"
                c = str(c)
                return c if c.startswith("citi_id_") else f"citi_id_{c}"

            if total > 0 and is_relevance_sort and page == 1 and not _is_model_query and not _has_flag_filter:
                hits = response["hits"]["hits"]
                
                def get_new_launch_flag(hit, target_city_id):
                    """Check if hit has new_launch_flag=true for user's city first, then city_id=0
                    
                    BRD: new_launch_flag must be checked at the user's city level first.
                    Only if user's city has no offer, check citi_id_0.
                    """
                    city_offers = hit.get("_source", {}).get("city_offers", [])
                    norm_city = _normalize_city(target_city_id)
                    # First: check user's specific city
                    for offer in city_offers:
                        cid = _normalize_city(offer.get("cityid"))
                        if cid == norm_city:
                            return offer.get("new_launch_flag") in (True, 1, "1", "true")
                    # Fallback: check citi_id_0
                    for offer in city_offers:
                        cid = _normalize_city(offer.get("cityid"))
                        if cid == "citi_id_0":
                            return offer.get("new_launch_flag") in (True, 1, "1", "true")
                    return False
                
                def get_business_score(hit, target_city_id):
                    """Get highest business score for user's city or fallback to city_id=0"""
                    city_offers = hit.get("_source", {}).get("city_offers", [])
                    norm_city = _normalize_city(target_city_id)
                    # First try user's city
                    for offer in city_offers:
                        if _normalize_city(offer.get("cityid")) == norm_city:
                            return offer.get("score", 0) or 0
                    # Fallback to citi_id_0
                    for offer in city_offers:
                        if _normalize_city(offer.get("cityid")) == "citi_id_0":
                            return offer.get("score", 0) or 0
                    return 0
                
                # Separate new_launch and non-new_launch from current page results
                new_launch_hits = []
                other_hits = []
                target_city = city_id if city_id else "0"
                
                for hit in hits:
                    if get_new_launch_flag(hit, target_city):
                        new_launch_hits.append(hit)
                    else:
                        other_hits.append(hit)
                
                # =================== FETCH NEW LAUNCH PRODUCTS IF NOT IN CURRENT PAGE ===================
                # Problem: ES page 1 may be all user-city products (high city priority sort),
                # and new_launch products (at citi_id_0) can end up on page 2+.
                # Fix: If we don't have 2 new_launch hits, run a SEPARATE ES query to fetch them.
                #
                # Strategy:
                #   Step A: Check user's city for new_launch products
                #   Step B: If not enough, check citi_id_0 for new_launch products
                #   Step C: Inject fetched new_launch hits into positions 1-2
                
                needed_nl = 2 - len(new_launch_hits)
                if needed_nl > 0:
                    logger.info(f"BRD: Need {needed_nl} more new_launch products (found {len(new_launch_hits)} in page). Fetching from ES...")
                    
                    # Collect model_ids already in results to avoid duplicates
                    existing_model_ids = set()
                    for h in hits:
                        mid = h.get("_source", {}).get("modelid") or h.get("fields", {}).get("modelid")
                        if mid:
                            existing_model_ids.add(str(mid))
                    for h in new_launch_hits:
                        mid = h.get("_source", {}).get("modelid")
                        if mid:
                            existing_model_ids.add(str(mid))
                    
                    # Build base query matching the same search criteria (category, brand, etc.)
                    nl_base_must = []
                    if query_info.get("processed"):
                        nl_base_must.append({"multi_match": {
                            "query": query_info["processed"],
                            "fields": ["search_field^3", "product_name^2", "manufacturer_desc"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }})
                    nl_base_filter = []
                    if query_info.get("category"):
                        cat_canonical = CATEGORY_CANONICAL.get(query_info["category"].lower(), query_info["category"])
                        # Use strict asset_category_name filter for furniture subcategories
                        STRICT_NL_ASSET_CAT = {"mattress": "Mattress", "mattresses": "Mattress"}
                        if cat_canonical.lower() in STRICT_NL_ASSET_CAT:
                            nl_base_filter.append({"term": {"asset_category_name": STRICT_NL_ASSET_CAT[cat_canonical.lower()]}})
                        else:
                            nl_base_filter.append({"term": {"actual_category": cat_canonical}})
                    if query_info.get("brand"):
                        nl_base_filter.append({"wildcard": {"manufacturer_desc": {"value": f"*{query_info['brand']}*", "case_insensitive": True}}})
                    
                    # FIX: Add price filter to new_launch fetch to respect user's price constraints
                    # This ensures "laptop under 40000" doesn't show MacBook (69900) as new_launch
                    if filters and 'price_range' in filters:
                        price_range = filters.get('price_range', {})
                        if price_range:
                            # Build nested price filter for new_launch fetch
                            nl_price_filter = {
                                "nested": {
                                    "path": "city_offers",
                                    "query": {
                                        "bool": {
                                            "must": [{"range": {"city_offers.offer_price": {}}}]
                                        }
                                    }
                                }
                            }
                            nl_price_range = nl_price_filter["nested"]["query"]["bool"]["must"][0]["range"]["city_offers.offer_price"]
                            if "gte" in price_range:
                                nl_price_range["gte"] = price_range["gte"]
                            if "lte" in price_range:
                                nl_price_range["lte"] = price_range["lte"]
                            nl_base_filter.append(nl_price_filter)
                            logger.info(f"BRD: Added price filter to new_launch fetch: {price_range}")
                    
                    # FIX: If query has a series keyword (e.g., "fold" → "z fold"), require new_launch
                    # products to MATCH the series. This prevents Samsung S26 (new_launch) from
                    # appearing above Samsung Z Fold when user searches "samsung fold".
                    if _series_boost_term:
                        nl_base_must.append({"match_phrase": {"sku_name": {"query": _series_boost_term}}})
                        logger.info(f"BRD Req1: Filtering new_launch to match series: '{_series_boost_term}'")
                    
                    fetched_nl_hits = []
                    
                    # Step A: Try user's city first
                    norm_target = _normalize_city(target_city)
                    if norm_target != "citi_id_0":
                        nl_city_query = {
                            "size": needed_nl * 3,  # fetch extra in case of model_id overlap
                            "query": {"bool": {
                                "must": nl_base_must if nl_base_must else [{"match_all": {}}],
                                "filter": nl_base_filter + [{
                                    "nested": {
                                        "path": "city_offers",
                                        "query": {"bool": {"must": [
                                            {"term": {"city_offers.new_launch_flag": 1}},
                                            {"term": {"city_offers.cityid": norm_target}}
                                        ]}},
                                        "inner_hits": {"size": 5}
                                    }
                                }]
                            }},
                            "sort": [{"city_offers.score": {
                                "order": "desc", "mode": "max",
                                "nested": {"path": "city_offers",
                                           "filter": {"term": {"city_offers.cityid": norm_target}}}
                            }}],
                            "collapse": {"field": "modelid", "inner_hits": {"name": "sku_variants", "size": 10}},
                            "_source": True
                        }
                        try:
                            nl_city_resp = self.es.search(index=self.index, body=nl_city_query)
                            for nlh in nl_city_resp["hits"]["hits"]:
                                mid = str(nlh.get("_source", {}).get("modelid", ""))
                                if mid not in existing_model_ids:
                                    fetched_nl_hits.append(nlh)
                                    existing_model_ids.add(mid)
                                    if len(fetched_nl_hits) >= needed_nl:
                                        break
                            logger.info(f"BRD: Fetched {len(fetched_nl_hits)} new_launch from user city {norm_target}")
                        except Exception as e:
                            logger.warning(f"BRD: Failed to fetch new_launch from user city: {e}")
                    
                    # Step B: If still need more, try citi_id_0 (global)
                    still_needed = needed_nl - len(fetched_nl_hits)
                    if still_needed > 0:
                        nl_global_query = {
                            "size": still_needed * 3,
                            "query": {"bool": {
                                "must": nl_base_must if nl_base_must else [{"match_all": {}}],
                                "filter": nl_base_filter + [{
                                    "nested": {
                                        "path": "city_offers",
                                        "query": {"bool": {"must": [
                                            {"term": {"city_offers.new_launch_flag": 1}},
                                            {"term": {"city_offers.cityid": "citi_id_0"}}
                                        ]}},
                                        "inner_hits": {"size": 5}
                                    }
                                }]
                            }},
                            "sort": [{"city_offers.score": {
                                "order": "desc", "mode": "max",
                                "nested": {"path": "city_offers",
                                           "filter": {"term": {"city_offers.cityid": "citi_id_0"}}}
                            }}],
                            "collapse": {"field": "modelid", "inner_hits": {"name": "sku_variants", "size": 10}},
                            "_source": True
                        }
                        try:
                            nl_global_resp = self.es.search(index=self.index, body=nl_global_query)
                            for nlh in nl_global_resp["hits"]["hits"]:
                                mid = str(nlh.get("_source", {}).get("modelid", ""))
                                if mid not in existing_model_ids:
                                    fetched_nl_hits.append(nlh)
                                    existing_model_ids.add(mid)
                                    if len(fetched_nl_hits) >= needed_nl:
                                        break
                            logger.info(f"BRD: Fetched {len(fetched_nl_hits)} total new_launch (added global citi_id_0)")
                        except Exception as e:
                            logger.warning(f"BRD: Failed to fetch new_launch from citi_id_0: {e}")
                    
                    # Merge fetched new_launch into the new_launch_hits pool
                    new_launch_hits.extend(fetched_nl_hits)
                
                # Sort new_launch by business score (desc), sort others by business score (desc)
                new_launch_hits.sort(key=lambda h: get_business_score(h, target_city), reverse=True)
                other_hits.sort(key=lambda h: get_business_score(h, target_city), reverse=True)
                
                # BRD: First 2 should be new_launch, then rest by score
                if len(new_launch_hits) >= 2:
                    reranked_hits = new_launch_hits[:2]
                    remaining_all = new_launch_hits[2:] + other_hits
                    remaining_all.sort(key=lambda h: get_business_score(h, target_city), reverse=True)
                    reranked_hits.extend(remaining_all)
                    # Trim to original page size (we may have added extra from fetch)
                    response["hits"]["hits"] = reranked_hits[:page_size]
                    logger.info(f"BRD Reranked: First 2 are new_launch products (out of {len(new_launch_hits)} new_launch, {len(other_hits)} other)")
                elif len(new_launch_hits) == 1:
                    reranked_hits = new_launch_hits + other_hits
                    response["hits"]["hits"] = reranked_hits[:page_size]
                    logger.info(f"BRD Reranked: Only 1 new_launch product found, placed first")
                else:
                    # No new_launch anywhere - FALLBACK: boost newest models by launch date
                    # For brand-only queries, users expect to see latest models first
                    is_brand_only = query_info.get("brand") and not query_info.get("category")
                    # Also check is_brand_only_phone_query for cases like "vivo" where category=smartphone is auto-assigned
                    is_brand_only_phone = query_info.get("is_brand_only_phone_query", False)
                    
                    if (is_brand_only or is_brand_only_phone) and other_hits:
                        from datetime import datetime
                        
                        def get_launch_date_score(hit):
                            """Extract model_launch_date and convert to sortable value"""
                            src = hit.get("_source", {})
                            launch_date_str = src.get("model_launch_date", "2000-01-01")
                            try:
                                # Parse date and return timestamp for sorting
                                dt = datetime.strptime(str(launch_date_str)[:10], "%Y-%m-%d")
                                return dt.timestamp()
                            except:
                                return 0  # Old/invalid dates go to bottom
                        
                        # Separate products with real launch dates (recent) vs placeholder "2000-01-01"
                        recent_launches = []
                        old_launches = []
                        
                        for hit in other_hits:
                            src = hit.get("_source", {})
                            launch_date = str(src.get("model_launch_date", "2000-01-01"))[:10]
                            if launch_date > "2020-01-01":  # Has real launch date
                                recent_launches.append(hit)
                            else:
                                old_launches.append(hit)
                        
                        if recent_launches:
                            # Sort recent launches by date (newest first)
                            recent_launches.sort(key=get_launch_date_score, reverse=True)
                            # Sort old launches by business score
                            old_launches.sort(key=lambda h: get_business_score(h, target_city), reverse=True)
                            
                            # Take first 2 newest as "virtual new_launch", rest by score
                            newest_2 = recent_launches[:2]
                            remaining = recent_launches[2:] + old_launches
                            remaining.sort(key=lambda h: get_business_score(h, target_city), reverse=True)
                            
                            reranked_hits = newest_2 + remaining
                            response["hits"]["hits"] = reranked_hits[:page_size]
                            
                            newest_names = [h.get("_source", {}).get("name", "")[:30] for h in newest_2]
                            logger.info(f"BRD Reranked: No new_launch_flag, boosted {len(newest_2)} newest by launch_date: {newest_names}")
                        else:
                            logger.info(f"BRD Reranked: No new_launch products and no recent launches, using score order")
                    else:
                        logger.info(f"BRD Reranked: No new_launch products found in ES, using score order")
            
            # =================== BRD REQ 3: MULTI-CATEGORY BRAND RERANKING ===================
            # When query is a brand-only query (e.g., "samsung", "vivo", "lg"), apply:
            # A. Show 2 SKUs per category, ranked by business score
            # B. Category ranking by product CATALOGUE distribution (total products in ES for brand)
            # C. City priority: user's city_id first, then citi_id_0
            # IMPORTANT: Preserve new_launch products in first 2 positions (from BRD Req 1)
            # Only apply on page 1, relevance sort, brand-only queries with no explicit category
            # NOTE: When user says "motorola mobile" or "oppo mobile", category IS set to "smartphone"
            # so we should NOT do multi-category reranking (it would pull in tablets/watches)
            # FIX: Also skip multi-category reranking when query has a model number (e.g., "nothing 4a", "samsung s24")
            # or model series keyword (e.g., "samsung fold", "hp victus", "lenovo thinkpad")
            # because user wants THAT specific model/series, not a variety of categories
            _processed_for_brand_check = query_info.get("processed", "").lower()
            _has_model_in_query = bool(re.search(r'(?<![a-z])\d+[a-z]|[a-z]\d+(?![a-z])', _processed_for_brand_check))
            # Also check for model series keywords (no digits, but specific product lines)
            _MODEL_SERIES_SKIP_KEYWORDS = {
                'fold', 'flip', 'ultra', 'plus', 'pro', 'lite', 'neo', 'fe',  # Samsung series
                'victus', 'omen', 'envy', 'spectre', 'pavilion', 'elitebook', 'probook',  # HP series
                'thinkpad', 'ideapad', 'legion', 'yoga', 'thinkbook', 'loq',  # Lenovo series
                'tuf', 'rog', 'vivobook', 'zenbook', 'proart',  # Asus series
                'inspiron', 'latitude', 'xps', 'alienware', 'vostro',  # Dell series
                'nitro', 'predator', 'aspire', 'swift', 'spin',  # Acer series
                'macbook', 'imac', 'ipad', 'airpods',  # Apple series
                'bravia', 'walkman', 'alpha',  # Sony series
                'matebook', 'matepad', 'freebuds',  # Huawei series
                'surface',  # Microsoft series
                'galaxy', 'note', 'book',  # Generic high-value series (book for Galaxy Book)
            }
            _query_words_for_check = set(_processed_for_brand_check.split())
            _has_series_keyword = bool(_query_words_for_check & _MODEL_SERIES_SKIP_KEYWORDS)
            is_brand_only = (
                query_info.get("brand") and not query_info.get("category") and not query_info.get("is_apple_query")
                and not _has_model_in_query  # Skip if query contains model pattern like "4a", "3a", "s24"
                and not _has_series_keyword  # Skip if query contains series keyword like "fold", "victus"
            )
            if (_has_model_in_query or _has_series_keyword) and query_info.get("brand"):
                logger.info(f"BRD Req3: Skipping multi-category rerank for model/series query: '{_processed_for_brand_check}'")
            # Only apply if multiple categories present and it's relevance sort
            if total > 0 and is_brand_only and is_relevance_sort and page == 1:
                hits = response["hits"]["hits"]
                
                # BRD Req 1+3 integration: Separate new_launch (first 2) from rest
                preserved_nl_hits = []
                for i, hit in enumerate(hits):
                    src = hit.get("_source", {})
                    is_nl = False
                    for co in src.get("city_offers", []):
                        if co.get("new_launch_flag") in (True, 1, "1", "true"):
                            is_nl = True
                            break
                    if is_nl and len(preserved_nl_hits) < 2:
                        preserved_nl_hits.append(hit)
                
                brand_name = query_info.get("brand", "")
                target_city = city_id if city_id else "0"
                norm_target = _normalize_city(target_city)
                
                # BRD Req3: Fetch top 2 per category via ES aggregation (covers ALL categories)
                # Uses terms agg on actual_category + top_hits sub-agg sorted by city score
                try:
                    city_sort_script = f"""
                        double score = 0;
                        if (doc.containsKey('city_offers.score') && doc['city_offers.score'].size() > 0 &&
                            doc.containsKey('city_offers.cityid') && doc['city_offers.cityid'].size() > 0) {{
                            for (int i = 0; i < doc['city_offers.cityid'].length; i++) {{
                                if (doc['city_offers.cityid'][i] == '{norm_target}') {{
                                    return doc['city_offers.score'][i];
                                }}
                                if (doc['city_offers.cityid'][i] == 'citi_id_0') {{
                                    score = doc['city_offers.score'][i];
                                }}
                            }}
                        }}
                        return score;
                    """
                    
                    # Build filter for user's city or citi_id_0 availability
                    city_filter = {"nested": {
                        "path": "city_offers",
                        "query": {"bool": {"should": [
                            {"term": {"city_offers.cityid": norm_target}},
                            {"term": {"city_offers.cityid": "citi_id_0"}}
                        ]}}
                    }}
                    
                    cat_top_hits_query = {
                        "size": 0,
                        "query": {"bool": {
                            "must": [{"wildcard": {"manufacturer_desc": {"value": f"*{brand_name}*", "case_insensitive": True}}}],
                            "filter": [city_filter]
                        }},
                        "aggs": {
                            "cats": {
                                "terms": {"field": "actual_category", "size": 50},
                                "aggs": {
                                    "top_products": {
                                        "top_hits": {
                                            "size": original_page_size,  # Fetch enough to fill page when few categories
                                            "sort": [{"_script": {
                                                "type": "number",
                                                "script": {"source": city_sort_script},
                                                "order": "desc"
                                            }}],
                                            "_source": True
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    cat_resp = self.es.search(index=self.index, body=cat_top_hits_query)
                    cat_buckets = cat_resp.get("aggregations", {}).get("cats", {}).get("buckets", [])
                    
                    # Build catalogue distribution + category hits from aggregation
                    # Deduplicate by modelid (since collapse isn't supported in top_hits)
                    catalog_category_counts = {}
                    category_hits = {}  # category -> [hit, hit, ...] (dynamic limit based on category count)
                    
                    # BRD Req3 fix: Calculate per-category limit dynamically to fill page
                    # If there are fewer categories, allow more products per category
                    num_categories = len(cat_buckets)
                    if num_categories == 0:
                        per_cat_limit = original_page_size
                    else:
                        # Distribute page size across categories, minimum 2
                        per_cat_limit = max(2, (original_page_size + num_categories - 1) // num_categories)
                    logger.info(f"BRD Req3: {num_categories} categories found, per_cat_limit={per_cat_limit}")
                    
                    for bucket in cat_buckets:
                        cat_name = bucket["key"]
                        cat_count = bucket["doc_count"]
                        catalog_category_counts[cat_name] = cat_count
                        cat_products = bucket.get("top_products", {}).get("hits", {}).get("hits", [])
                        # Deduplicate by modelid, keep first (highest score) per model
                        seen_models = set()
                        deduped = []
                        for hit in cat_products:
                            mid = str(hit.get("_source", {}).get("modelid", ""))
                            if mid not in seen_models:
                                seen_models.add(mid)
                                deduped.append(hit)
                                if len(deduped) >= per_cat_limit:
                                    break
                        category_hits[cat_name] = deduped
                    
                    logger.info(f"BRD Req3: Catalogue distribution for '{brand_name}': {catalog_category_counts}")
                    
                    if len(category_hits) > 1:
                        # Sort categories by catalogue distribution (most products first)
                        sorted_categories = sorted(category_hits.keys(),
                                                   key=lambda c: catalog_category_counts.get(c, 0), reverse=True)
                        
                        # BRD 3A: Take top SKUs per category, ordered by category rank
                        # Exclude model_ids already in preserved_nl_hits
                        nl_model_ids = set()
                        for nlh in preserved_nl_hits:
                            mid = str(nlh.get("_source", {}).get("modelid", ""))
                            if mid:
                                nl_model_ids.add(mid)
                        
                        multi_cat_hits = []
                        for cat in sorted_categories:
                            for hit in category_hits[cat]:
                                mid = str(hit.get("_source", {}).get("modelid", ""))
                                if mid not in nl_model_ids:
                                    multi_cat_hits.append(hit)
                        
                        # Final merge: preserved new_launch first, then multi-category ranked
                        final_hits = preserved_nl_hits + multi_cat_hits
                        
                        # BRD Req3 FIX: Backfill if multi-category doesn't fill page
                        # When a brand has fewer unique models than page_size, backfill from original hits
                        if len(final_hits) < original_page_size:
                            existing_model_ids = set(mid for mid in nl_model_ids)
                            for h in final_hits:
                                mid = str(h.get("_source", {}).get("modelid", ""))
                                if mid:
                                    existing_model_ids.add(mid)
                            # Backfill from original hits (which were sorted by score)
                            for h in hits:
                                if len(final_hits) >= original_page_size:
                                    break
                                mid = str(h.get("_source", {}).get("modelid", ""))
                                if mid not in existing_model_ids:
                                    final_hits.append(h)
                                    existing_model_ids.add(mid)
                            logger.info(f"BRD Req3: Backfilled to {len(final_hits)} products (original multi-cat had less)")
                        
                        response["hits"]["hits"] = final_hits[:original_page_size]
                        
                        # Log what categories made it to the final page
                        final_cats = {}
                        for h in response["hits"]["hits"]:
                            c = h.get("_source", {}).get("actual_category", "?")
                            final_cats[c] = final_cats.get(c, 0) + 1
                        logger.info(f"BRD Req3: Multi-category brand rerank applied (preserved {len(preserved_nl_hits)} NL). "
                                    f"Categories in page: {final_cats}. Catalogue: {dict(list(catalog_category_counts.items())[:5])}")
                    else:
                        # Single category brand — keep original order with NL preserved
                        if preserved_nl_hits:
                            others = [h for h in hits if id(h) not in set(id(n) for n in preserved_nl_hits)]
                            response["hits"]["hits"] = (preserved_nl_hits + others)[:original_page_size]
                        else:
                            response["hits"]["hits"] = hits[:original_page_size]
                        logger.info(f"BRD Req3: Only 1 category — no multi-category reranking needed")
                    
                except Exception as cat_err:
                    logger.warning(f"BRD Req3: Category aggregation failed: {cat_err}. Using original order.")
                    response["hits"]["hits"] = hits[:original_page_size]
            
            # BRD Req3: Trim over-fetched results back to original_page_size
            if _is_brand_only_prefetch and len(response["hits"]["hits"]) > original_page_size:
                response["hits"]["hits"] = response["hits"]["hits"][:original_page_size]
            
            # Model query: Trim over-fetched results back to original_page_size
            if _is_model_query_prefetch and len(response["hits"]["hits"]) > original_page_size:
                response["hits"]["hits"] = response["hits"]["hits"][:original_page_size]
                logger.info(f"Model query: Trimmed results from {page_size} to {original_page_size}")
            
            # Flag-based query: Trim over-fetched results back to original_page_size
            if _is_flag_query and len(response["hits"]["hits"]) > original_page_size:
                response["hits"]["hits"] = response["hits"]["hits"][:original_page_size]
                logger.info(f"Flag query: Trimmed results from {page_size} to {original_page_size}")
            
            # Restore page_size for downstream code
            page_size = original_page_size
            
            # =================== AUDIO DEVICE BRAND FALLBACK ===================
            # When "boat earbuds" returns only soundbars (boat has no earbuds in index),
            # retry without brand filter so earbuds from other brands show up.
            # This handles brands that don't have specific audio device types.
            # Includes typo variants so "boat erphn", "boat earfon", "boat nekband" etc. also trigger.
            original_lower = query_info.get("original", "").lower()
            _oq_words_fb = set(original_lower.split())
            _audio_device_kw_fb = {
                "earbuds", "earbud", "earbudz", "earbudss", "erbud", "erbuds", "airbud", "airbuds", "earbds", "earbd",
                "earphone", "earphones", "earphn", "earphon", "earfon", "earfone", "earpone",
                "erphn", "erpone", "erphone", "erfone", "erfon", "earphne", "eaphone", "earpho",
                "raphon", "raphne", "raphone", "raphn", "airphon", "airphone", "airphn", "airfon", "airfone",
                "headphone", "headphones", "headfon", "headfone", "hedphone", "headphn", "haedphone", "headphne", "headphon",
                "headset", "headsets",
                "neckband", "neckbands", "neckbnd", "nekband", "neckbad", "nckband", "necband",
                "airdopes", "airdope", "tws",
                "airpods", "airpod", "airpodz", "airpodss", "arpods", "arpod", "aiepods", "airpds", "aripods", "aripod",
            }
            _is_audio_device_fb = bool(_audio_device_kw_fb & _oq_words_fb) or \
                                  any(kw in original_lower for kw in ["ear bud", "ear phone", "neck band", "air pod", "air pods"])
            
            if query_info.get("brand") and query_info.get("category") == "audio video" and _is_audio_device_fb:
                # Case 1: total == 0 → brand has no matching audio products at all, drop brand immediately
                # Case 2: total > 0 AND brand has significant products → KEEP brand (user wants this brand)
                # Case 3: total > 0 but few products AND no matching device type → drop brand
                _should_drop_brand = False
                if total == 0:
                    _should_drop_brand = True
                elif total > 20:
                    # Brand has many products - user explicitly wants this brand, keep it
                    # e.g., "boat headphone" returns 73 boAt speakers/soundbars - show those
                    _should_drop_brand = False
                    logger.info(f"Brand '{query_info.get('brand')}' has {total} audio products - keeping brand filter")
                else:
                    # Check if ANY of the top results match the user's audio device type
                    # BUT exclude products that are clearly speakers/soundbars even if they contain "TWS" in name
                    _device_check_words = ["earbuds", "earbud", "buds", "tws", "earphone", "headphone",
                                           "headset", "neckband", "airdopes", "truly wireless",
                                           "wireless earbuds", "wireless earphone"]
                    _speaker_indicators = ["speaker", "soundbar", "sound bar", "party", "home theater",
                                           "home theatre", "trolley", "tower"]
                    _has_matching_device = False
                    for hit in response["hits"]["hits"][:10]:
                        pn_lower = hit.get("_source", {}).get("product_name", "").lower()
                        # Skip if product is clearly a speaker/soundbar
                        if any(si in pn_lower for si in _speaker_indicators):
                            continue
                        if any(dw in pn_lower for dw in _device_check_words):
                            _has_matching_device = True
                            break
                    if not _has_matching_device:
                        _should_drop_brand = True
                
                if _should_drop_brand:
                    logger.info(f"Brand '{query_info.get('brand')}' has no audio device products (total={total}), retrying without brand...")
                    fallback_info = query_info.copy()
                    fallback_info["brand"] = None  # Remove brand filter, keep category
                    # Also strip brand name from processed query so text search doesn't
                    # pull back the same brand's soundbars/speakers via multi_match
                    _brand_lower = (query_info.get("brand") or "").lower()
                    _proc = fallback_info.get("processed", "")
                    import re as _re
                    _proc = _re.sub(r'\b' + _re.escape(_brand_lower) + r'\b', '', _proc, flags=_re.IGNORECASE).strip()
                    _proc = _re.sub(r'\s+', ' ', _proc)  # collapse double spaces
                    fallback_info["processed"] = _proc if _proc else fallback_info.get("original", "")
                    fallback_query = self.build_query(fallback_info, city_id, filters, page, page_size, emi_range, sort_by)
                    response = self.es.search(index=self.index, body=fallback_query, request_timeout=2)
                    total = response["hits"]["total"]["value"]
                    logger.info(f"Audio device brand fallback returned {total} results")
            
            # =================== FALLBACK MECHANISM ===================
            # Priority: Keep category, remove brand first (for queries like "Bajaj mixer grinder")
            # This ensures we show mixer grinders from other brands if Bajaj doesn't have them
            
            # Detect Google Pixel queries early - we don't want to remove brand/category for these
            is_google_pixel_query = query_info.get("brand") == "google" and any(kw in original_lower for kw in ["pixel", "pixle", "pixxel", "pixal", "pxel", "piksel"])
            
            # FALLBACK 1: If no results AND has both brand AND category, try WITHOUT brand first
            # This prioritizes showing category results (e.g., show other mixer grinders)
            # SKIP for Google Pixel queries - we want to keep Google brand filter
            # SKIP if running low on time budget
            if total == 0 and query_info.get("brand") and query_info.get("category") and not query_info.get("is_apple_query") and not is_google_pixel_query and not should_skip_fallback():
                logger.info(f"Zero results with brand+category, trying without brand (keeping category)...")
                fallback_info = query_info.copy()
                fallback_info["brand"] = None  # Remove brand, keep category
                fallback_query = self.build_query(fallback_info, city_id, filters, page, page_size, emi_range, sort_by)
                response = self.es.search(index=self.index, body=fallback_query, request_timeout=2)
                total = response["hits"]["total"]["value"]
                logger.info(f"Category-only fallback returned {total} results")
            
            # FALLBACK 1.5: If no results with city_id, try WITHOUT city_id restriction (show citi_id_0 products)
            # This ensures products available globally (citi_id_0) are shown even if not in user's city
            # IMPORTANT: Keep category filter - don't show random products from other categories!
            # SKIP if running low on time budget
            if total == 0 and city_id and city_id != "citi_id_0" and query_info.get("category") and not should_skip_fallback():
                logger.info(f"Zero results with city_id={city_id} and category={query_info.get('category')}, trying WITHOUT city restriction...")
                fallback_info = query_info.copy()
                # Build query WITHOUT city_id to show citi_id_0 products
                fallback_query = self.build_query(fallback_info, None, filters, page, page_size, emi_range, sort_by)
                response = self.es.search(index=self.index, body=fallback_query, request_timeout=2)
                total = response["hits"]["total"]["value"]
                logger.info(f"City-relaxed fallback (keeping category) returned {total} results")
            
            # FALLBACK 2: If still no results, try without category (but keep brand if no category was detected)
            # SKIP for Google Pixel queries - we don't want to show random "pixel" products
            # SKIP if city_id was provided - we already tried city-relaxed fallback above
            # SKIP if running low on time budget
            if total == 0 and query_info.get("category") and not query_info.get("is_apple_query") and not is_google_pixel_query and not city_id and not should_skip_fallback():
                logger.info(f"Zero results with category filter, trying without category...")
                fallback_info = query_info.copy()
                fallback_info["category"] = None
                fallback_query = self.build_query(fallback_info, city_id, filters, page, page_size, emi_range, sort_by)
                response = self.es.search(index=self.index, body=fallback_query, request_timeout=2)
                total = response["hits"]["total"]["value"]
                logger.info(f"Fallback search returned {total} results")
            
            # FALLBACK 3: If still no results with brand filter, try without brand filter too
            # SKIP for Google Pixel queries
            # SKIP if city_id was provided - we must preserve category to avoid wrong SKUs!
            # SKIP if running low on time budget
            _has_city_filter = bool(city_id and city_id != "citi_id_0")
            if total == 0 and query_info.get("brand") and not query_info.get("is_apple_query") and not is_google_pixel_query and not _has_city_filter and not should_skip_fallback():
                logger.info(f"Zero results with brand filter, trying broader search...")
                fallback_info = query_info.copy()
                fallback_info["category"] = None
                fallback_info["brand"] = None
                fallback_query = self.build_query(fallback_info, None, filters, page, page_size, emi_range, sort_by)
                response = self.es.search(index=self.index, body=fallback_query, request_timeout=2)
                total = response["hits"]["total"]["value"]
                logger.info(f"Broad fallback search returned {total} results")
            
            # FALLBACK 3.5: If city_id was provided and still no results, try without brand + city but KEEP category
            # This ensures we never show wrong SKUs when city_id is provided
            if total == 0 and _has_city_filter and query_info.get("brand") and query_info.get("category") and not query_info.get("is_apple_query") and not is_google_pixel_query and not should_skip_fallback():
                logger.info(f"[CITY-SAFE FALLBACK] Zero results, trying without brand + city but keeping category={query_info.get('category')}...")
                fallback_info = query_info.copy()
                fallback_info["brand"] = None  # Remove brand
                # KEEP category! Build query WITHOUT city_id
                fallback_query = self.build_query(fallback_info, None, filters, page, page_size, emi_range, sort_by)
                response = self.es.search(index=self.index, body=fallback_query, request_timeout=2)
                total = response["hits"]["total"]["value"]
                logger.info(f"[CITY-SAFE FALLBACK] Brand+City relaxed (category={query_info.get('category')}) returned {total} results")
            
            # FALLBACK 4: For specific phone model queries (like "vivo x300", "oppo find x9", "realme 15", "infinix 50x")
            # If no results with processed query, try with just brand + category
            # This ensures user sees phones from that brand even if model doesn't exist
            # SKIP if running low on time budget
            if total == 0 and query_info.get("brand") and not should_skip_fallback():
                original = query_info.get("original", "").lower()
                brand = query_info.get("brand", "").lower()
                
                # List of phone brands for fallback
                phone_brands = {"vivo", "oppo", "realme", "samsung", "redmi", "poco", "infinix", 
                               "tecno", "motorola", "nokia", "oneplus", "iqoo", "nothing", "lava",
                               "xiaomi", "mi", "honor", "itel", "google"}
                
                # Check if this is a phone brand with a model that might not exist
                # Match patterns like: "vivo x300", "oppo find x9", "realme 15", "infinix 50x"
                # More flexible pattern: brand + any words/numbers
                if brand in phone_brands:
                    # Check if query has numbers (model number) or series letters
                    has_model_indicator = re.search(r'\d+|[xyzafkrscpn]\s*\d+', original.replace(brand, ""))
                    if has_model_indicator:
                        logger.info(f"Phone model not found for '{original}', showing other phones from brand '{brand}'...")
                        fallback_info = query_info.copy()
                        fallback_info["processed"] = brand  # Just use brand name
                        fallback_info["category"] = "smartphone"  # Force smartphone category
                        fallback_query = self.build_query(fallback_info, city_id, filters, page, page_size, emi_range, sort_by)
                        response = self.es.search(index=self.index, body=fallback_query, request_timeout=2)
                        total = response["hits"]["total"]["value"]
                        logger.info(f"Brand-only phone fallback returned {total} results")
            
            # =================== FALLBACK 4.5: TWO WHEELER MODEL WITHOUT SKU ===================
            # Special handling for models like Pulsar, Chetak that we don't have SKU for
            # Pulsar → show other brand motorcycles/bikes
            # Chetak → show other brand scooters
            if total == 0:
                original = query_info.get("original", "").lower()
                
                # Models we don't have SKU for - map to their subcategory
                MODELS_NO_SKU = {
                    # Bikes/Motorcycles - show other brand bikes
                    "pulsar": "motorcycle", "pulser": "motorcycle", "plsar": "motorcycle",
                    "dominar": "motorcycle", "avenger": "motorcycle", "platina": "motorcycle",
                    # Scooters - show other brand scooters  
                    "chetak": "scooter", "chetk": "scooter", "chetek": "scooter",
                }
                
                detected_model = None
                subcategory = None
                for model, subcat in MODELS_NO_SKU.items():
                    if model in original:
                        detected_model = model
                        subcategory = subcat
                        break
                
                if detected_model and subcategory:
                    logger.info(f"Model '{detected_model}' not in SKU, showing other {subcategory}s...")
                    fallback_info = query_info.copy()
                    fallback_info["brand"] = None  # Remove brand filter
                    fallback_info["category"] = "two wheeler"
                    
                    # Use subcategory-specific search
                    if subcategory == "scooter":
                        fallback_info["processed"] = "scooter"
                        fallback_info["two_wheeler_subcategory"] = "scooter"
                    else:  # motorcycle
                        fallback_info["processed"] = "motorcycle bike"
                        fallback_info["two_wheeler_subcategory"] = "motorcycle"
                    
                    fallback_query = self.build_query(fallback_info, city_id, filters, page, page_size, emi_range, sort_by)
                    response = self.es.search(index=self.index, body=fallback_query, request_timeout=2)
                    total = response["hits"]["total"]["value"]
                    logger.info(f"Two wheeler fallback ({subcategory}) returned {total} results")
            
            # =================== FALLBACK 5: SKU NAME FOCUSED SEARCH ===================
            # If still no results, try a more relaxed search with higher sku_name weight
            # This helps find products where the model name is in sku_name but not product_name
            # SKIP for Apple queries - Apple products must ONLY show Apple brand
            # SKIP if user has active attribute filters - their filter combo yielded 0 results;
            # do not override with unfiltered data (post_filter is not applied in raw fallback queries)
            has_active_attr_filters = bool(filters and any(
                k not in ("emi", "price_range", "price_max") and v
                for k, v in filters.items()
            ))
            # Also treat NL-derived post_filters (star, door, color) as active attribute filters
            # so Fallbacks 5 & 6 don't override them when their combo yields 0 results
            if not has_active_attr_filters:
                nl_attrs = query_info.get("attributes") or {}
                has_active_attr_filters = bool(
                    nl_attrs.get("_star_post_filter") or
                    nl_attrs.get("_door_type_post_filter") or
                    nl_attrs.get("_color_post_filter") or
                    nl_attrs.get("_wm_type_post_filter")
                )
            if total == 0 and query_info.get("processed") and not query_info.get("is_apple_query") and not has_active_attr_filters and not should_skip_fallback():
                logger.info(f"Zero results, trying sku_name focused fallback search...")
                original_query = query_info.get("original", "")
                
                # FIX: For short alphanumeric model queries (4a, 3a, 5g, etc.), boost exact word match
                _is_short_model_query = bool(re.match(r'^[a-zA-Z]?\d+[a-zA-Z]?$', original_query.strip()))
                
                # Build should clauses with exact model boost for short queries
                should_clauses = []
                
                if _is_short_model_query:
                    # HIGHEST priority: exact word match like " 4a " or " Mi 4A "
                    should_clauses.append({"regexp": {"search_field": {"value": f".* {original_query} .*", "case_insensitive": True, "boost": 20}}})
                    should_clauses.append({"regexp": {"sku_name": {"value": f".* {original_query} .*", "case_insensitive": True, "boost": 20}}})
                    # Also try as word at end like "model 4a"
                    should_clauses.append({"regexp": {"search_field": {"value": f".* {original_query}$", "case_insensitive": True, "boost": 15}}})
                    logger.info(f"Short model query '{original_query}' - adding exact word match boost")
                
                # Standard fallback clauses
                should_clauses.extend([
                    # High weight for exact/near match in sku_name
                    {"match": {"sku_name": {"query": original_query, "fuzziness": "AUTO", "boost": 5}}},
                    # Also try search_field with original query
                    {"match": {"search_field": {"query": original_query, "fuzziness": "AUTO", "boost": 3}}},
                    # Wildcard for partial match
                    {"wildcard": {"sku_name": {"value": f"*{original_query.replace(' ', '*')}*", "case_insensitive": True, "boost": 2}}},
                ])
                
                # Build a special query focusing on sku_name field with fuzzy matching
                # FIX: Add category filter if category was detected to avoid showing wrong products
                sku_fallback_bool = {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
                if query_info.get("category"):
                    sku_fallback_bool["filter"] = [{"term": {"actual_category": query_info["category"]}}]
                
                sku_fallback_query = {
                    "query": {
                        "bool": sku_fallback_bool
                    },
                    "from": (page - 1) * page_size,
                    "size": page_size,
                    "sort": [
                        {"_score": {"order": "desc"}},
                        {"model_launch_date": {"order": "desc"}},
                        {"modelid": {"order": "asc"}}  # Unique tie-breaker for pagination
                    ],
                    "track_total_hits": True,
                    "collapse": {
                        "field": "modelid",
                        "inner_hits": {
                            "name": "sku_variants",
                            "size": 10
                        }
                    }
                }
                
                try:
                    fallback_response = self.es.search(index=self.index, body=sku_fallback_query, request_timeout=2)
                    fallback_total = fallback_response["hits"]["total"]["value"]
                    if fallback_total > 0:
                        logger.info(f"SKU name focused fallback returned {fallback_total} results")
                        response = fallback_response
                        total = fallback_total
                except Exception as e:
                    logger.warning(f"SKU fallback search failed: {e}")
            
            # =================== FALLBACK 6: AGGRESSIVE FUZZY SEARCH ===================
            # If STILL no results, use aggressive character-level fuzzy matching
            # This handles completely unknown typos that aren't in our dictionaries
            # SKIP for Apple queries - Apple products must ONLY show Apple brand
            # SKIP if user has active attribute filters - same reason as Fallback 5
            # SKIP if running low on time budget
            if total == 0 and query_info.get("original") and not query_info.get("is_apple_query") and not has_active_attr_filters and not should_skip_fallback():
                original_query = query_info.get("original", "").strip()
                logger.info(f"Zero results after all fallbacks, trying aggressive fuzzy search for: '{original_query}'")
                
                # Split query into words and build fuzzy match for each
                words = original_query.split()
                
                # Build aggressive fuzzy query
                should_clauses = []
                for word in words:
                    if len(word) >= 3:  # Only fuzzy match words with 3+ characters
                        # Fuzzy match on multiple fields with high fuzziness
                        should_clauses.append({
                            "multi_match": {
                                "query": word,
                                "fields": [
                                    "search_field^3",
                                    "product_name^2",
                                    "manufacturer_desc^2",
                                    "actual_category",
                                    "sku_name"
                                ],
                                "fuzziness": "2",  # Allow up to 2 character edits
                                "prefix_length": 1,
                                "boost": 3
                            }
                        })
                        # Also try wildcard for partial matches
                        should_clauses.append({
                            "wildcard": {
                                "search_field": {
                                    "value": f"*{word}*",
                                    "case_insensitive": True,
                                    "boost": 1
                                }
                            }
                        })
                
                # Add full query fuzzy match
                should_clauses.append({
                    "match": {
                        "search_field": {
                            "query": original_query,
                            "fuzziness": "AUTO",
                            "boost": 2
                        }
                    }
                })
                
                if should_clauses:
                    # FIX: Add category filter if category was detected
                    fuzzy_bool = {
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                    if query_info.get("category"):
                        fuzzy_bool["filter"] = [{"term": {"actual_category": query_info["category"]}}]
                    
                    fuzzy_fallback_query = {
                        "query": {
                            "bool": fuzzy_bool
                        },
                        "from": (page - 1) * page_size,
                        "size": page_size,
                        "sort": [
                            {"_score": {"order": "desc"}},
                            {"popularity_score_num": {"order": "desc", "missing": "_last"}},
                            {"modelid": {"order": "asc"}}
                        ],
                        "track_total_hits": True,
                        "collapse": {
                            "field": "modelid",
                            "inner_hits": {
                                "name": "sku_variants",
                                "size": 10
                            }
                        }
                    }
                    
                    try:
                        fuzzy_response = self.es.search(index=self.index, body=fuzzy_fallback_query, request_timeout=2)
                        fuzzy_total = fuzzy_response["hits"]["total"]["value"]
                        if fuzzy_total > 0:
                            logger.info(f"Aggressive fuzzy fallback returned {fuzzy_total} results")
                            response = fuzzy_response
                            total = fuzzy_total
                    except Exception as e:
                        logger.warning(f"Aggressive fuzzy fallback search failed: {e}")
            
            # =================== PAGINATION FIX ===================
            # Use cardinality aggregation for accurate unique model count
            # This gives the true count of unique models, not SKUs
            aggregations = response.get("aggregations", {})
            
            # Handle both cardinality structures:
            # 1. Plain cardinality (no attribute filters): {"value": N}
            # 2. Filter-wrapped cardinality (attribute filters active):
            #    {"doc_count": N, "filtered_count": {"value": N}}
            unique_models_agg = aggregations.get("unique_models", {})
            if "filtered_count" in unique_models_agg:
                # Attribute filters active → use filtered cardinality (correct count)
                unique_model_count = unique_models_agg["filtered_count"].get("value", total)
                logger.info(f"Using filter-wrapped cardinality: {unique_model_count} (attribute filter active)")
            else:
                # No attribute filters → plain cardinality
                unique_model_count = unique_models_agg.get("value", total)
            
            # If cardinality count is available and > 0, use it
            # Otherwise fall back to ES total (which is correct with collapse)
            if unique_model_count > 0:
                accurate_total = unique_model_count
            else:
                accurate_total = total
            
            logger.info(f"Pagination: ES total={total}, unique_models={unique_model_count}, returning={accurate_total}")
            
            result = {
                "success": True,
                "hits": response["hits"]["hits"],
                "total": accurate_total,  # Use accurate unique model count
                "aggregations": aggregations,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_models": accurate_total,
                    "total_pages": (accurate_total + page_size - 1) // page_size if accurate_total > 0 else 0,
                    "has_next": (page * page_size) < accurate_total
                }
            }

            # Propagate progressive filter relaxation metadata
            _relaxed = query_info.get("_relaxed_filters")
            if _relaxed:
                result["relaxed_filters"] = _relaxed

            return result
        except Exception as e:
            logger.error(f"Search error: {str(e)}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "hits": [],
                "total": 0,
                "aggregations": {}
            }


# =================== RESPONSE FORMATTER ===================
class ResponseFormatter:
    """Formats search results into API response structure"""
    
    # Normalize inconsistent category names from ES data
    CATEGORY_NORMALIZE = {
        "furnitures": "furniture",
    }
    
    @staticmethod
    def get_city_offer(hit: Dict, city_id: str) -> Optional[Dict]:
        """Extract city-specific offer from product data"""
        city_offer = None
        source = hit.get("_source", {})
        product_name = source.get("product_name", "Unknown")[:30]
        
        # Check inner_hits first
        if "inner_hits" in hit and "city_offers" in hit["inner_hits"]:
            city_hits = hit["inner_hits"]["city_offers"]["hits"]["hits"]
            logger.info(f"[{product_name}] Found {len(city_hits)} city_offers in inner_hits for city_id={city_id}")
            
            # First pass: look for exact city_id match
            for city_hit in city_hits:
                city_source = city_hit["_source"]
                if city_source.get("cityid") == city_id:
                    city_offer = city_source
                    logger.info(f"[{product_name}] Found exact city match: {city_id}, price={city_source.get('offer_price')}")
                    break
            
            # Second pass: fallback to citi_id_0
            if not city_offer:
                for city_hit in city_hits:
                    city_source = city_hit["_source"]
                    if city_source.get("cityid") == "citi_id_0":
                        city_offer = city_source
                        logger.info(f"[{product_name}] Using fallback citi_id_0, price={city_source.get('offer_price')}")
                        break
        else:
            logger.info(f"[{product_name}] No city_offers in inner_hits - keys: {list(hit.get('inner_hits', {}).keys())}")
        
        # Fallback to source if inner_hits not available
        if not city_offer and "city_offers" in source:
            logger.info(f"[{product_name}] Checking city_offers in source (no inner_hits)")
            for offer in source["city_offers"]:
                if offer.get("cityid") == city_id:
                    city_offer = offer
                    break
            if not city_offer:
                for offer in source["city_offers"]:
                    if offer.get("cityid") == "citi_id_0":
                        city_offer = offer
                        break
        
        return city_offer
    
    @staticmethod
    def update_image_url(source: Dict) -> Dict:
        """Update image URLs with domain"""
        if "image" in source and source["image"] and not source["image"].startswith("http"):
            source["image"] = IMAGE_DOMAIN + source["image"]
        
        if "products" in source:
            for prod in source["products"]:
                if "image" in prod and prod["image"] and not prod["image"].startswith("http"):
                    prod["image"] = IMAGE_DOMAIN + prod["image"]
        
        return source
    
    @staticmethod
    def parse_filter_aggregations(aggs_response: Dict, category: str = None) -> Dict:
        """Parse aggregations into filter format"""
        filters = {}
        
        # Get attributes to exclude for this category
        attributes_to_remove = set()
        if category:
            category_lower = category.lower()
            if category_lower in CATEGORY_FILTER_EXCLUSIONS_LOWER:
                attributes_to_remove = set(CATEGORY_FILTER_EXCLUSIONS_LOWER[category_lower])
            else:
                for cat_key, exclusions in CATEGORY_FILTER_EXCLUSIONS_LOWER.items():
                    if cat_key in category_lower or category_lower in cat_key:
                        attributes_to_remove.update(exclusions)
        
        for option in ALL_ATTRIBUTE_OPTIONS:
            es_key = option['es_key']
            display_key = option["display_key"].replace("_value", "")
            
            # Skip excluded attributes
            if es_key in attributes_to_remove:
                continue
            
            if es_key in aggs_response:
                buckets = aggs_response[es_key]["buckets"]
                values = []
                id_name_map = ATTRIBUTE_ID_NAME_MAP.get(es_key, {})
                
                for bucket in buckets:
                    if bucket["key"] == "__missing__":
                        continue
                    attr_id = str(bucket["key"])
                    attr_name = id_name_map.get(attr_id, attr_id)
                    values.append({
                        "id": attr_id,
                        "name": attr_name,
                        "count": bucket["doc_count"]
                    })
                
                if values:
                    filters[display_key] = values
        
        return filters
    
    @classmethod
    def format_response(cls, search_result: Dict, city_id: str = None, 
                        query_info: Dict = None, emi_range: Dict = None) -> Dict:
        """Format search results into final API response"""
        
        # Get original and processed queries
        # Get original query - prefer user_original_query (actual user input) over "original" (cleaned)
        original_query = query_info.get("user_original_query", query_info.get("original", "")) if query_info else ""
        processed_query = query_info.get("processed", "") if query_info else ""
        
        # =================== CORRECTED QUERY LOGIC ===================
        # Use the comprehensive generate_corrected_query function
        # This handles: typos, short forms, noise, canonical names
        corrected_query = generate_corrected_query(original_query, query_info, search_result)
        
        # Log correction if made
        if corrected_query:
            logger.info(f"Query corrected: '{original_query}' → '{corrected_query}'")
        
        # suggested_search_keyword shows what we searched for (corrected or original)
        suggested_keyword = corrected_query if corrected_query else (original_query or "*")
        
        # =================== APPLIED FILTERS FROM QUERY ===================
        # Show what filters were parsed from the query (EMI/price patterns)
        applied_filters = {}
        if query_info:
            parsed_filters = query_info.get("parsed_filters", {})
            if parsed_filters:
                if "lowest_emi" in parsed_filters:
                    emi_filter = parsed_filters["lowest_emi"]
                    applied_filters["emi"] = {
                        "min": emi_filter.get("gte"),
                        "max": emi_filter.get("lte")
                    }
                if "mop" in parsed_filters:
                    price_filter = parsed_filters["mop"]
                    applied_filters["price"] = {
                        "min": price_filter.get("gte"),
                        "max": price_filter.get("lte")
                    }
        
        # =================== PAGINATION METADATA ===================
        # Extract pagination info from search_result if available
        pagination_info = search_result.get("pagination", {})
        total_records = search_result.get("total", 0)
        
        # Build response matching reference structure with PostV1Productlist wrapper
        # Reference format: data.PostV1Productlist.status/message/data
        final_response = {
            "data": {
                "PostV1Productlist": {
                    "status": True,
                    "message": "Success",
                    "data": {
                        "products": [],
                        "totalrecords": total_records,
                        "suggested_search_keyword": suggested_keyword,
                        "corrected_query": corrected_query,
                        "applied_filters": applied_filters if applied_filters else None,
                        "relaxed_filters": search_result.get("relaxed_filters"),
                        "filters": []
                    }
                }
            }
        }
        
        if not search_result.get("success"):
            final_response["data"]["PostV1Productlist"]["status"] = False
            final_response["data"]["PostV1Productlist"]["message"] = search_result.get("error", "Search failed")
            return final_response
        
        hits = search_result.get("hits", [])
        aggs = search_result.get("aggregations", {})
        
        emi_values = []
        seen_models = set()
        detected_category = None
        
        for hit in hits:
            source = hit["_source"]
            source = cls.update_image_url(source)
            
            # Get category for filter exclusions
            if not detected_category:
                detected_category = source.get("actual_category", "")
            
            # Get city offer
            city_offer = cls.get_city_offer(hit, city_id) if city_id else None
            if city_id and not city_offer:
                continue
            
            # =================== EMI RANGE VALIDATION ===================
            # If EMI range filter is applied, validate that the displayed EMI 
            # (which may be from fallback city) is within the requested range
            # This prevents showing products with fallback EMI outside user's filter
            if emi_range and city_offer:
                displayed_emi = city_offer.get("lowest_emi", 0)
                emi_min = emi_range.get("gte", 0)
                emi_max = emi_range.get("lte", float('inf'))
                
                if displayed_emi < emi_min or displayed_emi > emi_max:
                    # Skip this product - its displayed EMI is outside the filter range
                    logger.debug(f"Skipping product due to EMI out of range: {displayed_emi} not in [{emi_min}, {emi_max}]")
                    continue
            
            # Deduplicate by model
            model_id = source.get("modelid", source.get("model_id"))
            if model_id in seen_models:
                continue
            seen_models.add(model_id)
            
            # Collect EMI for slider
            if city_offer:
                emi_values.append(city_offer.get("lowest_emi", 0))
            
            # Build product dict
            product_dict = {
                "model_id": model_id,
                "model_launch_date": source.get("model_launch_date", "1970-01-01"),
                "mkp_active_flag": source.get("mkp_active_flag", 0),
                "avg_rating": source.get("avg_rating", 0),
                "rating_count": source.get("rating_count", 0),
                "asset_category_id": source.get("asset_category_id", 0),
                "asset_category_name": source.get("asset_category_name", "UNKNOWN"),
                "manufacturer_id": source.get("manufacturer_id", 999),
                "manufacturer_desc": source.get("manufacturer_desc", "UNKNOWN"),
                "category_type": source.get("category_type", "UNKNOWN"),
                "actual_category": ResponseFormatter.CATEGORY_NORMALIZE.get(source.get("actual_category", ""), source.get("actual_category", "")),
                "mop": source.get("mop", 0),
                "property": [],
                "products": []
            }
            
            # Add city-specific property
            if city_offer:
                # Helper to convert 0/1 to boolean
                def to_bool(val):
                    return bool(val) if isinstance(val, (int, float)) else bool(val)
                
                property_dict = {
                    "cityid": [city_offer.get("cityid")],
                    "transaction_count": city_offer.get("transaction_count", 0),
                    "lowest_emi": city_offer.get("lowest_emi", 0),
                    "mop": city_offer.get("offer_price", 0),
                    "offer_price": city_offer.get("offer_price", 0),
                    "score": city_offer.get("score", 0),
                    "ty_page_count": city_offer.get("ty_page_count", 0),
                    "one_emi_off": city_offer.get("one_emi_off", 0),
                    "pdp_view_count": city_offer.get("pdp_view_count", 0),
                    "off_percentage": city_offer.get("off_percentage", 0),
                    "zero_dp_flag": city_offer.get("zero_dp_flag", 0),
                    "new_launch_flag": city_offer.get("new_launch_flag", 0),
                    "most_viewed_flag": city_offer.get("most_viewed_flag", 0),
                    "top_seller_flag": city_offer.get("top_seller_flag", 0),
                    "highest_tenure": city_offer.get("highest_tenure", 0),
                    # Only these 4 flags as boolean (controls UI visibility)
                    # model_city_flag → "Sold at stores near you"
                    # phone_setup → "Phone set up"
                    # exchange_flag → "Phone exchange"
                    # installation_flag → installation service
                    "model_city_flag": to_bool(city_offer.get("model_city_flag", 0)),
                    "phone_setup": to_bool(city_offer.get("phone_setup", 0)),
                    "exchange_flag": to_bool(city_offer.get("exchange_flag", 0)),
                    "installation_flag": to_bool(city_offer.get("installation_flag", 0)),
                    # New fields from search feed
                    "ex_showroom": city_offer.get("ex_showroom", 0),
                    "on_road_price": city_offer.get("on_road_price", 0),
                    "additional_chg": city_offer.get("additional_chg", 0),
                    "auto_lease": city_offer.get("auto_lease", 0),
                    "corporate_rto": city_offer.get("corporate_rto", 0),
                    "individual_rto": city_offer.get("individual_rto", 0),
                    "registration_chg": city_offer.get("registration_chg", 0),
                }
                product_dict["property"].append(property_dict)
            
            # Add SKU products
            products_list = source.get("products", [])
            if products_list:
                for sku in products_list:
                    sku_dict = {
                        "name": sku.get("name", source.get("product_name", "")),
                        "sku": sku.get("sku", ""),
                        "image": sku.get("image", source.get("image", "")),
                        "product_url": sku.get("product_url", ""),
                        "keyword": sku.get("keyword", ""),
                        "attribute_set_id": sku.get("attribute_set_id", ""),
                        "attribute_swatch_color": sku.get("attribute_swatch_color", {})
                    }
                    product_dict["products"].append(sku_dict)
            else:
                # Single SKU product
                product_dict["products"].append({
                    "name": source.get("product_name", ""),
                    "sku": source.get("sku", ""),
                    "image": source.get("image", ""),
                    "product_url": source.get("product_url", ""),
                    "keyword": source.get("keyword", ""),
                    "attribute_set_id": source.get("attribute_set_id", ""),
                    "attribute_swatch_color": source.get("attribute_swatch_color", {})
                })
            
            # Add all attributes
            for key, value in source.items():
                if key.startswith("attribute_") and key not in product_dict:
                    product_dict[key] = value
            
            final_response["data"]["PostV1Productlist"]["data"]["products"].append(product_dict)
        
        # Parse filter aggregations
        category_for_filters = query_info.get("category") if query_info else detected_category
        logger.info(f"Aggregations keys: {list(aggs.keys()) if aggs else 'None'}")
        logger.info(f"Category for filters: {category_for_filters}")
        filters = cls.parse_filter_aggregations(aggs, category_for_filters)
        
        if filters:
            filter_obj = {"attributes": filters}
            
            # =================== EMI SLIDER (GLOBAL RANGE) ===================
            # Use global EMI slider range calculated BEFORE applying EMI filter
            # This ensures slider shows the full available range (reference behavior)
            emi_slider_range = search_result.get("emi_slider_range", {})
            slider_min = emi_slider_range.get("min", 0)
            slider_max = emi_slider_range.get("max", 0)
            
            # If global range not available, fall back to aggregation or collected values
            if slider_min == 0 and slider_max == 0:
                # Try aggregations first
                if "city_offers" in aggs:
                    nested_agg = aggs["city_offers"]
                    if "filtered_offers" in nested_agg:
                        slider_min = int(nested_agg["filtered_offers"].get("min_emi", {}).get("value") or 0)
                        slider_max = int(nested_agg["filtered_offers"].get("max_emi", {}).get("value") or 0)
                
                # Final fallback to collected EMI values from results
                if slider_min == 0 and slider_max == 0 and emi_values:
                    slider_min = min(emi_values)
                    slider_max = max(emi_values)
            
            # Set EMI filter in response
            if slider_min > 0 or slider_max > 0:
                # =================== EMI RESPONSE STRUCTURE ===================
                # Match old script structure:
                # - emi.min/max: User's selected range (or overall if no filter)
                # - emi_slider_range (top-level): Global range for slider bounds
                
                # If user provided EMI filter, use their selection, else use overall range
                if emi_range:
                    user_emi_min = emi_range.get("gte")
                    user_emi_max = emi_range.get("lte")
                    emi_obj = {
                        "min": int(user_emi_min) if user_emi_min is not None else slider_min,
                        "max": int(user_emi_max) if user_emi_max is not None else slider_max
                    }
                    logger.info(f"User EMI filter applied: min={user_emi_min}, max={user_emi_max}")
                else:
                    emi_obj = {
                        "min": slider_min,
                        "max": slider_max
                    }
                
                filter_obj["emi"] = emi_obj
                logger.info(f"EMI response: min={emi_obj['min']}, max={emi_obj['max']}")
            
            final_response["data"]["PostV1Productlist"]["data"]["filters"].append(filter_obj)
        
        # Add emi_slider_range as separate top-level field (like old script)
        # This provides the global EMI range for slider bounds
        emi_slider_range = search_result.get("emi_slider_range", {})
        if emi_slider_range.get("min", 0) > 0 or emi_slider_range.get("max", 0) > 0:
            final_response["data"]["PostV1Productlist"]["data"]["emi_slider_range"] = {
                "min": emi_slider_range.get("min", 0),
                "max": emi_slider_range.get("max", 0)
            }
            logger.info(f"EMI slider range (global): min={emi_slider_range.get('min')}, max={emi_slider_range.get('max')}")
        
        return final_response


# =================== INITIALIZE COMPONENTS ===================
query_processor = QueryProcessor()
search_engine = SearchEngine(es, PRODUCT_INDEX_NAME)


# =================== DEALER SEARCH HELPER FUNCTIONS ===================
# =================== DEALER SEARCH HELPER FUNCTIONS ===================
# These functions delegate to dealer_handler.py for actual implementation
def _execute_dealer_search(query, city_id, customer_lat, customer_long, categories, page, page_size):
    """
    Execute dealer-only search and return formatted response.
    Delegates to dealer_handler.execute_dealer_search()
    """
    if not is_dealer_search_enabled():
        return jsonify({"error": "Dealer search not enabled"}), 400
    
    response, status_code = execute_dealer_search(
        query=query,
        city_id=city_id,
        customer_lat=customer_lat,
        customer_long=customer_long,
        categories=categories,
        page=page,
        page_size=page_size
    )
    return jsonify(response) if status_code == 200 else (jsonify(response), status_code)


def _get_dealer_results_for_hybrid(query, city_id, customer_lat, customer_long, limit=5):
    """
    Get dealer results for hybrid search (to include with product results).
    Delegates to dealer_handler.get_dealer_results_for_hybrid()
    """
    if not is_dealer_search_enabled():
        return None
    
    return get_dealer_results_for_hybrid(
        query=query,
        city_id=city_id,
        customer_lat=customer_lat,
        customer_long=customer_long,
        limit=limit
    )


# =================== RESPONSE CLEANUP FOR HYBRID SEARCH ===================
# Essential fields to keep when cleaning products for hybrid dealer+product search

# Attribute value fields to keep (category-specific essential attributes)
_ESSENTIAL_ATTRIBUTE_VALUES = {
    "attribute_brand_new_value",
    "attribute_color_value",
    "attribute_capacity_litres_value",
    "attribute_screen_size_in_inches_value",
    "attribute_hd_value",
    "attribute_smart_tv_value",
    "attribute_capacity_tons_ac_value",
    "attribute_energy_rating_value",
    "attribute_capacity_wm_value",
    "attribute_function_type_wm_value",
    "attribute_defrosting_type_new_value",
    "attribute_door_type_value",
}

# Filter attributes to keep in hybrid search (reduce filter bloat)
_ESSENTIAL_FILTER_ATTRIBUTES = {
    "attribute_brand_new",
    "attribute_color",
    "attribute_screen_size_in_inches",
    "attribute_capacity_litres",
    "attribute_capacity_tons_ac",
    "attribute_energy_rating",
    "attribute_hd",
    "attribute_smart_tv"
}


def _clean_product_for_hybrid_search(product: dict) -> dict:
    """
    Clean product data for hybrid dealer+product search response.
    Removes unnecessary fields to reduce payload size.
    
    KEEPS:
    - model_id, actual_category, asset_category_name, manufacturer_desc
    - Essential attribute_*_value fields (brand, color, capacity, etc.)
    - property[] with only: cityid, lowest_emi, offer_price, off_percentage, 
      zero_dp_flag, highest_tenure, model_city_flag
    - products[] with only: name, image, product_url, attribute_swatch_color
    
    REMOVES:
    - model_launch_date, mkp_active_flag, avg_rating, rating_count
    - manufacturer_id, category_type, mop
    - All attribute_* ID fields (not _value)
    - property: score, ty_page_count, pdp_view_count, transaction_count, etc.
    - products: attribute_set_id, keyword, sku
    """
    if not product:
        return product
    
    cleaned = {}
    
    # Keep essential top-level fields
    essential_fields = {
        "model_id", "actual_category", "asset_category_id", "asset_category_name",
        "manufacturer_desc"
    }
    
    for field in essential_fields:
        if field in product:
            cleaned[field] = product[field]
    
    # Keep only essential attribute_*_value fields
    for key, value in product.items():
        if key.endswith("_value") and key in _ESSENTIAL_ATTRIBUTE_VALUES:
            cleaned[key] = value
    
    # Clean property array - keep only essential fields
    if "property" in product and product["property"]:
        cleaned_properties = []
        essential_prop_fields = {
            "cityid", "lowest_emi", "offer_price", "off_percentage",
            "zero_dp_flag", "highest_tenure", "model_city_flag"
        }
        for prop in product["property"]:
            cleaned_prop = {field: prop[field] for field in essential_prop_fields if field in prop}
            cleaned_properties.append(cleaned_prop)
        cleaned["property"] = cleaned_properties
    
    # Clean products (SKU) array - keep only essential fields
    if "products" in product and product["products"]:
        cleaned_skus = []
        essential_sku_fields = {"name", "image", "product_url", "attribute_swatch_color"}
        for sku in product["products"]:
            cleaned_sku = {field: sku[field] for field in essential_sku_fields if field in sku}
            cleaned_skus.append(cleaned_sku)
        cleaned["products"] = cleaned_skus
    
    return cleaned


def _clean_filters_for_hybrid_search(filters: list) -> list:
    """
    Clean filters for hybrid search response.
    Keeps only essential filter attributes to reduce payload.
    """
    if not filters:
        return []
    
    cleaned_filters = []
    for filter_item in filters:
        if "attributes" in filter_item:
            cleaned_attrs = {}
            for attr_key, attr_values in filter_item["attributes"].items():
                # Only keep essential attributes
                if attr_key in _ESSENTIAL_FILTER_ATTRIBUTES:
                    # Limit to top 10 values per attribute
                    cleaned_attrs[attr_key] = attr_values[:10] if len(attr_values) > 10 else attr_values
            
            if cleaned_attrs:
                cleaned_filter = {"attributes": cleaned_attrs}
                # Copy emi if present
                if "emi" in filter_item:
                    cleaned_filter["emi"] = filter_item["emi"]
                cleaned_filters.append(cleaned_filter)
        else:
            # Keep non-attribute filters as-is
            cleaned_filters.append(filter_item)
    
    return cleaned_filters


# =================== API ENDPOINTS ===================
@app.route('/api/mall_search', methods=['POST', 'GET'])
def mall_search_api():
    """Main search endpoint"""
    start_time = time.time()
    
    try:
        # Parse request
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args.to_dict()
        
        # Extract parameters - support multiple param names for compatibility
        query = data.get('searchText', data.get('searchterm', data.get('query', data.get('q', data.get('searchQuery', '')))))
        
        # =================== OUT-OF-SCOPE QUERY GUARDRAIL ===================
        # Check if query is outside business scope BEFORE any processing
        # This saves compute and returns graceful "no results" for irrelevant queries
        if query:
            is_blocked, blocked_category = is_out_of_scope_query(query)
            if is_blocked:
                return jsonify(get_out_of_scope_response(query, blocked_category))
        
        # =================== USED CAR SEARCH REDIRECT ===================
        # Check if query is for used/2nd hand cars - return as SKU cards
        # This is checked BEFORE main search to provide instant response
        # Feature flag controlled: set USED_CAR_SEARCH_ENABLED = False to disable
        if USED_CAR_SEARCH_ENABLED and query:
            used_car_result = check_used_car_query(query)
            if used_car_result.get("is_used_car") and used_car_result.get("should_redirect"):
                # Extract filters for used car brand filtering
                filters = {}
                if 'filters' in data and isinstance(data['filters'], dict):
                    filters.update(data['filters'])
                for key, value in data.items():
                    if key.startswith('filter_') or key.startswith('attribute_'):
                        filters[key.replace('filter_', '')] = value
                
                # Build and return used car response using handler
                response = build_used_car_response(query, used_car_result, filters)
                return jsonify(response)
        
        # Handle city_id - support multiple formats:
        # 1. "citi_id_504" (full format)
        # 2. "504" (just number)  
        # 3. "21" (just number from cityId field)
        # GLOBAL SEARCH - User requested permanent global catalog search
        # Store original city_id for response but use citi_id_0 for search
        raw_city_id = data.get('city_id', data.get('cityId', data.get('cityid', '0')))
        original_city_id = raw_city_id  # Keep for any city-specific response formatting
        # FORCE GLOBAL SEARCH: Always search citi_id_0 for complete catalog
        city_id = 'citi_id_0'
        logger.info(f"Global search enabled - searching citi_id_0 (requested: {raw_city_id})")
        
        # =================== PAGINATION ===================
        # Support both formats:
        # 1. page-based: page=1, page=2 (1-indexed)
        # 2. fromIndex-based: fromIndex is a 1-indexed PAGE NUMBER
        #    fromIndex: 1 → first page (items 0 to size-1)
        #    fromIndex: 2 → second page (items size to 2*size-1)
        #    Example with size=26:
        #      fromIndex=1 → items 0-25 (first 26 SKUs)
        #      fromIndex=2 → items 26-51 (next 26 SKUs)
        page_size = int(data.get('size', data.get('pagesize', data.get('pageSize', 26))))
        
        from_index = data.get('fromIndex', data.get('from_index', data.get('from', None)))
        from_offset = None  # Direct offset for ES query
        page = 1  # Default page
        
        if from_index is not None:
            # fromIndex is 1-indexed page number
            # fromIndex=1 means page 1, fromIndex=2 means page 2, etc.
            from_index = int(from_index)
            # Treat fromIndex as page number (minimum 1)
            page = max(1, from_index)
            # Calculate actual offset: (page - 1) * size
            from_offset = (page - 1) * page_size
            logger.info(f"fromIndex={from_index} interpreted as page {page}, offset={from_offset} (size={page_size})")
        else:
            page = int(data.get('page', data.get('p', 1)))
            from_offset = None  # Will be calculated from page in build_query
        
        # =================== SORTING ===================
        # Support multiple formats:
        # 1. Object format: {"sort": {"by": "price", "order": "asc"}} or {"sortBy": {"by": "price", "order": "asc"}}
        # 2. String format: sortBy="emi_low_high" or sort_by="price_high_low"
        sort_by = None
        # Check both 'sort' and 'sortBy' for object format
        sort_data = data.get('sort') or data.get('sortBy') or data.get('sort_by')
        
        if isinstance(sort_data, dict):
            # Object format: {"by": "score/price", "order": "desc/asc"}
            # Handle empty dict {} - treat as no sorting (default relevance)
            if not sort_data:  # Empty dict {}
                sort_by = None
                logger.info("Empty sort object received, using default sorting")
            else:
                sort_field = sort_data.get('by', 'score')
                sort_order = sort_data.get('order', 'desc')
                
                if sort_field == 'score':
                    sort_by = 'relevance'
                elif sort_field == 'price' or sort_field == 'emi':
                    # Convert to internal format that matches build_query expectations
                    # build_query expects: low_to_high, high_to_low
                    sort_by = 'low_to_high' if sort_order == 'asc' else 'high_to_low'
                else:
                    sort_by = sort_field  # Pass through as-is
                logger.info(f"Parsed sort object: by={sort_field}, order={sort_order} -> sort_by={sort_by}")
        elif isinstance(sort_data, str):
            sort_by = sort_data
        
        logger.info(f"Sort by: {sort_by}")
        
        # Extract filters - support both formats:
        # 1. Old format: {"filters": {"attribute_brand_new": "5003"}}
        # 2. New format: {"attribute_brand_new": "5003"} or query param ?attribute_brand_new=5003
        filters = {}
        
        # First check for "filters" object in POST body (old format)
        if 'filters' in data and isinstance(data['filters'], dict):
            filters.update(data['filters'])
            logger.info(f"Extracted filters from 'filters' object: {filters}")
        
        # Then check for individual filter params (new format)
        for key, value in data.items():
            if key.startswith('filter_') or key.startswith('attribute_'):
                filters[key.replace('filter_', '')] = value
        
        # Extract EMI range - support multiple formats (aligned with reference):
        # 1. Separate params: emi_min=1029, emi_max=1929
        # 2. Range string in filters: filters.emi="1029-1929" or "1029,1929"
        # 3. Dict format: filters.emi={"min": 1029, "max": 1929} or {"gte": 1029, "lte": 1929}
        emi_range = None
        if 'emi_min' in data or 'emi_max' in data:
            emi_range = {}
            if 'emi_min' in data:
                emi_range['gte'] = float(data['emi_min'])
            if 'emi_max' in data:
                emi_range['lte'] = float(data['emi_max'])
        elif 'emi' in filters:
            # Parse EMI filter - support multiple formats like reference
            emi_filter = filters.pop('emi')  # Remove from filters, handle separately
            emi_range = {}
            try:
                if isinstance(emi_filter, str):
                    # re module is already imported at the top of the file
                    # Normalize: remove spaces, replace commas with dashes
                    s = emi_filter.replace(' ', '').replace(',', '-')
                    # Match any two numbers separated by non-digit characters
                    match = re.match(r"(\d+)[^\d]+(\d+)", s)
                    if match:
                        emi_min, emi_max = int(match.group(1)), int(match.group(2))
                        # Swap if min > max
                        if emi_min > emi_max:
                            emi_min, emi_max = emi_max, emi_min
                        emi_range['gte'] = emi_min
                        emi_range['lte'] = emi_max
                        logger.info(f"Parsed EMI range from string: {emi_filter} -> {emi_range}")
                    else:
                        # Try single value
                        try:
                            val = int(s)
                            emi_range['gte'] = val
                            emi_range['lte'] = val
                        except Exception:
                            pass
                elif isinstance(emi_filter, dict):
                    # Handle dict format with min/max or gte/lte keys
                    emi_min = emi_filter.get("min") or emi_filter.get("gte")
                    emi_max = emi_filter.get("max") or emi_filter.get("lte")
                    # Swap if min > max
                    if emi_min is not None and emi_max is not None and emi_min > emi_max:
                        emi_min, emi_max = emi_max, emi_min
                    if emi_min is not None and emi_min >= 0:
                        emi_range['gte'] = emi_min
                    if emi_max is not None and emi_max >= 0:
                        emi_range['lte'] = emi_max
                    logger.info(f"Parsed EMI range from dict: {emi_filter} -> {emi_range}")
            except Exception as e:
                logger.warning(f"Failed to parse EMI filter '{emi_filter}': {e}")
                emi_range = None
        
        # =================== COMPREHENSIVE PRICE/EMI PARSING ===================
        # NEW: Parse comprehensive price/EMI patterns before other parsing
        from price_emi_parser import parse_price_emi_comprehensive
        
        query_after_price_emi, price_emi_filters = parse_price_emi_comprehensive(query)
        
        # Apply extracted price/EMI filters
        if price_emi_filters:
            if 'lowest_emi' in price_emi_filters:
                # EMI filter from query overrides user-selected emi_range
                if not emi_range:  # Only if user hasn't manually selected EMI range
                    emi_range = price_emi_filters['lowest_emi']
                    logger.info(f"EMI range from query: {emi_range}")
            
            if 'mop' in price_emi_filters:
                # Price filter from query
                filters['price_range'] = price_emi_filters['mop']
                logger.info(f"Price range from query: {filters['price_range']}")
        
        # Use cleaned query for further processing
        if query_after_price_emi and query_after_price_emi != query:
            query = query_after_price_emi
            logger.info(f"Query after price/EMI extraction: '{query}'")
        
        # =================== ENHANCED QUERY PARSING ===================
        # NEW: Parse complex natural language queries
        # Example: "washing machine full loaded 8kg near pune"
        enhanced_parser = get_enhanced_parser()
        complex_query_info = enhanced_parser.parse_complex_query(query)
        
        # Override city_id if detected from query (higher priority)
        if complex_query_info.get("city_id") and not raw_city_id:
            city_id = complex_query_info["city_id"]
            logger.info(f"City detected from query: {city_id}")
        
        # Override/merge attributes from query parsing
        parsed_attrs = complex_query_info.get("attributes", {})
        if parsed_attrs:
            logger.info(f"Attributes parsed from query: {parsed_attrs}")
            # These will be merged into query_info later
        
        # Use cleaned query for category/brand detection
        search_query = complex_query_info.get("cleaned_query", query)
        
        # Process query
        query_info = query_processor.process(search_query)
        
        # =================== TV/TVS DISAMBIGUATION (POST-PROCESSING) ===================
        # The enhanced_parser may strip TV context words like "inch" from the query,
        # leaving just "tvs" which gets detected as TVS two-wheeler brand.
        # Check the ORIGINAL query for TV context and override category/processed text.
        original_lower = query.lower()
        tv_context_indicators = {"inch", "inches", "led", "lcd", "oled", "qled", "smart", "4k", "8k", "uhd", "hdr",
                                  "sony", "lg", "samsung", "philips", "toshiba", "tcl", "vu", "hisense", "panasonic"}
        has_tv_context = any(ind in original_lower.split() for ind in tv_context_indicators)
        has_tvs_word = bool(re.search(r'\btvs\b', original_lower))
        if has_tv_context and has_tvs_word and query_info.get("category") != "television":
            query_info["category"] = "television"
            # Replace "tvs" with "television" in processed text to avoid matching TVS brand
            if re.search(r'\btvs\b', query_info.get("processed", ""), re.IGNORECASE):
                query_info["processed"] = re.sub(r'\btvs\b', 'television', query_info["processed"], flags=re.IGNORECASE)
            logger.info(f"TV/TVS disambiguation: overrode category to 'television' based on original query context")
        
        # =================== FLAG DETECTION FROM ORIGINAL QUERY ===================
        # IMPORTANT: Detect flag keywords from the RAW user query, not the cleaned search_query
        # This is because enhanced_parser might clean out flag keywords before we process them
        raw_query_lower = query.lower()
        
        # One EMI Off detection
        one_emi_off_keywords = [
            "one emi off", "1 emi off", "emi off", "emi free",
            "one month emi off", "1 month emi off", "first emi off",
            "no first emi", "skip first emi"
        ]
        if any(kw in raw_query_lower for kw in one_emi_off_keywords):
            query_info["detect_one_emi_off"] = True
            logger.info(f"[FLAG] Detected 'one emi off' in original query: '{query}'")
        
        # Zero DP detection
        zero_dp_keywords = [
            "zero down payment", "zero downpayment", "zero dp", "0 down payment",
            "0dp", "0 dp", "no down payment", "no downpayment", "no dp", "zero down",
            "0 down", "nodp", "zerodp", "without down payment", "without dp"
        ]
        if any(kw in raw_query_lower for kw in zero_dp_keywords):
            query_info["detect_zero_dp"] = True
            logger.info(f"[FLAG] Detected 'zero dp' in original query: '{query}'")
        
        # New launch detection
        new_launch_keywords = [
            "new launch", "newlaunch", "newly launched", "latest launch",
            "latest launched", "new arrival", "new arrivals", "just launched",
            "recently launched", "brand new", "latest model", "latest models",
            "newest", "new model", "new models", "latest"
        ]
        if any(kw in raw_query_lower for kw in new_launch_keywords):
            query_info["detect_new_launch"] = True
            logger.info(f"[FLAG] Detected 'new launch' in original query: '{query}'")
        
        # Best selling detection
        best_selling_keywords = [
            "best selling", "bestselling", "best seller", "bestseller",
            "top selling", "topselling", "top seller", "topseller",
            "most sold", "most popular", "popular products", "trending",
            "hot selling", "fast selling", "highest selling"
        ]
        if any(kw in raw_query_lower for kw in best_selling_keywords):
            query_info["detect_best_selling"] = True
            logger.info(f"[FLAG] Detected 'best selling' in original query: '{query}'")
        
        # =================== PRESERVE USER'S ORIGINAL QUERY ===================
        # IMPORTANT: Store the user's ACTUAL input (before any cleaning) for corrected_query
        # This allows generate_corrected_query to show proper typo corrections
        query_info["user_original_query"] = data.get('searchText', data.get('searchterm', data.get('query', data.get('q', data.get('searchQuery', '')))))
        
        # =================== STORE PARSED FILTERS FOR RESPONSE ===================
        # Add the parsed price/EMI filters to query_info for showing in response
        query_info["parsed_filters"] = price_emi_filters
        
        # Merge parsed attributes into query_info
        # IMPORTANT: Normalize attribute names from enhanced parser to match build_query expectations
        # Enhanced parser uses: ram_gb, storage_gb
        # build_query expects: ram, storage
        # NOTE: Do NOT overwrite attributes that query_processor already extracted correctly,
        # because query_processor has proper capitalization from pattern matching (e.g., "Double Door")
        # while enhanced_parser may have lowercase values (e.g., "double door")
        if parsed_attrs:
            if "attributes" not in query_info:
                query_info["attributes"] = {}
            # Normalize attribute names
            normalized_attrs = {}
            for key, value in parsed_attrs.items():
                if key == "ram_gb":
                    normalized_attrs["ram"] = value
                elif key == "storage_gb":
                    # Don't overwrite storage if storage_tb is already set (TB takes precedence)
                    if "storage_tb" not in query_info["attributes"]:
                        normalized_attrs["storage"] = value
                else:
                    normalized_attrs[key] = value
            # Only add NEW attributes from enhanced parser, don't overwrite existing ones
            # query_processor has correct capitalization that matches ES data
            for attr_key, attr_value in normalized_attrs.items():
                if attr_key not in query_info["attributes"]:
                    query_info["attributes"][attr_key] = attr_value
                    logger.info(f"Added attribute from enhanced parser: {attr_key}={attr_value}")
                else:
                    logger.info(f"Keeping existing attribute from query_processor: {attr_key}={query_info['attributes'][attr_key]} (enhanced parser had: {attr_value})")
            logger.info(f"Final merged attributes: {query_info['attributes']}")
        
        # Add enhanced parsing metadata
        query_info["enhanced_parsing"] = {
            "city_detected": complex_query_info.get("city_id") is not None,
            "price_emi_detected": bool(price_emi_filters),
            "attributes_detected": len(parsed_attrs) > 0,
            "comparison_field": "lowest_emi" if price_emi_filters and "lowest_emi" in price_emi_filters else "mop"
        }
        
        logger.info(f"Search request: query='{query}', city_id='{city_id}', page={page}, size={page_size}, sort={sort_by}")
        # PERF: Skip JSON dump for pagination requests (verbose)
        if page == 1:
            logger.debug(f"Query analysis: {json.dumps(query_info)}")
        
        # =================== DEALER SEARCH INTENT CLASSIFICATION ===================
        # Check if dealer search is enabled and classify intent
        # This allows routing to dealer search when user searches for stores/dealers
        # OPTIMIZATION: Skip for page > 1 (pagination) - intent doesn't change
        dealer_search_type = data.get('search_type', None)  # Allow forcing search type
        customer_lat = data.get('Customer_lat', data.get('lat', data.get('customer_lat')))
        customer_long = data.get('Customer_long', data.get('lon', data.get('lng', data.get('customer_long'))))
        
        # =================== CONFLICT BRAND PRE-CHECK ===================
        # Dealer names that are ALSO product brands on BajajMall
        # These should ALWAYS show both products AND dealers, regardless of intent
        _CONFLICT_BRAND_NAMES_PRECHECK = {
            "croma", "chroma", "kroma", "croma store", "chroma store",
            "reliance digital", "reliance",
        }
        
        # Check if query matches a conflict brand (before intent classification)
        _query_lower = query.lower().strip()
        _is_conflict_brand_query = any(
            _query_lower == cb or _query_lower.startswith(cb + " ") or _query_lower.endswith(" " + cb)
            for cb in _CONFLICT_BRAND_NAMES_PRECHECK
        )
        
        # Check if we have location info (lat/lng OR valid city_id)
        _has_location_info = (customer_lat or customer_long) or (city_id and city_id not in ['citi_id_0', '0', 'citi_id_'])
        
        # If conflict brand detected and location available, do hybrid search
        if page == 1 and _is_conflict_brand_query and is_dealer_search_enabled() and _has_location_info:
            try:
                logger.info(f"Conflict brand query detected: '{query}' - executing hybrid search (products + dealers)")
                clean_city_id = city_id.replace('citi_id_', '') if city_id else None
                if clean_city_id == '0' or clean_city_id == 0:
                    clean_city_id = None
                
                # Execute dealer search using DealerSearchEngine from dealer_search.py
                from Dealersearch.dealer_search import DealerSearchEngine as ConflictDealerEngine
                from Dealersearch.dealer_search_api_v2 import DealerResponseFormatter
                _cds_engine = ConflictDealerEngine(es)
                _cds_formatter = DealerResponseFormatter()
                
                # Build location dict for dealer search
                _cds_location = None
                if customer_lat and customer_long:
                    try:
                        _cds_location = {"lat": float(customer_lat), "lon": float(customer_long)}
                    except (ValueError, TypeError):
                        pass
                
                _cds_results = _cds_engine.search(
                    query=query,
                    city_id=clean_city_id,
                    location=_cds_location,
                    radius_km=30,
                    only_active=True
                )
                
                _cds_dealers = []
                if _cds_results and _cds_results.get('dealers'):
                    # Limit to 5 dealers
                    for idx, _dealer in enumerate(_cds_results.get('dealers', [])[:5]):
                        _formatted_dealer = _cds_formatter.format_dealer(
                            _dealer,
                            is_nearest=(idx == 0)  # First dealer is nearest
                        )
                        if _formatted_dealer:
                            _cds_dealers.append(clean_dealer_response(_formatted_dealer))
                
                logger.info(f"Conflict brand hybrid: found {len(_cds_dealers)} dealers for '{query}'")
                
                # Store in query_info for response building
                if _cds_dealers:
                    query_info['_conflict_brand_detected'] = True
                    query_info['_conflict_brand_dealers'] = _cds_dealers
                    query_info['_conflict_brand_total_dealers'] = _cds_results.get('total', len(_cds_dealers))
                    
            except Exception as e:
                logger.error(f"Conflict brand dealer search error: {e}")
        
        if page == 1 and is_dealer_search_enabled() and dealer_search_type != 'product':
            try:
                # Use handler for intent classification
                intent_result = classify_dealer_intent(
                    query=query,
                    city_id=city_id.replace('citi_id_', '') if city_id else None,
                    customer_lat=float(customer_lat) if customer_lat else None,
                    customer_long=float(customer_long) if customer_long else None
                )
                
                if intent_result:
                    SearchIntent = get_search_intent_class()
                    logger.info(f"Dealer search intent: {intent_result.intent.name}, confidence: {intent_result.confidence:.2f}")
                    
                    # Route based on intent or forced search_type
                    if dealer_search_type == 'dealer':
                        # Forced dealer search - route directly
                        clean_city_id = city_id.replace('citi_id_', '') if city_id else None
                        if clean_city_id == '0' or clean_city_id == 0:
                            clean_city_id = None
                        return _execute_dealer_search(
                            query=query,
                            city_id=clean_city_id,
                            customer_lat=customer_lat,
                            customer_long=customer_long,
                            categories=filters.get('category', []),
                            page=page,
                            page_size=page_size
                        )
                    
                    elif SearchIntent and intent_result.intent == SearchIntent.DEALER and intent_result.confidence > 0.6:
                        # =================== DEALER INTENT ROUTING ===================
                        # STRICT RULE: When dealer intent is detected, ALWAYS show dealer
                        # response (PostV1Productlistsearch > PostV1Dealerlist).
                        # NEVER show product list for dealer intent.
                        #
                        # ONE EXCEPTION — CONFLICT BRANDS:
                        # A small whitelist of dealer names that are ALSO product brands
                        # sold on BajajMall. Only these get both products + dealers.
                        # e.g., "croma" is both a dealer chain AND a product brand.
                        #
                        # Examples:
                        #   "samsung store near me"  → DEALER_ONLY
                        #   "mobile world"           → DEALER_ONLY (dealer name, not a product brand)
                        #   "dealer near me"         → DEALER_ONLY
                        #   "croma"                  → CONFLICT (also a product brand on BajajMall)

                        # Dealer names that are ALSO product brands on BajajMall
                        # Include common typos/variations
                        _CONFLICT_BRAND_NAMES = {
                            "croma", "chroma", "kroma", "croma store", "chroma store",
                            "reliance digital", "reliance",
                        }

                        _signals = intent_result.signals or []

                        # Check if query matches a known conflict brand
                        _matched_conflict_brand = None
                        for _cb in _CONFLICT_BRAND_NAMES:
                            if _cb in query.lower():
                                _matched_conflict_brand = _cb
                                break

                        if _matched_conflict_brand:
                            # ========== CONFLICT: dealer name is also a product brand ==========
                            try:
                                product_count_result = search_engine.search(
                                    query_info=query_info,
                                    city_id=city_id,
                                    filters=filters if filters else None,
                                    page=1,
                                    page_size=1
                                )
                                product_count = product_count_result.get("total", 0)

                                if product_count > 0:
                                    logger.info(
                                        f"CONFLICT DETECTED: '{query}' matches conflict brand "
                                        f"'{_matched_conflict_brand}' AND {product_count} products. Returning both."
                                    )
                                    clean_city_id = city_id.replace('citi_id_', '') if city_id else None
                                    if clean_city_id == '0' or clean_city_id == 0:
                                        clean_city_id = None

                                    dealer_response = _get_dealer_results_for_hybrid(
                                        query=query,
                                        city_id=clean_city_id,
                                        customer_lat=customer_lat,
                                        customer_long=customer_long,
                                        limit=page_size
                                    )
                                    if not dealer_response:
                                        logger.info(f"No dealers in city_id={clean_city_id}, broadening to all cities")
                                        dealer_response = _get_dealer_results_for_hybrid(
                                            query=query,
                                            city_id=None,
                                            customer_lat=None,
                                            customer_long=None,
                                            limit=page_size
                                        )
                                    query_info['_conflict_detected'] = True
                                    query_info['_conflict_dealer_response'] = dealer_response
                                    query_info['_include_dealer_results'] = True
                                    query_info['_dealer_intent'] = intent_result
                                    # Continue to product search below...
                                else:
                                    # Brand exists but no products matched → pure dealer
                                    clean_city_id = city_id.replace('citi_id_', '') if city_id else None
                                    if clean_city_id == '0' or clean_city_id == 0:
                                        clean_city_id = None
                                    return _execute_dealer_search(
                                        query=query,
                                        city_id=clean_city_id,
                                        customer_lat=customer_lat,
                                        customer_long=customer_long,
                                        categories=filters.get('category', []),
                                        page=page,
                                        page_size=page_size
                                    )
                            except Exception as conflict_err:
                                logger.error(f"Conflict detection error: {conflict_err}")
                                clean_city_id = city_id.replace('citi_id_', '') if city_id else None
                                if clean_city_id == '0' or clean_city_id == 0:
                                    clean_city_id = None
                                return _execute_dealer_search(
                                    query=query,
                                    city_id=clean_city_id,
                                    customer_lat=customer_lat,
                                    customer_long=customer_long,
                                    categories=filters.get('category', []),
                                    page=page,
                                    page_size=page_size
                                )
                        else:
                            # ========== STRICT DEALER_ONLY: all other dealer intents ==========
                            logger.info(
                                f"DEALER SEARCH (strict): '{query}' → DEALER_ONLY "
                                f"(signals: {[s for s in _signals if 'dealer' in s or 'primary' in s][:5]})"
                            )
                            clean_city_id = city_id.replace('citi_id_', '') if city_id else None
                            if clean_city_id == '0' or clean_city_id == 0:
                                clean_city_id = None
                            return _execute_dealer_search(
                                query=query,
                                city_id=clean_city_id,
                                customer_lat=customer_lat,
                                customer_long=customer_long,
                                categories=filters.get('category', []),
                                page=page,
                                page_size=page_size
                            )
                    
                    elif SearchIntent and intent_result.intent == SearchIntent.HYBRID:
                        # Hybrid search - return both product and dealer results
                        # Continue with product search below, dealer will be added to response
                        query_info['_include_dealer_results'] = True
                        query_info['_dealer_intent'] = intent_result
                    
            except Exception as e:
                logger.error(f"Dealer intent classification error: {e}")
                # Continue with product search on error
        
        # =================== GLOBAL EMI SLIDER CALCULATION ===================
        # Calculate EMI slider min/max BEFORE applying EMI filter (like reference)
        # This gives the user the full available range for the slider
        # OPTIMIZATION: Skip for pages > 1 (user already has slider range from page 1)
        # SKIP if running low on time (EMI aggregation can be slow)
        if page > 1:
            # Pagination request - no need to recalculate slider
            emi_slider_range = {"min": 0, "max": 0}
        else:
            elapsed = time.time() - start_time
            if elapsed < (MAX_REQUEST_TIME_SECONDS - 2.0):  # Only if we have > 2s remaining
                emi_slider_range = search_engine.get_global_emi_slider_range(
                    query_info=query_info,
                    city_id=city_id,
                    filters=filters if filters else None
                )
                logger.info(f"Global EMI slider range: {emi_slider_range}")
            else:
                emi_slider_range = {"min": 0, "max": 0}
                logger.warning(f"Skipping EMI slider calculation - time budget low ({elapsed:.2f}s elapsed)")
        
        # Execute search
        search_result = search_engine.search(
            query_info=query_info,
            city_id=city_id,
            filters=filters if filters else None,
            page=page,
            page_size=page_size,
            emi_range=emi_range,
            sort_by=sort_by,
            from_offset=from_offset,
            start_time=start_time  # Pass for timeout tracking
        )
        
        # Add EMI slider range to search result for response formatting
        search_result["emi_slider_range"] = emi_slider_range
        
        # Format response
        response = ResponseFormatter.format_response(
            search_result=search_result,
            city_id=city_id,
            query_info=query_info,
            emi_range=emi_range
        )
        
        # =================== UNIFIED RESPONSE FORMAT ===================
        # ALL responses use the same PostV1Productlistsearch wrapper with both
        # PostV1Productlist and PostV1Dealerlist inside data[0].
        # This ensures frontend always gets a consistent structure.
        #
        # Response types (convey intent, but structure is always the same):
        #   PRODUCT_ONLY  → PostV1Productlist populated, PostV1Dealerlist empty
        #   DEALER_ONLY   → handled above via _execute_dealer_search return
        #   CONFLICT      → both PostV1Productlist and PostV1Dealerlist populated
        
        conflict_detected = query_info.get('_conflict_detected', False)
        conflict_dealer_response = query_info.get('_conflict_dealer_response')
        
        # Also check for conflict brand pre-check results (for queries like "croma", "chroma")
        conflict_brand_detected = query_info.get('_conflict_brand_detected', False)
        conflict_brand_dealers = query_info.get('_conflict_brand_dealers', [])
        conflict_brand_total = query_info.get('_conflict_brand_total_dealers', 0)
        
        # Extract the product list section built by ResponseFormatter
        product_list_section = response["data"]["PostV1Productlist"]
        
        # =================== CLEAN PRODUCTS FOR HYBRID DEALER+PRODUCT SEARCH ===================
        # Only clean products for DEALER-INTENT conflict (from intent classifier), 
        # NOT for conflict BRAND queries (like "croma", "chroma") which are primarily product searches
        # Conflict brand queries should return FULL product data alongside dealers
        if conflict_detected and not conflict_brand_detected:
            cleaned_products = []
            for product in product_list_section.get("data", {}).get("products", []):
                cleaned_product = _clean_product_for_hybrid_search(product)
                cleaned_products.append(cleaned_product)
            product_list_section["data"]["products"] = cleaned_products
            
            # Also clean filters to show only essential ones
            if "filters" in product_list_section.get("data", {}):
                product_list_section["data"]["filters"] = _clean_filters_for_hybrid_search(
                    product_list_section["data"]["filters"]
                )
        
        # Build dealer list section based on conflict detection
        if conflict_detected and conflict_dealer_response and isinstance(conflict_dealer_response, dict):
            dealer_data = conflict_dealer_response.get("PostV1Dealerlist", {}).get("data", {})
            dealers_list = dealer_data.get("dealers", [])
            total_dealers = dealer_data.get("totaldealers", len(dealers_list))
            
            # Add total_dealers count into product data for frontend
            product_list_section["data"]["total_dealers"] = total_dealers
            
            dealer_list_section = conflict_dealer_response.get("PostV1Dealerlist", {
                "data": {"dealers": [], "totaldealers": 0, "dealer_filters": {"categories": []}, "original_query": query},
                "message": "No Search Found",
                "status": True
            })
            
            logger.info(
                f"[CONFLICT] '{query}' → {search_result.get('total', 0)} products + "
                f"{total_dealers} dealers (unified PostV1Productlistsearch)"
            )
        elif conflict_brand_detected and conflict_brand_dealers:
            # =================== CONFLICT BRAND (croma, chroma, etc.) ===================
            # These are brand names that are also dealer names - show both products and dealers
            total_dealers = conflict_brand_total
            
            # Add total_dealers count into product data for frontend
            product_list_section["data"]["total_dealers"] = total_dealers
            
            dealer_list_section = {
                "data": {
                    "dealer_filters": {"categories": []},
                    "dealers": conflict_brand_dealers,
                    "original_query": query,
                    "totaldealers": total_dealers
                },
                "message": "Success",
                "status": True
            }
            
            logger.info(
                f"[CONFLICT BRAND] '{query}' → {search_result.get('total', 0)} products + "
                f"{len(conflict_brand_dealers)} dealers (hybrid response)"
            )
        else:
            # Product-only: include empty dealer section for consistent structure
            dealer_list_section = {
                "data": {
                    "dealer_filters": {"categories": []},
                    "dealers": [],
                    "original_query": query,
                    "totaldealers": 0
                },
                "message": "No dealers found",
                "status": True
            }
        
        # Build unified response with both product and dealer sections
        # NOTE: Do NOT wrap in PostV1Productlistsearch here — the GraphQL gateway adds that layer.
        unified_response = {
            "data": {
                "PostV1Productlist": product_list_section,
                "PostV1Dealerlist": dealer_list_section
            }
        }
        
        # Log total request time for performance monitoring
        total_time = time.time() - start_time
        if total_time > 3.0:
            logger.warning(f"SLOW REQUEST: '{query}' took {total_time:.2f}s (limit: {MAX_REQUEST_TIME_SECONDS}s)")
        else:
            logger.info(f"Request completed in {total_time:.2f}s")
        
        return jsonify(unified_response)
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "data": {
                "PostV1Productlist": {
                    "status": False,
                    "message": f"Error: {str(e)}",
                    "data": {
                        "products": [],
                        "totalrecords": 0,
                        "filters": []
                    }
                },
                "PostV1Dealerlist": {
                    "data": {
                        "dealer_filters": {"categories": []},
                        "dealers": [],
                        "original_query": "",
                        "totaldealers": 0
                    },
                    "message": "Error",
                    "status": False
                }
            }
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        es_health = es.cluster.health()
        return jsonify({
            "status": "healthy",
            "elasticsearch": es_health["status"],
            "index": PRODUCT_INDEX_NAME,
            "indices": {
                "product": PRODUCT_INDEX_NAME,
                "category": CATEGORY_INDEX_NAME,
                "autosuggest": AUTOSUGGEST_INDEX_NAME,
                "brand": BRAND_INDEX_NAME,
                "dealer": DEALER_INDEX_NAME if is_dealer_search_enabled() else None
            },
            "attribute_id_name_map": {
                "path": ATTRIBUTE_ID_NAME_MAP_PATH,
                "loaded": ATTRIBUTE_ID_NAME_MAP_LOADED,
                "entries": len(ATTRIBUTE_ID_NAME_MAP),
                "error": ATTRIBUTE_ID_NAME_MAP_ERROR
            },
            "dealer_search_enabled": is_dealer_search_enabled(),
            "dealer_index": DEALER_INDEX_NAME if is_dealer_search_enabled() else None
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/api/analyze_query', methods=['POST', 'GET'])
def analyze_query():
    """Debug endpoint to analyze query processing"""
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args.to_dict()
        
        query = data.get('query', data.get('q', ''))
        query_info = query_processor.process(query)
        
        return jsonify({
            "status": "success",
            "analysis": query_info
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500





#================Replace autosuggest===============================


from autosuggest_v2 import init_autosuggest, get_autosuggest_engine

# Initialize with ES connection
_autosuggest_engine = None

def get_autosuggest():
    """Get or initialize autosuggest engine"""
    global _autosuggest_engine
    if _autosuggest_engine is None:
        _autosuggest_engine = init_autosuggest(
            es_client=es,
            product_index=PRODUCT_INDEX_NAME,
            autosuggest_index=AUTOSUGGEST_INDEX_NAME
        )
    return _autosuggest_engine


@app.route('/api/mall_autosuggest', methods=['POST', 'GET'])
def mall_autosuggest_api():
    """
    Autosuggest API - Returns max 7 suggestions
    
    Request:
        POST: {"query": "sam"} or {"q": "sam"}
        GET: ?query=sam or ?q=sam
    
    Response:
        {
            "message": "success",
            "response": {
                "keywords": ["Samsung", "Samsung phone", ...],  # Max 7
                "chips": ["4GB", "8GB", ...],  # Attribute chips for category
                "filter_text": "by internal storage",
                "synonym_suggestions": [],
                "language": "english"
            }
        }
    """
    try:
        start_time = time.time()
        
        # Get query from request
        if request.method == 'POST':
            data = request.get_json() or {}
            query = data.get('query', data.get('q', ''))
        else:
            data = request.args.to_dict()
            query = data.get('query', data.get('q', ''))
        
        if not query or not query.strip():
            return jsonify({
                "message": "Query parameter required",
                "response": {
                    "keywords": [],
                    "chips": [],
                    "filter_text": "",
                    "synonym_suggestions": [],
                    "language": "english"
                }
            }), 200
        
        # Extract optional location/city params for dealer autosuggest
        raw_city_id = data.get('city_id', data.get('cityId', data.get('cityid')))
        customer_lat = data.get('Customer_lat', data.get('lat', data.get('customer_lat')))
        customer_long = data.get('Customer_long', data.get('lon', data.get('lng', data.get('customer_long'))))
        
        # =================== OUT-OF-SCOPE QUERY GUARDRAIL ===================
        # Don't provide suggestions for out-of-scope queries
        is_blocked, blocked_category = is_out_of_scope_query(query)
        if is_blocked:
            logger.info(f"Autosuggest blocked for out-of-scope query: '{query}' (category: {blocked_category})")
            return jsonify({
                "message": "success",
                "response": {
                    "keywords": [],
                    "chips": [],
                    "filter_text": "",
                    "synonym_suggestions": [],
                    "language": "english"
                }
            }), 200
        
        # Get product suggestions from autosuggest engine
        engine = get_autosuggest()
        result = engine.get_suggestions(query.strip())
        
        # =================== DEALER AUTOSUGGEST INTEGRATION ===================
        # If dealer search is enabled, fetch dealer suggestions in TWO scenarios:
        #   A) Intent classifier detects dealer/hybrid intent (full words like "dealer near me")
        #   B) User is TYPING a dealer-related prefix (partial words like "deale", "stor", "show")
        #      → In this case, inject dealer keyword suggestions so the user sees them
        #        while still typing, and also fetch nearby dealer cards.
        if is_dealer_search_enabled():
            try:
                q_lower = query.strip().lower()
                q_stripped = q_lower.split()[-1] if q_lower.split() else q_lower  # last word

                # Dealer-related words that we want to detect as partial prefixes
                _DEALER_PREFIX_WORDS = [
                    "dealer", "dealers", "dealership",
                    "store", "stores", "storefront",
                    "showroom", "showrooms",
                    "shop", "shops",
                    "outlet", "outlets",
                    "retailer", "retailers",
                    "near me", "nearest",
                ]

                # Check if user is typing a prefix of any dealer keyword
                _is_dealer_prefix = any(
                    dw.startswith(q_stripped) or q_lower.endswith(q_stripped) and dw.startswith(q_stripped)
                    for dw in _DEALER_PREFIX_WORDS
                    if len(q_stripped) >= 3  # at least 3 chars to avoid false positives
                )

                # For prefix matches, expand query so intent classifier fires
                autosuggest_query = query.strip()
                if _is_dealer_prefix:
                    # Find the full dealer keyword that matches and use it
                    for dw in _DEALER_PREFIX_WORDS:
                        if dw.startswith(q_stripped):
                            # Replace partial word with full dealer keyword
                            words = query.strip().split()
                            words[-1] = dw
                            autosuggest_query = " ".join(words)
                            break

                dealer_suggest = get_dealer_autosuggest(
                    query=autosuggest_query,
                    city_id=raw_city_id,
                    customer_lat=float(customer_lat) if customer_lat else None,
                    customer_long=float(customer_long) if customer_long else None,
                    limit=3
                )
                dealer_cards = dealer_suggest.get("dealer", [])
                dealer_keywords = dealer_suggest.get("dealer_keywords", [])

                # For prefix-match scenario, also add keyword completions
                # e.g. typing "deale" → suggest "dealer near me", "dealer store"
                if _is_dealer_prefix and not dealer_keywords:
                    dealer_keywords = []
                    for dw in _DEALER_PREFIX_WORDS[:6]:
                        if dw.startswith(q_stripped) and dw != q_stripped:
                            dealer_keywords.append(f"{dw} near me")
                    if not dealer_keywords:
                        dealer_keywords = ["dealer near me", "stores near me"]

                if dealer_cards or dealer_keywords:
                    result["response"]["dealer"] = dealer_cards
                    # Append dealer keywords into the existing keywords list
                    # so they appear in the same suggestion dropdown
                    existing_keywords = result["response"].get("keywords", [])
                    for dk in dealer_keywords:
                        if dk not in existing_keywords:
                            existing_keywords.append(dk)
                    result["response"]["keywords"] = existing_keywords
                    # Hint for frontend: when user clicks a dealer suggestion,
                    # send search_type=dealer to get DEALER_ONLY response
                    result["response"]["dealer_search_type"] = "dealer"
                    logger.info(
                        f"Dealer autosuggest: {len(dealer_cards)} cards, "
                        f"{len(dealer_keywords)} keywords merged for '{query}'"
                        f"{' (prefix-match)' if _is_dealer_prefix else ''}"
                    )
            except Exception as de:
                logger.warning(f"Dealer autosuggest failed (non-fatal): {de}")
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Autosuggest API: '{query}' -> {len(result['response']['keywords'])} suggestions in {elapsed:.2f}ms")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Autosuggest error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "message": "Internal server error",
            "error": str(e),
            "response": {
                "keywords": [],
                "chips": [],
                "filter_text": "",
                "synonym_suggestions": [],
                "language": "english"
            }
        }), 500








'''
# =================== AUTOSUGGEST API ===================
# Initialize autosuggest engine at startup
from autosuggest_v2 import init_autosuggest, get_autosuggest_engine

# Initialize with ES connection
_autosuggest_engine = None

def get_autosuggest():
    """Get or initialize autosuggest engine"""
    global _autosuggest_engine
    if _autosuggest_engine is None:
        _autosuggest_engine = init_autosuggest(
            es_client=es,
            product_index=PRODUCT_INDEX_NAME,
            autosuggest_index=AUTOSUGGEST_INDEX_NAME
        )
    return _autosuggest_engine


@app.route('/api/mall_autosuggest', methods=['POST', 'GET'])
def mall_autosuggest_api():
    """
    Autosuggest API - Returns max 15 suggestions with dealer cards
    
    Request:
        POST: {"query": "sam", "cityId": 1948} or {"q": "sam"}
        GET: ?query=sam or ?q=sam
    
    Response Format:
        {
            "data": {
                "autoSuggest": {
                    "message": "no autosuggest issue",
                    "response": {
                        "chips": ["4 GB", "8 GB", ...],
                        "Card": [{"dealer": [{"name": "Chroma", "adress": "...", "dealer_id": "22"}]}],
                        "filter_text": "by internal storage",
                        "keywords": ["Samsung", "Samsung phone", ...],
                        "language": "english",
                        "__typename": "SuggestResponseData"
                    },
                    "__typename": "AutoSuggestOutput"
                }
            }
        }
    """
    try:
        start_time = time.time()
        
        # Get query from request
        if request.method == 'POST':
            data = request.get_json() or {}
            query = data.get('query', data.get('q', ''))
        else:
            data = request.args.to_dict()
            query = data.get('query', data.get('q', ''))
        
        if not query or not query.strip():
            return jsonify({
                "data": {
                    "autoSuggest": {
                        "message": "no autosuggest issue",
                        "response": {
                            "chips": [],
                            "Card": [],
                            "filter_text": "",
                            "keywords": [],
                            "language": "english",
                            "__typename": "SuggestResponseData"
                        },
                        "__typename": "AutoSuggestOutput"
                    }
                }
            }), 200
        
        # Extract optional location/city params for dealer autosuggest
        raw_city_id = data.get('city_id', data.get('cityId', data.get('cityid')))
        customer_lat = data.get('Customer_lat', data.get('lat', data.get('customer_lat')))
        customer_long = data.get('Customer_long', data.get('lon', data.get('lng', data.get('customer_long'))))
        
        # =================== OUT-OF-SCOPE QUERY GUARDRAIL ===================
        # Don't provide suggestions for out-of-scope queries
        is_blocked, blocked_category = is_out_of_scope_query(query)
        if is_blocked:
            logger.info(f"Autosuggest blocked for out-of-scope query: '{query}' (category: {blocked_category})")
            return jsonify({
                "data": {
                    "autoSuggest": {
                        "message": "no autosuggest issue",
                        "response": {
                            "chips": [],
                            "Card": [],
                            "filter_text": "",
                            "keywords": [],
                            "language": "english",
                            "__typename": "SuggestResponseData"
                        },
                        "__typename": "AutoSuggestOutput"
                    }
                }
            }), 200
        
        # Get product suggestions from autosuggest engine
        engine = get_autosuggest()
        result = engine.get_suggestions(query.strip())
        
        # =================== DEALER AUTOSUGGEST INTEGRATION ===================
        # If dealer search is enabled, fetch dealer suggestions:
        #   A) For ALL queries when location (cityId/lat/long) is provided - show nearby dealers
        #   B) For dealer-intent queries (dealer/store/showroom keywords) - show relevant dealers
        #   C) For dealer prefix typing (partial words) - suggest completions
        if is_dealer_search_enabled():
            try:
                q_lower = query.strip().lower()
                q_stripped = q_lower.split()[-1] if q_lower.split() else q_lower  # last word

                # Dealer-related words that we want to detect as partial prefixes
                _DEALER_PREFIX_WORDS = [
                    "dealer", "dealers", "dealership",
                    "store", "stores", "storefront",
                    "showroom", "showrooms",
                    "shop", "shops",
                    "outlet", "outlets",
                    "retailer", "retailers",
                    "near me", "nearest",
                ]

                # Check if user is typing a prefix of any dealer keyword
                _is_dealer_prefix = any(
                    dw.startswith(q_stripped) or q_lower.endswith(q_stripped) and dw.startswith(q_stripped)
                    for dw in _DEALER_PREFIX_WORDS
                    if len(q_stripped) >= 3  # at least 3 chars to avoid false positives
                )

                # For prefix matches, expand query so intent classifier fires
                autosuggest_query = query.strip()
                if _is_dealer_prefix:
                    # Find the full dealer keyword that matches and use it
                    for dw in _DEALER_PREFIX_WORDS:
                        if dw.startswith(q_stripped):
                            # Replace partial word with full dealer keyword
                            words = query.strip().split()
                            words[-1] = dw
                            autosuggest_query = " ".join(words)
                            break

                # First try intent-based dealer search
                dealer_suggest = get_dealer_autosuggest(
                    query=autosuggest_query,
                    city_id=raw_city_id,
                    customer_lat=float(customer_lat) if customer_lat else None,
                    customer_long=float(customer_long) if customer_long else None,
                    limit=3
                )
                dealer_cards = dealer_suggest.get("dealer", [])
                dealer_keywords = dealer_suggest.get("dealer_keywords", [])
                
                # If no dealers found but location is available, do a direct nearby search
                # This ensures we show nearby dealers for normal product queries like "mobile"
                has_location = raw_city_id or (customer_lat and customer_long)
                if not dealer_cards and has_location:
                    try:
                        from new_mall_pipeline.Dealersearch.dealer_handler import _dealer_search_engine, _autosuggest_dealer_formatter
                        if _dealer_search_engine and _autosuggest_dealer_formatter:
                            location = None
                            if customer_lat and customer_long:
                                try:
                                    location = {"lat": float(customer_lat), "lon": float(customer_long)}
                                except (ValueError, TypeError):
                                    pass
                            
                            # Clean city_id for dealer search (remove 'citi_id_' prefix if present)
                            dealer_city_id = None
                            if raw_city_id:
                                dealer_city_id = str(raw_city_id).replace('citi_id_', '')
                                if dealer_city_id in ('0', ''):
                                    dealer_city_id = None
                            
                            # Search for nearby dealers filtered by city
                            nearby_result = _dealer_search_engine.search(
                                query="",  # Empty query to get all nearby dealers
                                city_id=dealer_city_id,  # Filter by user's city
                                location=location,
                                radius_km=50,  # Larger radius for city-based search
                                sort_by="distance" if location else "relevance",
                                size=3
                            )
                            nearby_dealers = nearby_result.get("dealers", [])
                            
                            for d in nearby_dealers[:3]:
                                card = _autosuggest_dealer_formatter.format_dealer_card(d)
                                if card:
                                    dealer_cards.append(card)
                            
                            if dealer_cards:
                                logger.info(f"Nearby dealers fallback: found {len(dealer_cards)} dealers for '{query}' in city_id={dealer_city_id}")
                    except Exception as nearby_err:
                        logger.debug(f"Nearby dealers fallback failed: {nearby_err}")

                # For prefix-match scenario, also add keyword completions
                # e.g. typing "deale" → suggest "dealer near me", "dealer store"
                if _is_dealer_prefix and not dealer_keywords:
                    dealer_keywords = []
                    for dw in _DEALER_PREFIX_WORDS[:6]:
                        if dw.startswith(q_stripped) and dw != q_stripped:
                            dealer_keywords.append(f"{dw} near me")
                    if not dealer_keywords:
                        dealer_keywords = ["dealer near me", "stores near me"]

                if dealer_cards or dealer_keywords:
                    # Transform dealer cards to required Card format:
                    # Card: [{"dealer": [{"name": "Chroma", "adress": "...", "dealer_id": "22"}]}]
                    formatted_dealer_cards = []
                    for dc in dealer_cards:
                        # Get dealer_id directly or extract from redirection_Url
                        dealer_id = dc.get("dealer_id", "")
                        if not dealer_id:
                            redir_url = dc.get("redirection_Url", "")
                            if "dealerid=" in redir_url:
                                dealer_id = redir_url.split("dealerid=")[-1].split("&")[0]
                        
                        formatted_dealer_cards.append({
                            "name": dc.get("name", ""),
                            "adress": dc.get("address", ""),  # Note: 'adress' per spec (typo)
                            "dealer_id": str(dealer_id)
                        })
                    
                    # Store formatted cards for response building
                    result["_dealer_cards_formatted"] = formatted_dealer_cards
                    
                    # Detect if this is a dealer-name query (Croma, Chroma, Reliance Digital, etc.)
                    _KNOWN_DEALER_NAMES = {
                        "croma", "chroma", "kroma", "reliance digital", "reliance", 
                        "vijay sales", "vijaysales", "sangeetha", "sangeetha mobiles",
                        "poorvika", "lot mobiles", "big c", "bajaj electronics"
                    }
                    q_lower_check = query.strip().lower()
                    is_known_dealer = any(
                        q_lower_check == dn or q_lower_check.startswith(dn + " ") or 
                        q_lower_check.endswith(" " + dn) or dn in q_lower_check
                        for dn in _KNOWN_DEALER_NAMES
                    )
                    
                    # Set is_dealer_query if it's a known dealer name query
                    result["_is_dealer_query"] = is_known_dealer
                    result["_dealer_query_name"] = query.strip() if is_known_dealer else None
                    
                    # Append dealer keywords into the existing keywords list
                    # so they appear in the same suggestion dropdown
                    existing_keywords = result["response"].get("keywords", [])
                    for dk in dealer_keywords:
                        if dk not in existing_keywords:
                            existing_keywords.append(dk)
                    result["response"]["keywords"] = existing_keywords
                    
                    logger.info(
                        f"Dealer autosuggest: {len(dealer_cards)} cards, "
                        f"{len(dealer_keywords)} keywords merged for '{query}'"
                        f"{' (prefix-match)' if _is_dealer_prefix else ''}"
                    )
            except Exception as de:
                logger.warning(f"Dealer autosuggest failed (non-fatal): {de}")
        
        # =================== BUILD FINAL RESPONSE ===================
        # Transform internal result format to required API format
        inner_response = result.get("response", {})
        
        # Build Card section with dealer info
        card_section = []
        formatted_dealers = result.get("_dealer_cards_formatted", [])
        if formatted_dealers:
            card_section = [{"dealer": formatted_dealers}]
        
        # Determine filter_text based on query type
        filter_text = inner_response.get("filter_text", "")
        is_dealer_query = result.get("_is_dealer_query", False)
        dealer_query_name = result.get("_dealer_query_name")
        if is_dealer_query and formatted_dealers:
            # For dealer-specific queries like "Chroma", show "Chroma stores near you"
            display_name = dealer_query_name if dealer_query_name else query.strip()
            # Capitalize first letter for display
            display_name = display_name.title() if display_name else query.strip().title()
            filter_text = f"{display_name} stores near you"
        
        final_response = {
            "data": {
                "autoSuggest": {
                    "message": "no autosuggest issue",
                    "response": {
                        "chips": inner_response.get("chips", []),
                        "Card": card_section,
                        "filter_text": filter_text,
                        "keywords": inner_response.get("keywords", []),
                        "language": inner_response.get("language", "english"),
                        "__typename": "SuggestResponseData"
                    },
                    "__typename": "AutoSuggestOutput"
                }
            }
        }
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Autosuggest API: '{query}' -> {len(inner_response.get('keywords', []))} suggestions in {elapsed:.2f}ms")
        
        return jsonify(final_response), 200
        
    except Exception as e:
        logger.error(f"Autosuggest error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "data": {
                "autoSuggest": {
                    "message": f"Internal server error: {str(e)}",
                    "response": {
                        "chips": [],
                        "Card": [],
                        "filter_text": "",
                        "keywords": [],
                        "language": "english",
                        "__typename": "SuggestResponseData"
                    },
                    "__typename": "AutoSuggestOutput"
                }
            }
        }), 500

'''
# =================== DEALER SEARCH CONTROL ENDPOINTS ===================
@app.route('/api/dealer_search/status', methods=['GET'])
def dealer_search_status():
    """Get dealer search status"""
    return jsonify({
        "dealer_search_enabled": is_dealer_search_enabled(),
        "dealer_index": DEALER_INDEX_NAME,
        "config_enabled": DEALER_SEARCH_ENABLED
    })


@app.route('/api/dealer_search/toggle', methods=['POST'])
def toggle_dealer_search():
    """
    Enable or disable dealer search at runtime
    POST: {"enabled": true} or {"enabled": false}
    """
    global DEALER_SEARCH_ENABLED
    
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled')
        
        if enabled is None:
            return jsonify({"error": "Missing 'enabled' parameter"}), 400
        
        DEALER_SEARCH_ENABLED = bool(enabled)
        
        return jsonify({
            "status": "success",
            "dealer_search_enabled": is_dealer_search_enabled(),
            "message": f"Dealer search {'enabled' if DEALER_SEARCH_ENABLED else 'disabled'}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dealer_search', methods=['POST', 'GET'])
def standalone_dealer_search():
    """
    Standalone dealer search endpoint
    This endpoint ONLY returns dealer results, ignoring product search
    """
    if not is_dealer_search_enabled():
        return jsonify({
            "status": False,
            "message": "Dealer search is disabled",
            "data": {"dealers": [], "totaldealers": 0}
        }), 400
    
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args.to_dict()
        
        query = data.get('searchterm', data.get('query', data.get('q', ''))).strip()
        city_id = data.get('city_id', data.get('cityId'))
        customer_lat = data.get('Customer_lat', data.get('lat'))
        customer_long = data.get('Customer_long', data.get('lon', data.get('lng')))
        categories = data.get('categories', data.get('category', []))
        page = int(data.get('page', 1))
        page_size = int(data.get('size', data.get('pageSize', 26)))
        
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split(',') if c.strip()]
        
        return _execute_dealer_search(
            query=query,
            city_id=city_id,
            customer_lat=customer_lat,
            customer_long=customer_long,
            categories=categories,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Standalone dealer search error: {e}")
        return jsonify({"error": str(e)}), 500


# =================== COMPARE SEARCH API ===================
@app.route('/api/mallcomparesearch', methods=['POST'])
def mall_compare_search_api():
    """
    Compare search API for getting product suggestions in a category for comparison.
    Used to help users compare products within the same category.
    Delegates to compare_search_handler.execute_compare_search()

    Request Body:
    {
        "query": "vivo",
        "version": "v13",
        "platform": "web",
        "top_level_category_id": "47",
        "top_level_category_name": "Mobiles"
    }

    Response Body:
    {
        "data": {
            "autoSuggest": {
                "message": "no autosuggest issue",
                "response": [
                    {
                        "modelid": "486633",
                        "Product_Name": "vivo X200 Pro 5G",
                        "top_level_category_id": "47",
                        "top_level_category_name": "Mobiles"
                    }
                ],
                "__typename": "AutoSuggestOutput"
            }
        }
    }
    """
    try:
        # Check if compare search is enabled
        if not is_compare_search_enabled():
            return jsonify({
                "data": {
                    "autoSuggest": {
                        "message": "Compare search is disabled",
                        "response": [],
                        "__typename": "AutoSuggestOutput"
                    }
                }
            }), 503
        
        data = request.get_json() or {}
        logger.info(f"[COMPARE-SEARCH] Request: query='{data.get('query', '')}', category_id='{data.get('top_level_category_id', '')}'")

        query = data.get('query', '').strip()
        # Support both parameter names
        category_id = data.get('category_id') or data.get('top_level_category_id', '')
        category_name = data.get('category_name') or data.get('top_level_category_name', '')

        # Delegate to handler
        response_data, status_code = execute_compare_search(
            query=query,
            category_id=category_id,
            category_name=category_name,
            max_results=20
        )
        
        return jsonify(response_data), status_code

    except Exception as e:
        logger.error(f"[COMPARE-SEARCH-ERROR] error='{str(e)}'")
        return jsonify({
            "data": {
                "autoSuggest": {
                    "message": f"Internal server error: {str(e)}",
                    "response": [],
                    "__typename": "AutoSuggestOutput"
                }
            }
        }), 500



if __name__ == '__main__':
    logger.info("Starting BajajMall Search API v2 on port 8007")
    logger.info(f"Dealer search enabled: {is_dealer_search_enabled()}")
    logger.info(f"Compare search enabled: {is_compare_search_enabled()}")
    # Pre-initialize autosuggest at startup
    get_autosuggest()
    app.run(host='0.0.0.0', port=8564, debug=False)
