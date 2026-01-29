"""
BajajMall Search API v2 - Clean Architecture with BM25 + ES Multi-Match
Handles: Typo correction, long SKU queries, attribute queries, ambiguity, noise
Features: City-level response, filter aggregations, EMI slider
"""

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

# =================== CONFIGURATION ===================
# PRODUCT_INDEX_NAME = "bajajmall_products_s3_esidx3_15102025_chkpt_up26"
# CATEGORY_INDEX_NAME = "bajajmall_categories_s3_esidx3_159102025_chkpt_up26"
# AUTOSUGGEST_INDEX_NAME = "bajajmall_autosuggest_s3_esidx3_15102025_chkpt_up26"
# IMAGE_DOMAIN = "https://mc.bajajfinserv.in/media/catalog/product_up26"


PRODUCT_INDEX_NAME = "bajajmall_products_s3_esidx3_22012026"
CATEGORY_INDEX_NAME = "bajajmall_categories_s3_esidx3_22012026"
AUTOSUGGEST_INDEX_NAME = "bajajmall_autosuggest_s3_esidx3_22012026"
IMAGE_DOMAIN = "https://mc.bajajfinserv.in/media/catalog/product_up26"



# =================== DEALER SEARCH FEATURE FLAG ===================
# Set to True to enable dealer search, False to disable
DEALER_SEARCH_ENABLED = False
DEALER_INDEX_NAME = "bajajmall_dealers_index"

# Setup logging
logging.basicConfig(
    filename="app_v2.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
es = Elasticsearch("http://localhost:9200")

# =================== DEALER SEARCH INITIALIZATION ===================
# Initialize dealer search if enabled
_dealer_search_engine = None
_dealer_intent_classifier = None

if DEALER_SEARCH_ENABLED:
    try:
        import sys
        import os
        # Add Dealersearch to path
        dealersearch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dealersearch')
        if dealersearch_path not in sys.path:
            sys.path.insert(0, dealersearch_path)
        
        from dealer_search import DealerSearchEngine
        from dealer_intent_classifier import DealerIntentClassifier, SearchIntent
        from dealer_search_api_v2 import DealerResponseFormatter, AutosuggestDealerFormatter
        
        _dealer_search_engine = DealerSearchEngine(es, DEALER_INDEX_NAME)
        _dealer_intent_classifier = DealerIntentClassifier()
        logger.info("✅ Dealer search initialized successfully")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize dealer search: {e}")
        DEALER_SEARCH_ENABLED = False


def is_dealer_search_enabled():
    """Check if dealer search is enabled and initialized"""
    return DEALER_SEARCH_ENABLED and _dealer_search_engine is not None


# Load attribute ID to name mapping (using new 20112025 file)
try:
    with open('/datadrive1/deepak/new_mall_pipeline/data/attribute_id_name_mapping_27012026.json', 'r', encoding='utf-8') as f:
        ATTRIBUTE_ID_NAME_MAP = json.load(f)
except Exception as e:
    logger.error(f"Failed to load attribute mapping: {e}")
    ATTRIBUTE_ID_NAME_MAP = {}

# Build ALL_ATTRIBUTE_OPTIONS from the two lists
EXCLUDED_ATTRIBUTES = {"attribute_fuelType_value", "attribute_capacity_new", "attribute_capacity_new_value",
                        "attribute_fuelType", "attribute_hd_type", "attribute_hd_type_value",
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
    "onida", "videocon", "croma", "intex", "daikin", "hitachi", "blue star", "o general",
    # Audio brands
    "philips", "jbl", "bose", "boat", "zebronics", "harman kardon", "marshall", "sennheiser", "mivi",
    # Camera/Gaming brands
    "canon", "nikon", "fujifilm", "gopro", "dji", "insta360", "logitech", "razer", "corsair",
    # Home appliance brands
    "havells", "crompton", "orient", "usha", "prestige", "pigeon", "butterfly", "preethi", "maharaja", "sujata", "vidiem", "atomberg",
    "kent", "livpure", "aquaguard", "pureit", "eureka forbes", "ao smith", "racold",
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
    "realme gt", "realmegt", "gt neo",
    "realme c", "realmec", "realme c55", "realme c53", "realme c35",
    "realme 12", "realme12", "realme 11", "realme11",
    "realme 10", "realme10", "realme 9", "realme9",
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
    "bookshelf": ["bookshelf", "bookshelves", "shelf", "shelves", "rack", "book rack"],
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
    # Apple + laptop
    "apple macbook": "laptops",
    "apple laptop": "laptops",
    "applemacbook": "laptops",
    "apple mac book": "laptops",
    
    # ===== iMac (all variations) =====
    # NOTE: iMac is categorized as "unknown" in ES, not "laptops"
    "imac": "unknown",
    "i mac": "unknown",
    "imak": "unknown",
    "i-mac": "unknown",
    "imacc": "unknown",
    "imac 24": "unknown",
    "imac 27": "unknown",
    "imac pro": "unknown",
    "imacpro": "unknown",
    "apple imac": "unknown",
    "appleimac": "unknown",
    "apple desktop": "unknown",
    "apple computer": "unknown",
    "apple pc": "unknown",
    
    # ===== Mac (general) =====
    "mac": "laptops",
    "mac mini": "laptops",
    "macmini": "laptops",
    "mac studio": "laptops",
    "macstudio": "laptops",
    "mac desktop": "laptops",
    "apple mac": "laptops",
    "applemac": "laptops",
    
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
    
    # ===== Missing vowels =====
    "phn", "iphn", "phne", "iphne", "phon", "fon", "ifn", "ipn",
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
    # iPhone 15 variations
    "iphone 15", "iphone15", "i phone 15", "iphone15pro", "iphone 15 pro",
    "ifone 15", "ipone 15", "iphon 15", "iph 15", "ip 15",
    "apple 15", "appl 15", "aple 15", "15 pro max", "15 promax",
    "iphone 15 pro max", "iphone15promax", "15promax", "i15", "ip15",
    
    # iPhone 16 variations  
    "iphone 16", "iphone16", "i phone 16", "iphone16pro", "iphone 16 pro",
    "ifone 16", "ipone 16", "iphon 16", "iph 16", "ip 16",
    "apple 16", "appl 16", "aple 16", "16 pro max", "16 promax",
    "iphone 16 pro max", "iphone16promax", "16promax", "i16", "ip16",
    
    # iPhone 14 variations
    "iphone 14", "iphone14", "i phone 14", "iphone14pro", "iphone 14 pro",
    "ifone 14", "ipone 14", "iphon 14", "iph 14", "ip 14",
    "apple 14", "appl 14", "i14", "ip14",
    
    # iPhone 13 variations
    "iphone 13", "iphone13", "i phone 13", "iphone13pro", "iphone 13 pro",
    "ifone 13", "ipone 13", "iphon 13", "i13", "ip13",
    
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
    "mutual fund", "mutual funds", "mutualfund", "mutualfunds", "mf", "sip", "systematic investment",
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
    "home loan", "homeloan", "personal loan", "personalloan", "car loan", "carloan",
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
    """
    if not query:
        return None
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())
    query_no_space = query_lower.replace(" ", "")
    
    # Check for non-phone Apple products using STRICT WORD BOUNDARY matching
    # This prevents "washing machine" from matching "mac"
    for product, category in APPLE_NON_PHONE_PRODUCTS.items():
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
    non_apple_brands = {
        "samsung", "vivo", "oppo", "realme", "oneplus", "xiaomi", "mi", "redmi", 
        "poco", "motorola", "nokia", "tecno", "infinix", "iqoo", "nothing", "google", "pixel",
        "hp", "dell", "lenovo", "asus", "acer", "msi", "lg", "sony", "jbl", "boat", "boult",
        "noise", "fire-boltt", "fireboltt", "amazfit", "samsung galaxy", "galaxy"
    }
    for brand in non_apple_brands:
        if brand in query_lower:
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
    excluded_words = {"phone", "phones", "phon", "phne", "phn", "mobile", "mobiles", "android", 
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
        "macbok pro": "macbook pro", "makbook pro": "macbook pro",
        "macbok air": "macbook air", "makbook air": "macbook air",
        "mac book": "macbook", "mac bok": "macbook"
    }
    if query_cleaned in macbook_typo_map:
        query_lower = macbook_typo_map[query_cleaned]
        return query_lower
    if query_lower.replace(" ", "") in macbook_typo_map:
        query_lower = macbook_typo_map[query_lower.replace(" ", "")]
        return query_lower
    
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
    }
    
    # Check direct match first
    if query_cleaned in noisy_to_iphone:
        query_lower = noisy_to_iphone[query_cleaned]
    elif query_lower.replace(" ", "") in noisy_to_iphone:
        query_lower = noisy_to_iphone[query_lower.replace(" ", "")]
    else:
        # ===== STEP 2: Pattern-based normalization =====
        iphone_patterns = [
            # Space variations
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
    
    # iPad models
    "ipad": "iPad", "ipd": "iPad", "ipda": "iPad", "i pad": "iPad", "i-pad": "iPad",
    "ipaad": "iPad", "ipadd": "iPad", "ippad": "iPad",
    
    # MacBook models
    "macbook": "MacBook", "macbok": "MacBook", "macbuk": "MacBook", "makbook": "MacBook",
    "makbok": "MacBook", "mac book": "MacBook", "macboo": "MacBook", "macbookk": "MacBook",
    "macboook": "MacBook", "mcbook": "MacBook", "macbok": "MacBook",
    
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
        (r'\bmatress\b', 'mattress'), (r'\bmattres\b', 'mattress'),
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
    vivo_pattern = r'\b(vivo)\s*([xyvt])\s*(\d+)\s*(pro|plus|lite|ultra|fe|e|s|a|i|x|z)?\b'
    def normalize_vivo(m):
        brand, series, num, suffix = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        return f"{brand} {series}{num} {suffix}".strip()
    query = re.sub(vivo_pattern, normalize_vivo, query, flags=re.IGNORECASE)
    
    # Oppo model: oppof19 → oppo f19
    oppo_pattern = r'\b(oppo)\s*([afkr])\s*(\d+)\s*(pro|plus|lite|s|x|k)?\b'
    def normalize_oppo(m):
        brand, series, num, suffix = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        return f"{brand} {series}{num} {suffix}".strip()
    query = re.sub(oppo_pattern, normalize_oppo, query, flags=re.IGNORECASE)
    
    # Realme model: realme12 → realme 12
    realme_pattern = r'\b(realme)\s*(\d+|gt|narzo|c|p)\s*(\d*)\s*(pro|plus|neo|master|ultra)?\b'
    def normalize_realme(m):
        brand, series, num, suffix = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
        parts = [brand, series]
        if num: parts.append(num)
        if suffix: parts.append(suffix)
        return ' '.join(parts)
    query = re.sub(realme_pattern, normalize_realme, query, flags=re.IGNORECASE)
    
    # Samsung model: samsungs24 → samsung s24, samsung galaxy s24
    samsung_pattern = r'\b(samsung)\s*(galaxy)?\s*([asmzf])\s*(\d+)\s*(fe|ultra|plus|lite|s)?\b'
    def normalize_samsung(m):
        brand, galaxy, series, num, suffix = m.group(1), m.group(2) or "", m.group(3), m.group(4), m.group(5) or ""
        parts = [brand]
        if galaxy: parts.append(galaxy)
        parts.append(f"{series}{num}")
        if suffix: parts.append(suffix)
        return ' '.join(parts)
    query = re.sub(samsung_pattern, normalize_samsung, query, flags=re.IGNORECASE)
    
    # Redmi model: redminote13 → redmi note 13
    redmi_pattern = r'\b(redmi)\s*(note)?\s*(\d+)\s*(pro|plus|a|c|s|i)?\b'
    def normalize_redmi(m):
        brand, note, num, suffix = m.group(1), m.group(2) or "", m.group(3), m.group(4) or ""
        parts = [brand]
        if note: parts.append(note)
        parts.append(num)
        if suffix: parts.append(suffix)
        return ' '.join(parts)
    query = re.sub(redmi_pattern, normalize_redmi, query, flags=re.IGNORECASE)
    
    # iPhone model: iphone15 → iphone 15
    iphone_pattern = r'\b(iphone)\s*(\d+)\s*(pro|plus|max|mini|se)?\s*(max)?\b'
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
        "laptop": ["laptop", "laptops", "notebook", "notebooks", "ultrabook"],
        "tablet": ["tablet", "tablets", "tab", "ipad"],
        # Appliances
        "refrigerator": ["refrigerator", "refrigerators", "fridge", "fridges", "ref", "double door", "single door"],
        "washing machine": ["washing machine", "washer", "washers", "wm", "front load", "top load"],
        "air conditioner": ["air conditioner", "ac", "acs", "airconditioner", "split ac", "window ac", "inverter ac"],
        "microwave oven": ["microwave", "microwave oven", "oven", "microwaves", "otg"],
        "water heater": ["geyser", "geysers", "water heater", "water heaters", "instant geyser", "storage geyser"],
        "water purifier": ["ro", "water purifier", "purifier", "water filter", "kent", "aquaguard"],
        "air cooler": ["cooler", "air cooler", "coolers", "desert cooler", "room cooler"],
        "air fryer": ["air fryer", "airfryer", "air fryers"],
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
                # BUT only apply boost if lengths are similar (within 2 chars) to prevent
                # "car" matching "carrier" or "refrigerator" matching "royal enfield"
                len_diff = abs(len(word_lower) - len(variation.lower()))
                if len_diff <= 2 and (word_lower in variation.lower() or variation.lower() in word_lower):
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
    "refrigerator", "fridge", "ac", "washing", "machine", "laptop", "tablet", 
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
            (r'\bsamsungs25\b', 'samsung s25'), (r'\bsamsungs24\b', 'samsung s24'),
            (r'\bsamsungs23\b', 'samsung s23'), (r'\bsamsungm34\b', 'samsung m34'),
            (r'\bsamsungf54\b', 'samsung f54'), (r'\bsamsungf55\b', 'samsung f55'),
            
            (r'\biphone16promax\b', 'iphone 16 pro max'), (r'\biphone15promax\b', 'iphone 15 pro max'),
            (r'\biphone16pro\b', 'iphone 16 pro'), (r'\biphone15pro\b', 'iphone 15 pro'),
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
            (r'\bfrig\b', 'fridge'), (r'\bfrdge\b', 'fridge'),
            (r'\brefridgerator\b', 'refrigerator'), (r'\brefregerator\b', 'refrigerator'),
            # Kitchen appliance typos
            (r'\bmiksi\b', 'mixer'), (r'\bmikser\b', 'mixer'), (r'\bmixar\b', 'mixer'),
            (r'\bgijer\b', 'geyser'), (r'\bgijar\b', 'geyser'), (r'\bgizer\b', 'geyser'), (r'\bgeysar\b', 'geyser'),
            (r'\bjusar\b', 'juicer'), (r'\bjuisar\b', 'juicer'), (r'\bjucer\b', 'juicer'),
            (r'\bfarnichar\b', 'furniture'), (r'\bfurnichar\b', 'furniture'), (r'\bfurnitur\b', 'furniture'),
            (r'\bfarniture\b', 'furniture'), (r'\bfernicher\b', 'furniture'),
            
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
            (r'\bbluestarr\b', 'bluestar'), (r'\bblue\s*star\b', 'bluestar'),
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
        
        # =================== VIVO MODEL NORMALIZATION ===================
        # Handle Vivo model variations: "vivox200", "vivo x 200", "vivox 200", "vivo x200 pro" etc.
        # Normalize to proper format: "vivo x200", "vivo x200 pro"
        
        # Pattern: vivo + optional space + model letter + optional space + number + optional suffix
        # Matches: vivox200, vivo x200, vivo x 200, vivox 200, vivox200pro, vivo x200 pro, vivo x 200 pro
        vivo_pattern = r'\b(vivo)\s*([xyvt])\s*(\d+)\s*(pro|plus|lite|ultra|fe|e|s|a|i|x|z)?\b'
        
        def normalize_vivo(match):
            brand = match.group(1)  # vivo
            series = match.group(2)  # x, y, v, t
            number = match.group(3)  # 200, 100, 40, etc.
            suffix = match.group(4) or ""  # pro, plus, lite, etc.
            if suffix:
                return f"{brand} {series}{number} {suffix}"
            return f"{brand} {series}{number}"
        
        query_lower = re.sub(vivo_pattern, normalize_vivo, query_lower, flags=re.IGNORECASE)
        
        # Also handle: "vivo neo", "vivo iqoo" style queries
        # Handle "vivoy" standalone -> "vivo y" etc.
        query_lower = re.sub(r'\bvivo([xyvt])(\s|$)', r'vivo \1\2', query_lower)
        
        # =================== OPPO MODEL NORMALIZATION ===================
        # Handle Oppo model variations: "oppof19", "oppo f 19", "oppoa53" etc.
        oppo_pattern = r'\b(oppo)\s*([afkr])\s*(\d+)\s*(pro|plus|lite|s|x|k)?\b'
        
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
        realme_pattern = r'\b(realme)\s*(\d+|gt|narzo|c|p)\s*(\d*)\s*(pro|plus|neo|master|ultra)?\b'
        
        def normalize_realme(match):
            brand = match.group(1)  # realme
            series = match.group(2)  # number or gt/narzo/c/p
            number = match.group(3) or ""  # additional number
            suffix = match.group(4) or ""
            parts = [brand, series]
            if number:
                parts.append(number)
            if suffix:
                parts.append(suffix)
            return " ".join(parts)
        
        query_lower = re.sub(realme_pattern, normalize_realme, query_lower, flags=re.IGNORECASE)
        
        # =================== SAMSUNG MODEL NORMALIZATION ===================
        # Handle Samsung Galaxy variations: "samsunggalaxys24", "samsung galaxy s 24" etc.
        samsung_pattern = r'\b(samsung)\s*(galaxy)?\s*([asmzf])\s*(\d+)\s*(fe|ultra|plus|lite|s)?\b'
        
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
        redmi_pattern = r'\b(redmi)\s*(note)?\s*(\d+)\s*(pro|plus|a|c|s|i)?\b'
        
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
        poco_pattern = r'\b(poco)\s*([mxfc])\s*(\d+)\s*(pro|plus|gt)?\b'
        
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
        # Handle OnePlus variations: "oneplus12", "one plus 12 pro" etc.
        # First normalize "one plus" -> "oneplus" to avoid double "plus"
        query_lower = re.sub(r'\bone\s+plus\b', 'oneplus', query_lower, flags=re.IGNORECASE)
        query_lower = re.sub(r'\b1\s*\+', 'oneplus', query_lower, flags=re.IGNORECASE)
        
        oneplus_pattern = r'\b(oneplus)\s*(\d+|nord|ace)\s*(\d*)\s*(pro|r|ce|t)?\b'
        
        def normalize_oneplus(match):
            brand = "oneplus"  # normalize brand
            series = match.group(2)  # number or nord/ace
            number = match.group(3) or ""
            suffix = match.group(4) or ""
            parts = [brand, series]
            if number:
                parts.append(number)
            if suffix:
                parts.append(suffix)
            return " ".join(parts)
        
        query_lower = re.sub(oneplus_pattern, normalize_oneplus, query_lower, flags=re.IGNORECASE)
        
        # =================== MOTOROLA MODEL NORMALIZATION ===================
        # Handle Motorola variations: "motorolag84", "motorola g 84" etc.
        moto_pattern = r'\b(motorola|moto)\s*([gex])\s*(\d+)\s*(power|plus|stylus|pro)?\b'
        
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
        compound_normalizations = {
            "mobilephone": "mobile phone",
            "mobail": "mobile phone",
            "mobil": "mobile phone",
            "mobale": "mobile phone",
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
                          "electric", "ev", "electrical"}
        
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
        
        # print("Corrected Query for Category Detection:", query_lower)
        # print("Query Words Set:", query_words)
        # print("Query No Space:", query_no_space)
                
            
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
        for keyword in TVS_KEYWORDS:
            if keyword in query_lower:
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
            
        television_keywords = [
            "television", "tv", "led tv", "smart tv", "4k tv", "8k tv",
            "lcd tv", "oled tv", "qled tv"
        ]
        for keyword in television_keywords:
            if keyword in query_lower or keyword in query_words:
                return "television"
            
        mobile_keyword = [
            "motorola", "motrola", "moto g", "moto g series", "moto e", "moto e series",
            # Infinix brand - primarily smartphone brand, NOT TV
            "infinix", "infnix", "infinx", "infinix hot", "infinix note", "infinix zero",
            "infinix smart", "infinix gt",
            # Vivo brand (for queries like "vivo x300" to return smartphones)
            "vivo", "vevo", "viovo", "vivo x", "vivo y", "vivo v", "vivo t",
            # Other phone brands that should force smartphone category
            "oppo", "realme", "redmi", "poco", "iqoo", "tecno", "lava", "itel",
            "nothing phone", "google pixel",
        ]
        for keyword in mobile_keyword:
            if keyword in query_lower or keyword in query_words:
                return "smartphone"  # FIXED: was "smarphone" (typo)
        
        # =================== HIGH PRIORITY: AUDIO VIDEO ===================
        # Priority 0.85: Audio/Earphones/Earbuds - MUST come before smartphone detection
        # This ensures "earphones", "earbuds" return audio products, not smartphones
        for keyword in AUDIO_VIDEO_KEYWORDS:
            if keyword in query_lower or keyword in query_words:
                return "audio video"
        
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
        scooter_typos = ["scooter", "scooty", "scootr", "scouter", "scoter", "skutar", "scootie", "scutty"]
        bike_typos = ["bike", "byke", "motorbike", "motorcycle"]
        for word in query_words:
            if len(word) >= 4:
                for typo in scooter_typos + bike_typos:
                    if fuzz.ratio(word, typo) >= 80:
                        return "two wheeler"
        
        # Priority 2: Remaining four-wheeler keywords (models etc)
        for keyword in FOUR_WHEELER_KEYWORDS:
            if keyword in query_words or keyword in query_lower:
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
        
        # Priority 0.26: Check for washing machine keywords (including typos)
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
        
        # Priority 0.5: Check Realme keywords (handles "relme", "ralame", typos)
        for keyword in REALME_KEYWORDS:
            if keyword in query_lower:
                return "realme"
        
        # Sort brands by length (longer first) to match "samsung mobiles" before "samsung"
        sorted_brands = sorted(BRAND_NAMES, key=len, reverse=True)
        
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
                return brand_lower
        
        # =================== FUZZY FALLBACK FOR BRAND ===================
        # Try fuzzy matching for brand names as a last resort
        for word in query_words:
            if len(word) >= 3:  # Only try fuzzy on words with 3+ chars
                brand = get_brand_from_fuzzy(word, threshold=0.70)
                if brand:
                    logger.info(f"Fuzzy brand match: '{word}' → brand '{brand}'")
                    return brand
        
        return None
    
    def extract_attributes(self, query: str) -> Dict[str, Any]:
        """Extract product attributes like storage, RAM, color from query"""
        attrs = {}
        query_lower = query.lower()
        
        # Storage patterns (128gb, 256 gb, etc.)
        storage_match = re.search(r'(\d+)\s*(gb|tb)', query_lower)
        if storage_match:
            size = int(storage_match.group(1))
            unit = storage_match.group(2)
            if unit == 'tb':
                size *= 1024
            attrs['storage'] = size
        
        # RAM patterns (8gb ram, 12 gb ram)
        ram_match = re.search(r'(\d+)\s*gb\s*ram', query_lower)
        if ram_match:
            attrs['ram'] = int(ram_match.group(1))
        
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
        
        # Color patterns
        colors = ['black', 'white', 'silver', 'gold', 'blue', 'red', 'green', 'grey', 'gray', 
                  'purple', 'pink', 'orange', 'yellow', 'brown', 'titanium', 'graphite', 'midnight']
        for color in colors:
            if color in query_lower:
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
            processed = self.correct_typos(query)
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
            "is_electric_scooter_query": is_electric_scooter_query
        }


# =================== SEARCH ENGINE ===================
class SearchEngine:
    """Handles Elasticsearch query building and execution with BM25"""
    
    def __init__(self, es_client, index_name: str):
        self.es = es_client
        self.index = index_name
    
    def build_query(self, query_info: Dict, city_id: str = None, filters: Dict = None,
                    page: int = 1, page_size: int = 20, emi_range: Dict = None,
                    sort_by: str = None, from_offset: int = None) -> Dict:
        """Build Elasticsearch query with BM25 multi-match
        
        Args:
            sort_by: Sorting option - 'relevance' (default), 'emi_low_high', 'emi_high_low', 
                     'price_low_high', 'price_high_low', 'newest'
            from_offset: Direct offset for pagination (overrides page calculation if provided)
        """
        
        query = query_info.get("processed", "")
        category = query_info.get("category")
        brand = query_info.get("brand")
        attributes = query_info.get("attributes", {})
        is_apple_query = query_info.get("is_apple_query", False)
        product_boost = query_info.get("product_boost", {})
        phone_brand_boost = query_info.get("phone_brand_boost", [])
        is_brand_only_phone_query = query_info.get("is_brand_only_phone_query", False)
        
        # Build must clauses
        must_clauses = []
        should_clauses = []
        filter_clauses = []
        
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
            if category:
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
            # E.g., "reno 14 pro" should rank higher than "reno10 pro+" for query "reno 14 pro"
            should_clauses.append({
                "match_phrase": {
                    "product_name": {
                        "query": query,
                        "boost": 10
                    }
                }
            })
            should_clauses.append({
                "match_phrase": {
                    "sku_name": {
                        "query": query,
                        "boost": 8
                    }
                }
            })
        
        # Category anchor - use filter for exact match (skip if Apple query already set it)
        if category and not is_apple_query:
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
        if brand and not is_apple_query:
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
        
        # Attribute filters from query
        if attributes.get("storage"):
            should_clauses.append({
                "match": {
                    "attribute_internal_storage_value": str(attributes["storage"]) + " GB"
                }
            })
        
        if attributes.get("ram"):
            should_clauses.append({
                "match": {
                    "attribute_ram_value": str(attributes["ram"]) + " GB"
                }
            })
        
        if attributes.get("screen_size"):
            should_clauses.append({
                "match": {
                    "attribute_screen_size_in_inches_value": str(attributes["screen_size"])
                }
            })
        
        if attributes.get("tonnage"):
            should_clauses.append({
                "match": {
                    "attribute_capacity_in_tons_value": str(attributes["tonnage"])
                }
            })
        
        if attributes.get("capacity_kg"):
            should_clauses.append({
                "match": {
                    "attribute_capacity_wm_value": str(attributes["capacity_kg"])
                }
            })
        
        if attributes.get("color"):
            should_clauses.append({
                "match": {
                    "attribute_color_value": attributes["color"]
                }
            })
        
        # NOTE: User filters are now applied via post_filter for multi-selection support
        # This allows aggregations to show ALL options, not just filtered ones
        # See build_query() where post_filter is added to es_query
        
        # City filter (nested) - We want to return ALL city_offers in inner_hits
        # so we can select city-specific pricing or fallback to citi_id_0
        if city_id:
            filter_clauses.append({
                "nested": {
                    "path": "city_offers",
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"city_offers.cityid": city_id}},
                                {"term": {"city_offers.cityid": "citi_id_0"}}
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    # Return more inner_hits to get both city-specific and global offers
                    "inner_hits": {
                        "size": 10,
                        "sort": [
                            # Sort so exact city_id match comes first
                            {
                                "_script": {
                                    "type": "number",
                                    "script": {
                                        "source": f"doc['city_offers.cityid'].value == params.target_city ? 0 : 1",
                                        "params": {"target_city": city_id}
                                    },
                                    "order": "asc"
                                }
                            }
                        ]
                    }
                }
            })
        
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
        
        # MOP (price) range filter - NEW: support for "under 30000" without EMI mention
        if filters and 'price_max' in filters:
            price_filter = {"range": {"mop": {"lte": filters['price_max']}}}
            filter_clauses.append(price_filter)
        elif filters and 'price_range' in filters:
            price_range_filter = {"range": {"mop": {}}}
            if "gte" in filters['price_range']:
                price_range_filter["range"]["mop"]["gte"] = filters['price_range']["gte"]
            if "lte" in filters['price_range']:
                price_range_filter["range"]["mop"]["lte"] = filters['price_range']["lte"]
            filter_clauses.append(price_range_filter)
        
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
        final_query = {"bool": bool_query}
        
        if phone_brand_boost and not is_apple_query:
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
        if is_brand_only_phone_query and not is_apple_query and not phone_brand_boost:
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
        if product_boost.get("boost_type") and not is_apple_query and not phone_brand_boost:
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
        
        # =================== EARBUDS BOOST ===================
        # For earbuds queries, boost actual earbuds over headphones
        original_query = query_info.get("original", "").lower()
        if "earbud" in original_query or "ear bud" in original_query:
            earbud_functions = [
                # Boost products with "earbud" or "buds" in name
                {"filter": {"match": {"product_name": "earbuds"}}, "weight": 20},
                {"filter": {"match": {"product_name": "buds"}}, "weight": 15},
                {"filter": {"match": {"product_name": "tws"}}, "weight": 15},
                {"filter": {"match": {"product_name": "truly wireless"}}, "weight": 10},
                # Demote headphones
                {"filter": {"match": {"product_name": "headphone"}}, "weight": 0.3},
                {"filter": {"match": {"product_name": "over ear"}}, "weight": 0.2},
                {"filter": {"match": {"product_name": "on ear"}}, "weight": 0.2},
            ]
            final_query = {
                "function_score": {
                    "query": final_query,
                    "functions": earbud_functions,
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            }
            logger.info("Earbuds boost applied - prioritizing earbuds over headphones")
        
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
        
        # Build post_filter for user-selected filters (multi-selection support)
        # post_filter applies AFTER aggregations, so all filter options remain visible
        post_filter_clauses = []
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
        
        es_query = {
            "query": final_query,
            "from": calculated_from,
            "size": page_size,
            "track_total_hits": True,  # Get accurate total count after collapse
            "aggs": self._build_aggregations(city_id),
            "collapse": {
                "field": "modelid",
                "inner_hits": {
                    "name": "sku_variants",
                    "size": 10
                }
            }
        }
        
        # =================== SORTING (3 OPTIONS ONLY) ===================
        # Supported sort options:
        # 1. "relevance" (default) - Best match based on query
        # 2. "low_to_high" - Price/EMI lowest to highest
        # 3. "high_to_low" - Price/EMI highest to lowest
        sort_by_lower = (sort_by or "").lower().strip()
        
        if sort_by_lower in ("low_to_high", "lowtohigh", "asc", "ascending"):
            # Sort by lowest EMI ascending (requires nested sort for city_offers)
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
            logger.info("Sorting: Low to High (EMI ascending)")
            
        elif sort_by_lower in ("high_to_low", "hightolow", "desc", "descending"):
            # Sort by lowest EMI descending
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
            logger.info("Sorting: High to Low (EMI descending)")
            
        else:
            # Default: relevance (score) with secondary priority for flagged products
            # Applies when sort_by is "relevance", None, or any unrecognized value
            # CHANGE: Score is now TOP priority, then flags, to ensure exact matches rank first
            # E.g., "oppo k13" should return "K13" before "K13x" even if K13x has new_launch_flag
            es_query["sort"] = [
                {"_score": {"order": "desc"}},  # ES score is TOP priority for relevance
                {
                    "_script": {
                        "type": "number",
                        "script": {
                            "lang": "painless",
                            "source": """
                                // Check if product has new_launch_flag or most_viewed_flag in city_offers
                                def hasFlag = false;
                                if (doc.containsKey('city_offers.new_launch_flag') && 
                                    doc['city_offers.new_launch_flag'].size() > 0) {
                                    for (int i = 0; i < doc['city_offers.new_launch_flag'].length; i++) {
                                        if (doc['city_offers.new_launch_flag'][i] == true) {
                                            hasFlag = true;
                                            break;
                                        }
                                    }
                                }
                                if (!hasFlag && doc.containsKey('city_offers.most_viewed_flag') && 
                                    doc['city_offers.most_viewed_flag'].size() > 0) {
                                    for (int i = 0; i < doc['city_offers.most_viewed_flag'].length; i++) {
                                        if (doc['city_offers.most_viewed_flag'][i] == true) {
                                            hasFlag = true;
                                            break;
                                        }
                                    }
                                }
                                return hasFlag ? 1 : 0;
                            """
                        },
                        "order": "desc"
                    }
                },
                {"model_launch_date": {"order": "desc"}},
                {"modelid": {"order": "asc"}}
            ]
            logger.info("Sorting: Relevance (score first, then flags)")
        
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
    
    def _build_aggregations(self, city_id: str = None) -> Dict:
        """Build aggregations for filters, EMI range, and unique model count"""
        aggs = {}
        
        # =================== UNIQUE MODEL COUNT ===================
        # Cardinality aggregation to get accurate unique model count for pagination
        aggs["unique_models"] = {
            "cardinality": {
                "field": "modelid",
                "precision_threshold": 40000  # High precision for accuracy
            }
        }
        
        # Attribute aggregations
        for option in ALL_ATTRIBUTE_OPTIONS:
            es_key = option['es_key']
            aggs[es_key] = {
                "terms": {
                    "field": es_key,
                    "size": 1000,
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
            
            response = self.es.search(index=self.index, body=es_query)
            
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
               page: int = 1, page_size: int = 20, emi_range: Dict = None,
               sort_by: str = None, from_offset: int = None) -> Dict:
        """Execute search and return results with fallback for zero results
        
        Args:
            sort_by: Sorting option - 'relevance', 'emi_low_high', 'emi_high_low', 
                     'price_low_high', 'price_high_low', 'newest'
            from_offset: Direct offset for pagination (if provided, overrides page calculation)
        """
        try:
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
            
            es_query = self.build_query(query_info, city_id, filters, page, page_size, emi_range, sort_by, from_offset)
            
            logger.info(f"ES Query: {json.dumps(es_query, indent=2)}")
            
            response = self.es.search(index=self.index, body=es_query)
            total = response["hits"]["total"]["value"]
            
            # =================== EXACT MODEL MATCH RERANKING ===================
            # Re-rank results to prioritize exact model matches over partial matches
            # E.g., "oppo k13" query should show "K13" before "K13x"
            # This compensates for ES edge ngram tokenizer matching both equally
            if total > 0 and query_info.get("processed"):
                hits = response["hits"]["hits"]
                processed_query = query_info.get("processed", "").lower()
                query_words = processed_query.split()
                
                # Find alphanumeric model numbers in query (e.g., "k13", "reno14", "a55")
                model_patterns = []
                for word in query_words:
                    if len(word) >= 2 and word.isalnum() and re.search(r'\d', word):
                        model_patterns.append(word.lower())
                
                if model_patterns:
                    def calc_exact_match_score(hit):
                        """Calculate bonus score for exact model match (word boundary)"""
                        product_name = hit.get("_source", {}).get("product_name", "").lower()
                        bonus = 0
                        for pattern in model_patterns:
                            # Check if pattern appears as complete word (followed by space, non-alphanumeric, or end)
                            # E.g., "k13 " or "k13," or "k13" at end - but NOT "k13x"
                            import re
                            pattern_regex = rf'\b{re.escape(pattern)}(?![a-z0-9])'
                            if re.search(pattern_regex, product_name):
                                bonus += 1000  # Large bonus for exact word boundary match
                        return bonus
                    
                    # Sort hits by: (exact_match_bonus DESC, original_position ASC)
                    # This keeps ES relevance order but promotes exact matches
                    indexed_hits = [(i, hit, calc_exact_match_score(hit)) for i, hit in enumerate(hits)]
                    sorted_hits = sorted(indexed_hits, key=lambda x: (-x[2], x[0]))
                    response["hits"]["hits"] = [hit for _, hit, _ in sorted_hits]
                    
                    logger.info(f"Reranked {len(hits)} results for exact model match")
            
            # =================== FALLBACK MECHANISM ===================
            # Priority: Keep category, remove brand first (for queries like "Bajaj mixer grinder")
            # This ensures we show mixer grinders from other brands if Bajaj doesn't have them
            
            # FALLBACK 1: If no results AND has both brand AND category, try WITHOUT brand first
            # This prioritizes showing category results (e.g., show other mixer grinders)
            if total == 0 and query_info.get("brand") and query_info.get("category") and not query_info.get("is_apple_query"):
                logger.info(f"Zero results with brand+category, trying without brand (keeping category)...")
                fallback_info = query_info.copy()
                fallback_info["brand"] = None  # Remove brand, keep category
                fallback_query = self.build_query(fallback_info, city_id, filters, page, page_size, emi_range, sort_by)
                response = self.es.search(index=self.index, body=fallback_query)
                total = response["hits"]["total"]["value"]
                logger.info(f"Category-only fallback returned {total} results")
            
            # FALLBACK 2: If still no results, try without category (but keep brand if no category was detected)
            if total == 0 and query_info.get("category") and not query_info.get("is_apple_query"):
                logger.info(f"Zero results with category filter, trying without category...")
                fallback_info = query_info.copy()
                fallback_info["category"] = None
                fallback_query = self.build_query(fallback_info, city_id, filters, page, page_size, emi_range, sort_by)
                response = self.es.search(index=self.index, body=fallback_query)
                total = response["hits"]["total"]["value"]
                logger.info(f"Fallback search returned {total} results")
            
            # FALLBACK 3: If still no results with brand filter, try without brand filter too
            if total == 0 and query_info.get("brand") and not query_info.get("is_apple_query"):
                logger.info(f"Zero results with brand filter, trying broader search...")
                fallback_info = query_info.copy()
                fallback_info["category"] = None
                fallback_info["brand"] = None
                fallback_query = self.build_query(fallback_info, city_id, filters, page, page_size, emi_range, sort_by)
                response = self.es.search(index=self.index, body=fallback_query)
                total = response["hits"]["total"]["value"]
                logger.info(f"Broad fallback search returned {total} results")
            
            # FALLBACK 4: For specific phone model queries (like "vivo x300", "oppo find x9", "realme 15", "infinix 50x")
            # If no results with processed query, try with just brand + category
            # This ensures user sees phones from that brand even if model doesn't exist
            if total == 0 and query_info.get("brand"):
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
                        response = self.es.search(index=self.index, body=fallback_query)
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
                    response = self.es.search(index=self.index, body=fallback_query)
                    total = response["hits"]["total"]["value"]
                    logger.info(f"Two wheeler fallback ({subcategory}) returned {total} results")
            
            # =================== FALLBACK 5: SKU NAME FOCUSED SEARCH ===================
            # If still no results, try a more relaxed search with higher sku_name weight
            # This helps find products where the model name is in sku_name but not product_name
            # SKIP for Apple queries - Apple products must ONLY show Apple brand
            if total == 0 and query_info.get("processed") and not query_info.get("is_apple_query"):
                logger.info(f"Zero results, trying sku_name focused fallback search...")
                original_query = query_info.get("original", "")
                
                # Build a special query focusing on sku_name field with fuzzy matching
                sku_fallback_query = {
                    "query": {
                        "bool": {
                            "should": [
                                # High weight for exact/near match in sku_name
                                {"match": {"sku_name": {"query": original_query, "fuzziness": "AUTO", "boost": 5}}},
                                # Also try search_field with original query
                                {"match": {"search_field": {"query": original_query, "fuzziness": "AUTO", "boost": 3}}},
                                # Wildcard for partial match
                                {"wildcard": {"sku_name": {"value": f"*{original_query.replace(' ', '*')}*", "case_insensitive": True, "boost": 2}}},
                            ],
                            "minimum_should_match": 1
                        }
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
                    fallback_response = self.es.search(index=self.index, body=sku_fallback_query)
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
            if total == 0 and query_info.get("original") and not query_info.get("is_apple_query"):
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
                    fuzzy_fallback_query = {
                        "query": {
                            "bool": {
                                "should": should_clauses,
                                "minimum_should_match": 1
                            }
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
                        fuzzy_response = self.es.search(index=self.index, body=fuzzy_fallback_query)
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
            unique_model_count = aggregations.get("unique_models", {}).get("value", total)
            
            # If cardinality count is available and > 0, use it
            # Otherwise fall back to ES total (which is correct with collapse)
            if unique_model_count > 0:
                accurate_total = unique_model_count
            else:
                accurate_total = total
            
            logger.info(f"Pagination: ES total={total}, unique_models={unique_model_count}, returning={accurate_total}")
            
            return {
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
                "actual_category": source.get("actual_category", ""),
                "mop": source.get("mop", 0),
                "property": [],
                "products": []
            }
            
            # Add city-specific property
            if city_offer:
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
                    "model_city_flag": city_offer.get("model_city_flag", 0),
                    "phone_setup": city_offer.get("phone_setup", 0),
                    "exchange_flag": city_offer.get("exchange_flag", 0),
                    "installation_flag": city_offer.get("installation_flag", 0),
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
                filter_obj["emi"] = {
                    "min": slider_min,
                    "max": slider_max
                }
                logger.info(f"EMI slider in response: min={slider_min}, max={slider_max}")
            
            final_response["data"]["PostV1Productlist"]["data"]["filters"].append(filter_obj)
        
        return final_response


# =================== INITIALIZE COMPONENTS ===================
query_processor = QueryProcessor()
search_engine = SearchEngine(es, PRODUCT_INDEX_NAME)


# =================== DEALER SEARCH HELPER FUNCTIONS ===================
def _execute_dealer_search(query, city_id, customer_lat, customer_long, categories, page, page_size):
    """
    Execute dealer-only search and return formatted response
    """
    if not is_dealer_search_enabled():
        return jsonify({"error": "Dealer search not enabled"}), 400
    
    try:
        from_index = (page - 1) * page_size
        
        # Build location if available
        location = None
        if customer_lat and customer_long:
            try:
                location = {"lat": float(customer_lat), "lon": float(customer_long)}
            except (ValueError, TypeError):
                pass
        
        # Execute dealer search
        logger.info(f"Executing dealer search: query='{query}', city_id={city_id}, location={location}, size={page_size}")
        dealer_result = _dealer_search_engine.search(
            query=query,
            city_id=city_id,
            location=location,
            radius_km=10,  # 10km radius as per AC
            categories=categories if isinstance(categories, list) else [categories] if categories else None,
            size=page_size,
            from_index=from_index
        )
        logger.info(f"Dealer search result: total={dealer_result.get('total', 0)}, dealers={len(dealer_result.get('dealers', []))}")
        
        total = dealer_result.get("total", 0)
        dealers = dealer_result.get("dealers", [])
        
        # Format dealers using response formatter
        formatted_dealers = []
        for idx, dealer in enumerate(dealers):
            is_nearest = (idx == 0)  # First dealer gets "Nearest Store" tag
            formatted_dealers.append(
                DealerResponseFormatter.format_dealer(dealer, is_nearest)
            )
        
        # Build response
        response = {
            "data": {
                "PostV1Productlistsearch": {
                    "status": True,
                    "message": "success",
                    "data": [
                        {
                            "PostV1Dealerlist": {
                                "data": {
                                    "dealer_filters": {
                                        "categories": dealer_result.get("filters", {}).get("categories", [])
                                    },
                                    "dealers": formatted_dealers,
                                    "totaldealers": total,
                                    "original_query": query
                                },
                                "message": "Success" if dealers else "No Search Found",
                                "status": True
                            }
                        }
                    ],
                    "__typename": "REST_b_fd_lcatalog_data_product_list_search_details_interface"
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Dealer search error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "data": {
                "PostV1Productlistsearch": {
                    "status": False,
                    "message": f"Dealer search error: {str(e)}",
                    "data": []
                }
            }
        }), 500


def _get_dealer_results_for_hybrid(query, city_id, customer_lat, customer_long, limit=5):
    """
    Get dealer results for hybrid search (to include with product results)
    """
    if not is_dealer_search_enabled():
        return None
    
    try:
        location = None
        if customer_lat and customer_long:
            try:
                location = {"lat": float(customer_lat), "lon": float(customer_long)}
            except (ValueError, TypeError):
                pass
        
        dealer_result = _dealer_search_engine.search(
            query=query,
            city_id=city_id,
            location=location,
            radius_km=10,
            size=limit
        )
        
        dealers = dealer_result.get("dealers", [])
        if not dealers:
            return None
        
        formatted_dealers = []
        for idx, dealer in enumerate(dealers):
            is_nearest = (idx == 0)
            formatted_dealers.append(
                DealerResponseFormatter.format_dealer(dealer, is_nearest)
            )
        
        return {
            "PostV1Dealerlist": {
                "data": {
                    "dealer_filters": {
                        "categories": dealer_result.get("filters", {}).get("categories", [])
                    },
                    "dealers": formatted_dealers,
                    "totaldealers": dealer_result.get("total", 0),
                    "original_query": query
                },
                "message": "Success",
                "status": True
            }
        }
    except Exception as e:
        logger.error(f"Hybrid dealer search error: {e}")
        return None


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
        
        # Extract parameters
        query = data.get('searchterm', data.get('query', data.get('q', data.get('searchQuery', ''))))
        
        # =================== OUT-OF-SCOPE QUERY GUARDRAIL ===================
        # Check if query is outside business scope BEFORE any processing
        # This saves compute and returns graceful "no results" for irrelevant queries
        if query:
            is_blocked, blocked_category = is_out_of_scope_query(query)
            if is_blocked:
                return jsonify(get_out_of_scope_response(query, blocked_category))
        
        # Handle city_id - support multiple formats:
        # 1. "citi_id_504" (full format)
        # 2. "504" (just number)  
        # 3. "21" (just number from cityId field)
        raw_city_id = data.get('city_id', data.get('cityId', data.get('cityid', '0')))
        if raw_city_id and not str(raw_city_id).startswith('citi_id_'):
            city_id = f"citi_id_{raw_city_id}"
        else:
            city_id = raw_city_id if raw_city_id else 'citi_id_0'
        
        # =================== PAGINATION ===================
        # Support both formats:
        # 1. page-based: page=1, page=2 (1-indexed)
        # 2. fromIndex-based: fromIndex is a 1-indexed PAGE NUMBER
        #    fromIndex: 1 → first page (items 0 to size-1)
        #    fromIndex: 2 → second page (items size to 2*size-1)
        #    Example with size=26:
        #      fromIndex=1 → items 0-25 (first 26 SKUs)
        #      fromIndex=2 → items 26-51 (next 26 SKUs)
        page_size = int(data.get('size', data.get('pagesize', data.get('pageSize', 20))))
        
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
                    import re
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
        
        # =================== PRESERVE USER'S ORIGINAL QUERY ===================
        # IMPORTANT: Store the user's ACTUAL input (before any cleaning) for corrected_query
        # This allows generate_corrected_query to show proper typo corrections
        query_info["user_original_query"] = data.get('searchterm', data.get('query', data.get('q', data.get('searchQuery', ''))))
        
        # =================== STORE PARSED FILTERS FOR RESPONSE ===================
        # Add the parsed price/EMI filters to query_info for showing in response
        query_info["parsed_filters"] = price_emi_filters
        
        # Merge parsed attributes into query_info
        if parsed_attrs:
            if "attributes" not in query_info:
                query_info["attributes"] = {}
            query_info["attributes"].update(parsed_attrs)
        
        # Add enhanced parsing metadata
        query_info["enhanced_parsing"] = {
            "city_detected": complex_query_info.get("city_id") is not None,
            "price_emi_detected": bool(price_emi_filters),
            "attributes_detected": len(parsed_attrs) > 0,
            "comparison_field": "lowest_emi" if price_emi_filters and "lowest_emi" in price_emi_filters else "mop"
        }
        
        logger.info(f"Search request: query='{query}', city_id='{city_id}', page={page}, size={page_size}, sort={sort_by}")
        logger.info(f"Query analysis: {json.dumps(query_info)}")
        
        # =================== DEALER SEARCH INTENT CLASSIFICATION ===================
        # Check if dealer search is enabled and classify intent
        # This allows routing to dealer search when user searches for stores/dealers
        dealer_search_type = data.get('search_type', None)  # Allow forcing search type
        customer_lat = data.get('Customer_lat', data.get('lat', data.get('customer_lat')))
        customer_long = data.get('Customer_long', data.get('lon', data.get('lng', data.get('customer_long'))))
        
        if is_dealer_search_enabled() and dealer_search_type != 'product':
            try:
                intent_result = _dealer_intent_classifier.classify(
                    query=query,
                    city_id=city_id.replace('citi_id_', '') if city_id else None,
                    customer_lat=float(customer_lat) if customer_lat else None,
                    customer_long=float(customer_long) if customer_long else None
                )
                
                logger.info(f"Dealer search intent: {intent_result.intent.name}, confidence: {intent_result.confidence:.2f}")
                
                # Route based on intent or forced search_type
                if dealer_search_type == 'dealer' or (intent_result.intent == SearchIntent.DEALER and intent_result.confidence > 0.6):
                    # Pure dealer search - route to dealer search endpoint
                    # Clean city_id - convert 'citi_id_0' or '0' to None (no city filter)
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
                
                elif intent_result.intent == SearchIntent.HYBRID:
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
        emi_slider_range = search_engine.get_global_emi_slider_range(
            query_info=query_info,
            city_id=city_id,
            filters=filters if filters else None
        )
        logger.info(f"Global EMI slider range: {emi_slider_range}")
        
        # Execute search
        search_result = search_engine.search(
            query_info=query_info,
            city_id=city_id,
            filters=filters if filters else None,
            page=page,
            page_size=page_size,
            emi_range=emi_range,
            sort_by=sort_by,
            from_offset=from_offset
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
        
        return jsonify(response)
    
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
            query = request.args.get('query', request.args.get('q', ''))
        
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
        
        # Get suggestions from autosuggest engine``
        engine = get_autosuggest()
        result = engine.get_suggestions(query.strip())
        
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
        page_size = int(data.get('size', data.get('pageSize', 20)))
        
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
    start_time = time.time()
    try:
        data = request.get_json() or {}
        logger.info(f"[COMPARE-SEARCH] Request: query='{data.get('query', '')}', category_id='{data.get('top_level_category_id', '')}'")

        query = data.get('query', '').strip()
        # Support both parameter names
        category_id = data.get('category_id') or data.get('top_level_category_id', '')
        category_name = data.get('category_name') or data.get('top_level_category_name', '')

        if not query:
            return jsonify({
                "data": {
                    "autoSuggest": {
                        "message": "Query is required",
                        "response": [],
                        "__typename": "AutoSuggestOutput"
                    }
                }
            }), 400

        # Build ES query for product search with category filter
        es_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "product_name^4",
                                "product_name.ngram^2",
                                "product_name.edge^3",
                                "manufacturer_desc^3",
                                "search_field^2",
                                "search_keywords^2",
                                "attribute_brand_new_value^2"
                            ],
                            "fuzziness": "AUTO",
                            "operator": "or",
                            "minimum_should_match": "50%"
                        }
                    }
                ],
                "filter": []
            }
        }

        # Add category filter if provided (top_level_category_id)
        if category_id:
            es_query["bool"]["filter"].append({
                "term": {"top_level_category_id": int(category_id) if str(category_id).isdigit() else category_id}
            })

        try:
            response = es.search(
                index=PRODUCT_INDEX_NAME,
                body={
                    "query": es_query,
                    "size": 20,
                    "_source": [
                        "modelid", "product_name", "top_level_category_id", "top_level_category_name",
                        "manufacturer_desc", "attribute_brand_new_value"
                    ],
                    "collapse": {
                        "field": "modelid"  # Deduplicate by modelid
                    }
                }
            )

            # Convert ES results to autosuggest format
            formatted_results = []
            seen_models = set()
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                model_id = str(source.get("modelid", ""))
                
                # Skip duplicates
                if model_id in seen_models:
                    continue
                seen_models.add(model_id)
                
                # Get brand name
                brand = source.get("attribute_brand_new_value") or source.get("manufacturer_desc", "")
                product_name = source.get("product_name", "")
                
                formatted_results.append({
                    "modelid": model_id,
                    "Product_Name": product_name,
                    "Brand": brand,
                    "top_level_category_id": str(source.get("top_level_category_id", category_id)),
                    "top_level_category_name": source.get("top_level_category_name", category_name)
                })

            logger.info(f"[COMPARE-SEARCH] Found {len(formatted_results)} results for query='{query}'")

        except Exception as e:
            logger.error(f"[COMPARE-SEARCH] ES search failed: {e}")
            formatted_results = []

        # Format response according to spec
        response_data = {
            "data": {
                "autoSuggest": {
                    "message": "no autosuggest issue" if formatted_results else "no results found",
                    "response": formatted_results,
                    "__typename": "AutoSuggestOutput"
                }
            }
        }

        # Log KPIs
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[COMPARE-SEARCH-KPI] query='{query}', category_id='{category_id}', results={len(formatted_results)}, time_ms={response_time_ms}")

        return jsonify(response_data), 200

    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[COMPARE-SEARCH-ERROR] error='{str(e)}', time_ms={response_time_ms}")
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
    # Pre-initialize autosuggest at startup
    get_autosuggest()
    app.run(host='0.0.0.0', port=8007, debug=False)
