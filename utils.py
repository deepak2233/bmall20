# utils.py
# BajajMall Search API Utility Constants & Helpers

# =================== CATEGORY CANONICAL MAPPING ===================
CATEGORY_CANONICAL = {
    "mobile phones": "smartphone",
    "mobiles": "smartphone",
    "smartphone": "smartphone",
    "phone": "smartphone",
    "phones": "smartphone",
    "cellphone": "smartphone",
    "laptop": "laptops",
    "laptops": "laptops",
    "notebook": "laptops",
    "ultrabook": "laptops",
    "television": "tv and home entertainment",
    "tv": "tv and home entertainment",
    "smart tv": "tv and home entertainment",
    "led tv": "tv and home entertainment",
    "refrigerator": "refrigerators",
    "fridge": "refrigerators",
    "refrigerators": "refrigerators",
    "freezer": "refrigerators",
    "washing machine": "washing machines",
    "washer": "washing machines",
    "washing machines": "washing machines",
    "ac": "ac",
    "air conditioner": "ac",
    "split ac": "ac",
    "window ac": "ac",
    "air cooler": "air coolers",
    "cooler": "air coolers",
    "air coolers": "air coolers",
    "two wheeler": "two-wheeler",
    "two-wheeler": "two-wheeler",
    "bike": "two-wheeler",
    "scooter": "two-wheeler",
    "motorcycle": "two-wheeler",
    "car": "new cars",
    "cars": "new cars",
    "new cars": "new cars",
    "four wheeler": "new cars",
    "tractor": "tractor",
    "tractors": "tractor",
    "water purifier": "water purifier",
    "purifier": "water purifier",
    "microwave": "microwave",
    "oven": "microwave",
    "chimney": "chimney",
    "kitchen chimney": "chimney",
    "dishwasher": "dishwasher",
    "geyser": "water heater",
    "water heater": "water heater",
    "vacuum cleaner": "vacuum cleaner",
    "vacuum": "vacuum cleaner",
    "fan": "fans",
    "ceiling fan": "fans",
    "table fan": "fans",
    "smartwatch": "smartwatch",
    "smart watch": "smartwatch",
    "watch": "smartwatch",
    "tablet": "tablets",
    "tablets": "tablets",
    "ipad": "tablets",
    "camera": "camera",
    "dslr": "camera",
    "printer": "printers",
    "printers": "printers",
    "monitor": "desktop monitor",
    "desktop monitor": "desktop monitor",
    "speaker": "audio & video",
    "speakers": "audio & video",
    "headphone": "audio & video",
    "headphones": "audio & video",
    "earphone": "audio & video",
    "earbuds": "audio & video",
    "furniture": "furniture",
    "sofa": "furniture",
    "bed": "furniture",
    "mattress": "furniture",
}

# =================== BUSINESS SYNONYMS ===================
BUSINESS_SYNONYMS = {
    "mobiles": ["mobile phones", "mobiles", "smartphone", "phone", "phones", "cellphone", "cell phone", "handset", "android phone", "ios phone"],
    "laptops": ["laptop", "laptops", "notebook", "ultrabook", "chromebook", "macbook", "gaming laptop"],
    "tv and home entertainment": ["television", "tv", "tvs", "led tv", "smart tv", "oled tv", "4k tv", "home theater"],
    "refrigerators": ["refrigerator", "fridge", "freezer", "double door", "single door", "side by side"],
    "washing machines": ["washing machine", "washer", "front load", "top load", "semi automatic", "fully automatic"],
    "ac": ["ac", "air conditioner", "airconditioner", "split ac", "window ac", "inverter ac"],
    "air coolers": ["air cooler", "cooler", "desert cooler", "tower cooler", "personal cooler"],
    "two-wheeler": ["two wheeler", "bike", "motorcycle", "scooter", "scooty", "activa", "bullet", "pulsar", "splendor"],
    "new cars": ["car", "cars", "four wheeler", "sedan", "hatchback", "suv", "muv"],
    "tractor": ["tractor", "tractors", "farm tractor", "agriculture"],
    "water purifier": ["water purifier", "ro", "uv purifier", "aquaguard"],
    "microwave": ["microwave", "oven", "otg", "microwave oven", "convection"],
    "chimney": ["chimney", "kitchen chimney", "auto clean chimney"],
    "dishwasher": ["dishwasher", "dish washer"],
    "water heater": ["geyser", "water heater", "instant geyser", "storage geyser"],
    "vacuum cleaner": ["vacuum cleaner", "vacuum", "robotic vacuum"],
    "fans": ["fan", "ceiling fan", "table fan", "pedestal fan", "exhaust fan"],
    "smartwatch": ["smartwatch", "smart watch", "fitness band", "fitness tracker"],
    "tablets": ["tablet", "tablets", "ipad", "tab"],
    "camera": ["camera", "dslr", "mirrorless", "digital camera"],
    "printers": ["printer", "printers", "inkjet", "laser printer"],
    "desktop monitor": ["monitor", "desktop monitor", "led monitor", "gaming monitor"],
    "audio & video": ["speaker", "speakers", "headphone", "headphones", "earphone", "earbuds", "soundbar", "home theater"],
    "furniture": ["furniture", "sofa", "bed", "mattress", "table", "chair", "wardrobe"],
    "apple": ["apple", "iphone", "iphones", "apple phone", "ios", "macbook", "ipad", "airpods"],
}

# =================== TWO WHEELER AUTOSUGGEST ===================
TWOWHEELER_AUTOSUGGEST = [
    "hero splendor", "honda activa", "bajaj pulsar", "royal enfield", "tvs jupiter",
    "yamaha fz", "suzuki access", "hero passion", "bajaj avenger", "tvs ntorq",
    "honda shine", "hero glamour", "bajaj dominar", "ktm duke", "tvs apache",
    "ola electric", "ather 450", "hero electric", "okinawa", "revolt rv400",
    "vespa", "aprilia", "hero destini", "honda dio", "suzuki burgman",
]

# =================== SCOOTER SYNONYMS ===================
SCOOTER_SYNONYMS = [
    "scooter", "scooty", "activa", "jupiter", "ntorq", "access", "dio", "pleasure",
    "destini", "maestro", "fascino", "ray", "grazia", "burgman", "vespa", "aprilia",
]

# =================== APPLE TERMS ===================
APPLE_TERMS = set([
    "apple", "iphone", "iphones", "ios", "macbook", "ipad", "airpods", "apple watch",
    "imac", "mac mini", "mac studio", "apple tv", "homepod", "airtag",
])

# =================== CATEGORY HARDCODED CHIPS ===================
# Format: category -> (chips_list, filter_text)
CATEGORY_HARDCODED_CHIPS = {
    "smartphone": (["64 GB", "128 GB", "256 GB", "512 GB"], "attribute_internal_storage"),
    "mobile phones": (["64 GB", "128 GB", "256 GB", "512 GB"], "attribute_internal_storage"),
    "mobiles": (["64 GB", "128 GB", "256 GB", "512 GB"], "attribute_internal_storage"),
    "laptops": (["256 GB SSD", "512 GB SSD", "1 TB", "8 GB RAM", "16 GB RAM"], "attribute_storage_size"),
    "tv and home entertainment": (["32 inch", "43 inch", "50 inch", "55 inch", "65 inch"], "attribute_screen_size_in_inches"),
    "refrigerators": (["180 L", "260 L", "300 L", "400 L", "500 L"], "attribute_capacity_litres"),
    "washing machines": (["6 kg", "7 kg", "8 kg", "9 kg", "10 kg"], "attribute_capacity_wm"),
    "ac": (["1 Ton", "1.5 Ton", "2 Ton"], "attribute_capacity_in_tons"),
    "air coolers": (["20 L", "35 L", "50 L", "70 L"], "attribute_capacity_air_cooler"),
    "two-wheeler": (["100 cc", "125 cc", "150 cc", "200 cc", "350 cc"], "attribute_engine_capacity_new"),
    "new cars": (["Hatchback", "Sedan", "SUV", "MUV"], "attribute_body_type"),
    "tractor": (["30 HP", "40 HP", "50 HP", "60 HP"], "attribute_engine_capacity"),
    "tablets": (["64 GB", "128 GB", "256 GB"], "attribute_internal_storage"),
    "smartwatch": (["Fitness", "Premium", "Kids"], "attribute_type"),
}

# =================== FILTER ATTRIBUTE EXCLUSIONS ===================
# Attributes to exclude from filter display for each category
FILTER_ATTRIBUTE_EXCLUSIONS = {
    "smartphone": set(["attribute_product_id", "attribute_set_id", "attribute_brand_id"]),
    "laptops": set(["attribute_product_id", "attribute_set_id"]),
    "tv and home entertainment": set(["attribute_product_id"]),
    "refrigerators": set(["attribute_product_id"]),
    "washing machines": set(["attribute_product_id"]),
    "ac": set(["attribute_product_id"]),
    "two-wheeler": set(["attribute_product_id"]),
    "new cars": set(["attribute_product_id"]),
    "blank": set(),
}

# =================== BUSINESS AUTOSUGGEST ===================
# Keyword -> list of suggestions
BUSINESS_AUTOSUGGEST = {
    "mobile": ["Samsung Galaxy", "iPhone", "OnePlus", "Vivo", "Oppo", "Realme", "Redmi"],
    "smartphone": ["Samsung Galaxy", "iPhone", "OnePlus", "Vivo", "Oppo", "Realme", "Redmi"],
    "phone": ["Samsung Galaxy", "iPhone", "OnePlus", "Vivo", "Oppo", "Realme", "Redmi"],
    "laptop": ["HP Laptop", "Dell Laptop", "Lenovo Laptop", "Asus Laptop", "MacBook"],
    "tv": ["Samsung TV", "LG TV", "Sony TV", "Mi TV", "OnePlus TV"],
    "refrigerator": ["Samsung Refrigerator", "LG Refrigerator", "Whirlpool Refrigerator", "Godrej Refrigerator"],
    "fridge": ["Samsung Refrigerator", "LG Refrigerator", "Whirlpool Refrigerator", "Godrej Refrigerator"],
    "washing machine": ["Samsung Washing Machine", "LG Washing Machine", "Whirlpool Washing Machine", "IFB Washing Machine"],
    "ac": ["Voltas AC", "Daikin AC", "LG AC", "Samsung AC", "Blue Star AC"],
    "air conditioner": ["Voltas AC", "Daikin AC", "LG AC", "Samsung AC", "Blue Star AC"],
    "bike": ["Hero Splendor", "Honda Shine", "Bajaj Pulsar", "TVS Apache", "Royal Enfield"],
    "scooter": ["Honda Activa", "TVS Jupiter", "Suzuki Access", "Hero Destini", "Ola Electric"],
    "car": ["Maruti Swift", "Hyundai i20", "Tata Nexon", "Mahindra XUV700", "Kia Seltos"],
}

# =================== KNOWN BRANDS ===================
KNOWN_BRANDS = [
    # Mobile brands
    "samsung", "apple", "oppo", "vivo", "realme", "redmi", "xiaomi", "oneplus", "nothing",
    "motorola", "nokia", "google", "iqoo", "infinix", "tecno", "poco", "lava", "micromax",
    # Appliance brands
    "lg", "panasonic", "sony", "whirlpool", "godrej", "haier", "bosch", "ifb", "voltas",
    "daikin", "blue star", "carrier", "hitachi", "lloyd", "kent", "aquaguard", "eureka forbes",
    # Laptop brands
    "hp", "dell", "lenovo", "asus", "acer", "msi",
    # TV brands
    "mi", "tcl", "vu", "thomson",
    # Two-wheeler brands
    "hero", "honda", "bajaj", "tvs", "royal enfield", "yamaha", "suzuki", "ktm",
    "ather", "ola", "revolt", "okinawa",
    # Car brands
    "maruti", "hyundai", "tata", "mahindra", "kia", "toyota", "skoda", "volkswagen",
    # Audio brands
    "jbl", "bose", "sennheiser", "boat", "sony",
    # Kitchen brands
    "prestige", "pigeon", "philips", "morphy richards", "wonderchef", "havells", "crompton",
    "usha", "orient", "bajaj",
]

# =================== COMMON PRODUCT TERMS ===================
# Terms that should be included in fuzzy matching pool
COMMON_PRODUCT_TERMS = [
    # Storage
    "64gb", "128gb", "256gb", "512gb", "1tb", "2tb",
    "64 gb", "128 gb", "256 gb", "512 gb", "1 tb", "2 tb",
    # RAM
    "4gb ram", "6gb ram", "8gb ram", "12gb ram", "16gb ram",
    # Screen sizes
    "32 inch", "43 inch", "50 inch", "55 inch", "65 inch", "75 inch",
    # AC capacity
    "1 ton", "1.5 ton", "2 ton",
    # Washing machine capacity
    "6 kg", "7 kg", "8 kg", "9 kg", "10 kg",
    # Fridge capacity
    "180 litre", "260 litre", "300 litre", "400 litre",
    # Display types
    "amoled", "oled", "led", "qled", "4k", "full hd", "hd",
    # 5G
    "5g", "4g",
    # Colors
    "black", "white", "blue", "red", "gold", "silver", "grey", "green", "pink", "purple",
    # Variants
    "pro", "max", "plus", "mini", "ultra", "lite", "air",
]
