R_M = 1000.0
EARTH_R = 6371000.0
DELTA_LAT = R_M / 111000.0


EMBEDDED_PATH = "dbfs:/vibebnb/data/europe_countries_embedded"          
LSH_MODEL_PATH = "dbfs:/vibebnb/models/lsh_global"
FULL_PATH = "dbfs:/vibebnb/data/europe_countries_scored.parquet"

continents = {
    "africa": [
        "AO","BF","BI","BJ","BW","CD","CF","CG","CI","CM","CV","DJ","DZ","EG","EH","ER","ES","ET",
        "GA","GH","GM","GN","GQ","GW","IL","KE","KM","LR","LS","LY","MA","MG","ML","MR","MU","MW",
        "MZ","NA","NE","NG","RE","RW","SC","SD","SH","SL","SN","SO","SS","ST","SZ","TD","TF","TG",
        "TN","TZ","UG","YT","ZA","ZM","ZW"
    ],

    "antarctica": [
        "AQ","AR","AU","CL","FK","GS","NZ","TF","ZA"
    ],

    "asia": [
        "AE","AF","AM","AZ","BD","BH","BN","BT","CN","EG","GE","HK","ID","IL","IN","IQ","IR","JO",
        "JP","KG","KH","KP","KR","KW","KZ","LA","LB","LK","MM","MN","MO","MP","MV","MY","NP","OM",
        "PH","PK","PS","QA","RU","SA","SG","SY","TH","TJ","TL","TM","TR","TW","UA","UZ","VN","YE"
    ],

    "australia_oceania": [
        "AS","AU","CC","CK","CL","CX","FJ","FM","GU","ID","KI","MH","MP","MX","NC","NF","NR","NU",
        "NZ","PF","PG","PN","PW","SB","TF","TK","TO","TV","VU","WF","WS"
    ],

    "central_america": [
        "AG","AI","AW","BB","BL","BQ","BS","BZ","CO","CR","CU","CW","DM","DO","GD","GP","GT","HN",
        "HT","JM","KN","KY","LC","MF","MQ","MS","MX","NI","PA","PR","SV","SX","TC","TT","VC","VE",
        "VG","VI"
    ],

    "europe": [
        "AD","AL","AM","AT","AX","AZ","BA","BE","BG","BY","CH","CY","CZ","DE","DK","DZ","EE","ES",
        "FI","FO","FR","GB","GE","GG","GI","GR","HR","HU","IE","IM","IQ","IR","IS","IT","JE","LI",
        "LT","LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RS","RU","SE","SI","SJ",
        "SK","SM","SY","TN","TR","UA","VA","XK"
    ],

    "north_america": [
        "BM","CA","GL","GT","IS","MX","PM","RU","SJ","US"
    ],

    "south_america": [
        "AR","AW","BO","BQ","BR","CL","CO","CW","EC","FK","GF","GS","GY","MS","PA","PE","PY","SR",
        "TT","UY","VE"
    ]
}

TEXT_COLS_DEFAULT = ["listing_title", "amenities_text","room_type_text","description"]


SCORED_NUM_COLS = [
    "price_score",
    "property_quality",
    "host_quality",
    "n_beds",
    "n_baths",
    "n_bedrooms", 
]
ENV_GROUPS =["Sightseeing",
    "Culture",
    "Family",
    "Nightlife",
    "Food",
    "Nature",
    "Transport",
    "Leisure",
    "Shopping",
    "Supplies",
    "Services",
    "Health"]
ENV_COLS = [
    "env_culture",
    "env_family",
    "env_food",
    "env_health",
    "env_leisure",
    "env_nature",
    "env_nightlife",
    "env_services",
    "env_shopping",
    "env_sightseeing",
    "env_supplies",
    "env_transport",
]

