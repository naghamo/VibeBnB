R_M = 1000.0
EARTH_R = 6371000.0
DELTA_LAT = R_M / 111000.0


continents = {
    "Africa": [
        "South Africa","Morocco","Egypt","Kenya","Nigeria","Ghana","Senegal","Tunisia",
        "Algeria","Ethiopia","Uganda","Tanzania","Rwanda","Zimbabwe","Cameroon","Namibia",
        "Botswana","Zambia","Malawi","Lesotho","Liberia","Sierra Leone","Gambia","Sudan",
        "South Sudan","Niger","Chad","Congo","Democratic Republic of the Congo",
        "Burkina Faso","Benin","Togo","Guinea","Guinea-Bissau","Gabon","Mali",
        "Central African Republic","Libya","Somalia","Djibouti","Equatorial Guinea",
        "Mauritius","Seychelles","Cabo Verde","São Tomé & Príncipe","Mayotte"
    ],

    "Asia": [
        "India","Thailand","South Korea","Japan","Turkey","Vietnam","Malaysia",
        "Philippines","Sri Lanka","Pakistan","Nepal","Bangladesh","Indonesia",
        "Cambodia","Laos","Myanmar","Afghanistan","Hong Kong","Taiwan","Singapore",
        "Mongolia","Kazakhstan","Uzbekistan","Kyrgyzstan","Tajikistan","Armenia",
        "Azerbaijan","Georgia","Russia","Israel","Saudi Arabia",
        "United Arab Emirates","Jordan","Lebanon","Iraq","Kuwait","Qatar","Bahrain",
        "Oman","Palestinian Territories","Timor-Leste"
    ],

    "Europe": [
        "France","Italy","Spain","United Kingdom","Germany","Greece","Croatia",
        "Portugal","Poland","Norway","Sweden","Denmark","Netherlands","Switzerland",
        "Austria","Belgium","Ireland","Romania","Czechia","Finland","Hungary",
        "Slovakia","Slovenia","Bulgaria","Serbia","Ukraine","Latvia","Lithuania",
        "Estonia","Montenegro","Albania","Bosnia & Herzegovina","North Macedonia",
        "Malta","Luxembourg","Iceland","Andorra","San Marino","Monaco","Kosovo",
        "Belarus","Moldova","Liechtenstein","Cyprus"
    ],

    "North America": [
        "United States","Canada","Mexico","Costa Rica","Dominican Republic",
        "Guatemala","Panama","El Salvador","Honduras","Nicaragua","Cuba","Jamaica",
        "Haiti","Bahamas","Barbados","Trinidad & Tobago","Belize","Grenada","Dominica",
        "St Lucia","St Vincent & Grenadines","St Kitts & Nevis","Antigua & Barbuda",
        "Puerto Rico","US Virgin Islands","British Virgin Islands","Cayman Islands",
        "Turks & Caicos Islands","Bermuda","Greenland"
    ],

    "South America": [
        "Brazil","Colombia","Argentina","Chile","Peru","Ecuador","Uruguay",
        "Bolivia","Venezuela","Paraguay","Suriname","Guyana","French Guiana"
    ],

    "Oceania": [
        "Australia","New Zealand","Fiji","Samoa","Tonga","Vanuatu","Solomon Islands",
        "Micronesia","Kiribati","Tuvalu","Niue","Palau","Cook Islands","Norfolk Island",
        "New Caledonia","French Polynesia","Wallis & Futuna","Christmas Island"
    ]
}
