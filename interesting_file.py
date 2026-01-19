# Mysterious Dataset from the Archives of Dr. Elena Voss
# Recovered from research station Theta-7, dated 2031-2047
# Classification: ANOMALOUS

STATION_LOG = [
    {"day": 1, "temp_c": 12.0, "pressure_hpa": 1013, "crew": 8, "incidents": 0, "notes": "Arrived safely"},
    {"day": 2, "temp_c": 11.5, "pressure_hpa": 1012, "crew": 8, "incidents": 0, "notes": "Setup complete"},
    {"day": 3, "temp_c": 11.0, "pressure_hpa": 1010, "crew": 8, "incidents": 0, "notes": "First readings nominal"},
    {"day": 5, "temp_c": 10.0, "pressure_hpa": 1008, "crew": 8, "incidents": 1, "notes": "Minor equipment fault"},
    {"day": 8, "temp_c": 8.5, "pressure_hpa": 1005, "crew": 8, "incidents": 0, "notes": "Readings continue"},
    {"day": 13, "temp_c": 6.0, "pressure_hpa": 1001, "crew": 7, "incidents": 2, "notes": "Dr. Marsh reassigned"},
    {"day": 21, "temp_c": 2.5, "pressure_hpa": 995, "crew": 7, "incidents": 1, "notes": "Unusual sounds reported"},
    {"day": 34, "temp_c": -2.0, "pressure_hpa": 987, "crew": 6, "incidents": 3, "notes": "Equipment relocated"},
    {"day": 55, "temp_c": -8.5, "pressure_hpa": 976, "crew": 5, "incidents": 2, "notes": "Reduced operations"},
    {"day": 89, "temp_c": -17.0, "pressure_hpa": 962, "crew": 4, "incidents": 5, "notes": "Emergency protocol beta"},
    {"day": 144, "temp_c": -28.5, "pressure_hpa": 944, "crew": 3, "incidents": 8, "notes": "Request evacuation"},
    {"day": 233, "temp_c": -43.0, "pressure_hpa": 921, "crew": 2, "incidents": 13, "notes": "Signal lost briefly"},
    {"day": 377, "temp_c": -61.5, "pressure_hpa": 892, "crew": 1, "incidents": 21, "notes": "Final entry"},
]

RECOVERED_TRANSMISSIONS = """
D001: ALL SYSTEMS GREEN - VOSS
D003: THE ICE SINGS AT NIGHT BUT WE IGNORE IT - MARSH
D008: PRESSURE DROPPING FASTER THAN MODELS PREDICT - CHEN
D013: SEVEN IS A BETTER NUMBER FOR THIS WORK - VOSS
D021: THEY MOVE BENEATH US - KUMAR (RETRACTED)
D034: MATHEMATICAL HARMONY IN THE READINGS - VOSS
D055: THE SEQUENCE REVEALS ITSELF TO THOSE WHO COUNT - UNKNOWN
D089: FOUR CORNERS FOUR SOULS FOUR CHANCES - VOSS
D144: THREE REMAIN THREE PATHS THREE CHOICES - VOSS
D233: WE DIVIDED WRONG - CHEN (LAST TRANSMISSION)
D377: I UNDERSTAND NOW. THE RATIO IS EVERYWHERE. GOLDEN. - VOSS
"""

ARTIFACT_MEASUREMENTS = {
    "alpha": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
    "beta": [2, 6, 18, 54, 162, 486, 1458],
    "gamma": [1, 4, 9, 16, 25, 36, 49, 64, 81],
    "delta": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
    "epsilon": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
}

FINAL_COORDINATES = [
    (61.8, 38.2), (38.2, 23.6), (23.6, 14.6),
    (14.6, 9.0), (9.0, 5.6), (5.6, 3.4),
]

VOSS_PERSONAL_CIPHER = "GURER VF AB QNATRE. GUR CNGGREA VF YVSR."

# What happened at Station Theta-7?
# Why does the data follow these patterns?
# What was Dr. Voss trying to tell us?
