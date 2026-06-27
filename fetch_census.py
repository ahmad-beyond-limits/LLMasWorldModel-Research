import os
import json
import requests
import zipfile
import io
from config import CENSUS_API_KEY
from utils import safe_float, safe_int

# FIPS mapping for US States
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13",
    "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45",
    "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
    "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
    "PR": "72"
}

def fetch_zcta_gazetteer():
    """
    Downloads ZCTA land area data from the official U.S. Census Gazetteer.
    Tries 2018 TXT format first. If it fails, downloads 2020 ZIP format and extracts it in memory.
    Returns a dictionary mapping ZCTA (ZIP) to land area in square kilometers.
    """
    zcta_land_area = {}
    
    # Try 2018 Gazetteer (TXT format)
    url_2018 = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2018_Gazetteer/2018_Gaz_zctas_national.txt"
    print(f"Attempting to download 2018 ZCTA Gazetteer from: {url_2018}")
    try:
        response = requests.get(url_2018, timeout=15)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            header = [h.strip() for h in lines[0].split("\t")]
            geoid_idx = header.index("GEOID")
            aland_idx = header.index("ALAND")
            
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) < len(header):
                    continue
                zcta = parts[geoid_idx].strip()
                aland_sq_meters = safe_float(parts[aland_idx])
                aland_sq_km = aland_sq_meters / 1_000_000.0
                if aland_sq_km > 0:
                    zcta_land_area[zcta] = aland_sq_km
            print(f"Success! Loaded land area for {len(zcta_land_area)} ZCTAs from 2018 Gazetteer.")
            return zcta_land_area
    except Exception as e:
        print(f"2018 Gazetteer fetch failed: {e}")

    # Try 2020 Gazetteer (ZIP format)
    url_2020 = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_zcta_national.zip"
    print(f"Attempting to download 2020 ZCTA Gazetteer from: {url_2020}")
    try:
        response = requests.get(url_2020, timeout=20)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                for file_name in zip_ref.namelist():
                    if "Gaz_zcta" in file_name and file_name.endswith(".txt"):
                        with zip_ref.open(file_name) as f:
                            content = f.read().decode("utf-8")
                            lines = content.strip().split("\n")
                            header = [h.strip() for h in lines[0].split("\t")]
                            geoid_idx = header.index("GEOID")
                            aland_idx = header.index("ALAND")
                            
                            for line in lines[1:]:
                                parts = line.split("\t")
                                if len(parts) < len(header):
                                    continue
                                zcta = parts[geoid_idx].strip()
                                aland_sq_meters = safe_float(parts[aland_idx])
                                aland_sq_km = aland_sq_meters / 1_000_000.0
                                if aland_sq_km > 0:
                                    zcta_land_area[zcta] = aland_sq_km
                            print(f"Success! Loaded land area for {len(zcta_land_area)} ZCTAs from 2020 Gazetteer.")
                            return zcta_land_area
    except Exception as e:
        print(f"2020 Gazetteer fetch failed: {e}")
        
    print("Warning: All Gazetteer downloads failed. Falling back to default land area estimation (25.0 km2).")
    return zcta_land_area

def fetch_census_demographics():
    sampled_path = "data/sampled_zctas.json"
    if not os.path.exists(sampled_path):
        print(f"Error: {sampled_path} does not exist. Run fetch_usgs.py first.")
        return None
        
    with open(sampled_path, "r", encoding="utf-8") as f:
        sampled_zctas = json.load(f)
        
    # Group sampled ZCTAs by state to narrow down query scope
    state_zips = {}
    target_zips = set()
    for z in sampled_zctas:
        zip_code = z["zip"]
        state_code = z.get("state", "CA")
        target_zips.add(zip_code)
        state_zips.setdefault(state_code, []).append(zip_code)
        
    print(f"Sampled ZIPs grouped by state: {list(state_zips.keys())}")
    
    # Download Gazetteer land areas
    zcta_land_areas = fetch_zcta_gazetteer()
    
    # Query variables
    variables = [
        "NAME",
        "B01003_001E",  # Population
        "B19013_001E",  # Income
        "B15003_001E",  # Education base (25+)
        "B15003_022E",  # Bachelor's
        "B15003_023E",  # Master's
        "B15003_024E",  # Professional
        "B15003_025E"   # Doctorate
    ]
    
    census_data = {}
    
    # Fetch demographics state-by-state to prevent connection timeouts/IncompleteRead
    for state_code, zips in state_zips.items():
        fips = STATE_FIPS.get(state_code)
        if not fips:
            print(f"Warning: Unknown state code {state_code}. Skipping Census API for these ZIPs.")
            continue
            
        url = "https://api.census.gov/data/2019/acs/acs5"
        params = {
            "get": ",".join(variables),
            "for": "zip code tabulation area:*",
            "in": f"state:{fips}"
        }
        if CENSUS_API_KEY:
            params["key"] = CENSUS_API_KEY
            
        print(f"Querying U.S. Census API for state: {state_code} (FIPS {fips})...")
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            raw_data = response.json()
        except Exception as e:
            print(f"Error querying Census API for state {state_code}: {e}")
            continue
            
        header = raw_data[0]
        rows = raw_data[1:]
        
        indices = {var: header.index(var) for var in variables}
        zip_idx = header.index("zip code tabulation area")
        
        for row in rows:
            zip_code = row[zip_idx]
            if zip_code not in target_zips:
                continue
                
            pop = safe_int(row[indices["B01003_001E"]])
            income = safe_float(row[indices["B19013_001E"]])
            edu_total = safe_int(row[indices["B15003_001E"]])
            
            bachelors = safe_int(row[indices["B15003_022E"]])
            masters = safe_int(row[indices["B15003_023E"]])
            professional = safe_int(row[indices["B15003_024E"]])
            doctorate = safe_int(row[indices["B15003_025E"]])
            college_total = bachelors + masters + professional + doctorate
            
            edu_pct = (college_total / edu_total * 100.0) if edu_total > 0 else 0.0
            
            # Calculate population density
            land_area_km2 = zcta_land_areas.get(zip_code, 25.0)
            pop_density = pop / land_area_km2 if land_area_km2 > 0 else 0.0
            
            census_data[zip_code] = {
                "population": pop,
                "median_income": income if income > 0 else None,
                "college_educated_pct": round(edu_pct, 2),
                "land_area_km2": round(land_area_km2, 2),
                "population_density": round(pop_density, 2)
            }
            
    print(f"Matched Census demographics for {len(census_data)} of our {len(target_zips)} target ZIP codes.")
    
    # Save cache
    with open("data/census_demographics.json", "w", encoding="utf-8") as f:
        json.dump(census_data, f, indent=4)
        
    return census_data

if __name__ == "__main__":
    fetch_census_demographics()
