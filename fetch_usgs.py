import os
import json
import random
import requests
from utils import haversine_distance, safe_float, safe_int

def fetch_usgs_event_and_dyfi(event_id="ci38457511"):
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # 1. Fetch event detail to get epicenter and product list
    event_url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={event_id}&format=geojson"
    print(f"Querying USGS API for event details: {event_url}")
    
    try:
        response = requests.get(event_url, timeout=15)
        response.raise_for_status()
        event_data = response.json()
    except Exception as e:
        print(f"Error querying event catalog: {e}")
        return None
        
    properties = event_data.get("properties", {})
    mag = properties.get("mag")
    title = properties.get("title")
    time_ms = properties.get("time")
    
    geom = event_data.get("geometry", {})
    coords = geom.get("coordinates", [])
    if len(coords) < 3:
        print("Error: Incomplete coordinates in event metadata.")
        return None
        
    epicenter_lon = coords[0]
    epicenter_lat = coords[1]
    depth = coords[2]
    
    print(f"Event: {title} | Mag: {mag} | Epicenter: ({epicenter_lat}, {epicenter_lon}) | Depth: {depth} km")
    
    # Save event metadata
    meta = {
        "event_id": event_id,
        "title": title,
        "magnitude": mag,
        "latitude": epicenter_lat,
        "longitude": epicenter_lon,
        "depth": depth,
        "time_ms": time_ms
    }
    with open("data/event_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
        
    # 2. Find cdi_zip.txt URL in the product list
    products = properties.get("products", {})
    dyfi_list = products.get("dyfi", [])
    if not dyfi_list:
        print("Error: No DYFI products found in this event.")
        return None
        
    contents = dyfi_list[0].get("contents", {})
    dyfi_txt_url = None
    for key, item in contents.items():
        if "cdi_zip.txt" in key:
            dyfi_txt_url = item.get("url")
            break
            
    if not dyfi_txt_url:
        print("Error: Could not find 'cdi_zip.txt' in DYFI product list.")
        return None
        
    print(f"Downloading ground-truth MMI reports from: {dyfi_txt_url}")
    try:
        res = requests.get(dyfi_txt_url, timeout=15)
        res.raise_for_status()
        dyfi_txt_data = res.text
    except Exception as e:
        print(f"Error downloading cdi_zip.txt: {e}")
        return None
        
    # 3. Parse cdi_zip.txt
    # Format is usually tab-separated or comma-separated:
    # ZIP, CDI, responses, latitude, longitude (or similar headers)
    lines = dyfi_txt_data.strip().split("\n")
    if not lines:
        print("Error: Empty DYFI data downloaded.")
        return None
        
    header = [h.strip().lower() for h in lines[0].split("\t")]
    if len(header) <= 1:
        # Try comma-separated
        header = [h.strip().lower() for h in lines[0].split(",")]
        separator = ","
    else:
        separator = "\t"
        
    print(f"Parsed headers: {header}")
    
    # Map headers to indices
    zip_idx = -1
    cdi_idx = -1
    resp_idx = -1
    lat_idx = -1
    lon_idx = -1
    state_idx = -1
    
    for i, h in enumerate(header):
        if "zip" in h or "location" in h:
            zip_idx = i
        elif "cdi" in h or "mmi" in h:
            cdi_idx = i
        elif "response" in h or "no. of" in h or "nresp" in h:
            resp_idx = i
        elif "latitude" in h or "lat" in h:
            lat_idx = i
        elif "longitude" in h or "lon" in h:
            lon_idx = i
        elif "state" in h:
            state_idx = i
            
    if zip_idx == -1 or cdi_idx == -1:
        print(f"Error parsing headers: Could not identify required columns (zip, cdi) in {header}.")
        return None


    max_required_idx = max(zip_idx, cdi_idx, resp_idx, lat_idx, lon_idx, state_idx)
    zctas = []
    for line in lines[1:]:
        parts = line.strip().split(separator)
        if len(parts) <= max_required_idx:
            continue
            
        zip_code = parts[zip_idx].strip().strip('"').strip("'")
        # Clean zip code (pad with leading zeros to make 5 digits)
        if len(zip_code) < 5 and zip_code.isdigit():
            zip_code = zip_code.zfill(5)
            
        cdi = safe_float(parts[cdi_idx])
        num_resp = safe_int(parts[resp_idx]) if resp_idx != -1 else 1
        z_lat = safe_float(parts[lat_idx]) if lat_idx != -1 else 0.0
        z_lon = safe_float(parts[lon_idx]) if lon_idx != -1 else 0.0
        state_val = parts[state_idx].strip().strip('"').strip("'") if state_idx != -1 else "CA"
        
        # Calculate distance to epicenter
        dist = haversine_distance(epicenter_lat, epicenter_lon, z_lat, z_lon)
        
        zctas.append({
            "zip": zip_code,
            "actual_mmi": cdi,
            "responses": num_resp,
            "latitude": z_lat,
            "longitude": z_lon,
            "distance_km": round(dist, 2),
            "state": state_val
        })
        
    print(f"Total reporting ZIP codes: {len(zctas)}")
    
    # 4. Filter ZIP codes: within 200km of epicenter & at least 10 responses
    filtered_zctas = [
        z for z in zctas 
        if z["distance_km"] <= 200.0 and z["responses"] >= 10 and z["latitude"] != 0.0
    ]
    
    print(f"Reporting ZIP codes within 200km with >=10 responses: {len(filtered_zctas)}")
    
    if len(filtered_zctas) < 50:
        print(f"Warning: Only {len(filtered_zctas)} ZIP codes meet the filter criteria. Proceeding with all of them.")
        sampled_zctas = filtered_zctas
    else:
        # Deterministic sampling for academic reproducibility
        random.seed(42)
        sampled_zctas = random.sample(filtered_zctas, 50)
        
    print(f"Sampled {len(sampled_zctas)} ZIP codes for the study.")
    
    # Save outputs
    with open("data/sampled_zctas.json", "w", encoding="utf-8") as f:
        json.dump(sampled_zctas, f, indent=4)
        
    return sampled_zctas

if __name__ == "__main__":
    fetch_usgs_event_and_dyfi()
