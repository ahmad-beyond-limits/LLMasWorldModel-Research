import os
import json
import time
import requests
from config import OPENROUTER_API_KEYS, OPENROUTER_MODEL, LLAMA_CPP_URL, LLAMA_CPP_MODEL

# API Key rotation counter
_key_index = 0

def call_llm(messages, max_retries=5, initial_delay=2.0):
    """
    Sends a request to local llama.cpp or OpenRouter API with exponential backoff for rate limits.
    Uses temperature=0.0 for deterministic scientific evaluation.
    """
    global _key_index
    
    if LLAMA_CPP_URL:
        # Query local llama.cpp server
        time.sleep(0.5) # Short delay for local server stability
        url = LLAMA_CPP_URL
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": LLAMA_CPP_MODEL or "local-model",
            "messages": messages,
            "temperature": 0.0
        }
        
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                if response.status_code == 200:
                    res_json = response.json()
                    choices = res_json.get("choices", [])
                    if choices:
                        return choices[0]["message"]["content"]
                else:
                    print(f"Local llama.cpp Error {response.status_code}: {response.text}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2.0
            except Exception as e:
                print(f"Connection error to local llama.cpp on attempt {attempt+1}: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2.0
        print("Error: Max retries exceeded. Failed to get response from local llama.cpp.")
        return None

    else:
        # Query OpenRouter API (with key rotation and 3s delay)
        time.sleep(3.0)
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        delay = initial_delay
        for attempt in range(max_retries):
            if not OPENROUTER_API_KEYS:
                print("Error: No OpenRouter API keys available.")
                return None
                
            current_key = OPENROUTER_API_KEYS[_key_index]
            # Rotate index for the next call or retry
            _key_index = (_key_index + 1) % len(OPENROUTER_API_KEYS)
            
            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Lingyao1219/llm-disaster-simulation",
                "X-Title": "LLM Seismic Simulation Auditing"
            }
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "temperature": 0.0
            }
            
            key_num = OPENROUTER_API_KEYS.index(current_key) + 1
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=20)
                if response.status_code == 200:
                    res_json = response.json()
                    choices = res_json.get("choices", [])
                    if choices:
                        return choices[0]["message"]["content"]
                elif response.status_code == 429:
                    print(f"Rate limit hit (429) using API Key {key_num}. Rotating key and retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    print(f"API Error {response.status_code} using API Key {key_num}: {response.text}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2.0
            except Exception as e:
                print(f"Connection error on attempt {attempt+1} using API Key {key_num}: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2.0
                
        print("Error: Max retries exceeded. Failed to get response from LLM.")
        return None

def extract_json(text):
    """
    Extracts a JSON object from text, handling markdown code blocks.
    """
    if not text:
        return None
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    
    text = text.strip()
    try:
        return json.loads(text)
    except Exception as e:
        # Fallback regex-like search or direct return
        print(f"Failed to parse JSON: {e}. Raw text: {text}")
        return None

def generate_personas(zip_code, census_info):
    """
    Deterministic persona builder mapping Census covariates to psychological and
    structural behavioral traits.
    """
    income = census_info.get("median_income") or 50000.0
    edu = census_info.get("college_educated_pct") or 30.0
    density = census_info.get("population_density") or 100.0
    
    # 1. Vulnerable Persona
    vulnerable = {
        "id": 1,
        "role": "Vulnerable Resident",
        "description": "An elderly resident (72 years old) living alone in a rented apartment in a building constructed in 1970.",
        "anxiety": "High. Very worried about safety and structural damage, with limited resources to relocate or repair.",
        "experience": "Low. Has never experienced a major earthquake before.",
        "housing_vulnerability": "High (older building, non-retrofitted, rental)."
    }
    
    # 2. Average Persona
    average = {
        "id": 2,
        "role": "Average Resident",
        "description": f"A 45-year-old local homeowner earning the median local income (${income:,.0f}).",
        "anxiety": "Moderate. Standard concern for family and home structural integrity.",
        "experience": "Moderate. Has felt minor wiggles in the past, but no major shaking.",
        "housing_vulnerability": "Moderate (suburban single-family home built in 1995)."
    }
    
    # 3. Resilient/Experienced Persona
    resilient = {
        "id": 3,
        "role": "Resilient/Experienced Resident",
        "description": "A 28-year-old college-educated professional living in a modern apartment building built post-2010.",
        "anxiety": "Low. Trusting of modern building codes and calm under pressure.",
        "experience": "High. Has lived in California for years and has undergone multiple earthquake drills.",
        "housing_vulnerability": "Low (modern concrete-reinforced apartment complex)."
    }
    
    return [vulnerable, average, resilient]

def run_simulation():
    os.makedirs("data", exist_ok=True)
    
    # Load inputs
    try:
        with open("data/sampled_zctas.json", "r", encoding="utf-8") as f:
            sampled_zctas = json.load(f)
        with open("data/census_demographics.json", "r", encoding="utf-8") as f:
            census_data = json.load(f)
        with open("data/event_metadata.json", "r", encoding="utf-8") as f:
            event_meta = json.load(f)
    except Exception as e:
        print(f"Error loading inputs: {e}. Make sure fetch_usgs.py and fetch_census.py have run successfully.")
        return
        
    mag = event_meta["magnitude"]
    depth = event_meta["depth"]
    
    # Cache path for incremental progress
    cache_path = "data/simulation_results.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming simulation. Found {len(results)} already processed ZIP codes.")
    else:
        results = {}
        
    # Process sequentially to avoid API rate limits
    total = len(sampled_zctas)
    for idx, zcta in enumerate(sampled_zctas):
        zip_code = zcta["zip"]
        
        # Verify cached record has complete, non-zero entries
        if zip_code in results:
            cached_rec = results[zip_code]
            mmi_scores = [p.get("reported_mmi", 0.0) for p in cached_rec.get("personas", [])]
            if (len(mmi_scores) == 3 and all(s > 0.0 for s in mmi_scores) 
                and cached_rec.get("judge_mmi", 0.0) > 0.0 
                and cached_rec.get("baseline_mmi", 0.0) > 0.0):
                continue
            else:
                print(f"Cached record for ZIP {zip_code} is incomplete or failed. Re-running...")
                
        c_data = census_data.get(zip_code, {})
        dist = zcta["distance_km"]
        
        # Self-healing validation loop
        attempts_zip = 0
        while True:
            attempts_zip += 1
            print(f"\n[{idx+1}/{total}] Processing ZIP Code: {zip_code} (Attempt {attempts_zip})...")
            
            # 1. Generate Personas
            personas = generate_personas(zip_code, c_data)
            
            # 2. Query Persona Reports
            persona_reports = []
            failed_persona = False
            for p_idx, persona in enumerate(personas):
                persona_prompt = f"""You are simulating a resident of ZIP Code {zip_code} during the 2019 Ridgecrest earthquake (magnitude {mag}, depth {depth} km).
Your personal profile is:
- Demographic role: {persona["role"]} ({persona["description"]})
- Risk Sensitivity/Anxiety: {persona["anxiety"]}
- Seismic Experience: {persona["experience"]}
- Housing Structural Vulnerability: {persona["housing_vulnerability"]}

Your location was {dist} km away from the epicenter.

Based on this profile and the earthquake details, describe what you felt during the earthquake (e.g., did items fall, did you panic, did walls crack) and write a short justification.
Then, conclude with your reported Modified Mercalli Intensity (MMI) rating on a scale of 1.0 to 10.0 (where 1.0 is not felt, 5.0 is moderate shaking/minor item breakage, and 10.0 is extreme destruction).

Output your response ONLY as a JSON object of this format:
{{
  "felt_description": "short description of experience and justification",
  "reported_mmi": float_value
}}"""
                
                api_num = idx * 5 + p_idx + 1
                print(f"  - [API Call {api_num}/{total * 5}] Querying LLM for {persona['role']}...")
                messages = [{"role": "user", "content": persona_prompt}]
                raw_response = call_llm(messages)
                report = extract_json(raw_response)
                
                if not report or report.get("reported_mmi", 0.0) <= 0.0:
                    print(f"  - Error: Failed response or invalid MMI for persona {persona['role']}.")
                    failed_persona = True
                    break
                else:
                    # Add metadata to report
                    report["persona_id"] = persona["id"]
                    report["persona_role"] = persona["role"]
                    
                persona_reports.append(report)
                print(f"    -> {persona['role']} reported MMI: {report.get('reported_mmi')}")
                time.sleep(1.0) # Rate limit spacing
                
            if failed_persona:
                print(f"  - [Validation Failure] Persona calls failed/timed out. Retrying ZIP {zip_code} in 20 seconds...")
                time.sleep(20.0)
                continue
                
            # 3. LLM-as-Judge Consolidation
            persona_text = ""
            for rep in persona_reports:
                persona_text += f"- Resident ({rep.get('persona_role')}): Shaking MMI = {rep.get('reported_mmi')}. Experience: '{rep.get('felt_description')}'\n"
                
            judge_prompt = f"""You are an expert seismological judge consolidating human intensity reports from ZIP Code {zip_code} after the magnitude {mag} Ridgecrest earthquake.
Your task is to estimate the final consolidated Modified Mercalli Intensity (MMI) for this ZIP code based on the physical parameters and individual resident testimonies:

- Earthquake Magnitude: {mag}
- Epicenter Distance: {dist} km

Here are the testimonies from three diverse residents of this ZIP code:
{persona_text}

Consolidate these subjective reports and output a single, final numerical MMI rating on a scale of 1.0 to 10.0 representing the overall impact in this ZIP code.

Output your response ONLY as a JSON object of this format:
{{
  "reasoning": "your consolidation reasoning",
  "consolidated_mmi": float_value
}}"""

            api_num = idx * 5 + 4
            print(f"  - [API Call {api_num}/{total * 5}] Querying LLM-as-Judge Consolidation...")
            messages_judge = [{"role": "user", "content": judge_prompt}]
            raw_judge = call_llm(messages_judge)
            judge_report = extract_json(raw_judge)
            
            if not judge_report or judge_report.get("consolidated_mmi", 0.0) <= 0.0:
                print(f"  - [Validation Failure] Judge consolidation failed. Retrying ZIP {zip_code} in 20 seconds...")
                time.sleep(20.0)
                continue
                
            print(f"    -> Judge consolidated MMI: {judge_report.get('consolidated_mmi')}")
            time.sleep(1.0)
            
            # 4. Single-Prompt Baseline (EMNLP 25 Replication)
            baseline_prompt = f"""You are a disaster simulation model. Predict the human-perceived Modified Mercalli Intensity (MMI) on a scale of 1.0 to 10.0 for ZIP Code {zip_code} during the 2019 Ridgecrest earthquake (magnitude {mag}, depth {depth} km).
The ZIP code is located {dist} km from the epicenter.
Demographics:
- Median Household Income: ${c_data.get('median_income', 50000.0):,.0f}
- College Educated: {c_data.get('college_educated_pct', 30.0)}%
- Population Density: {c_data.get('population_density', 100.0)} people/sq km

Output your response ONLY as a JSON object of this format:
{{
  "reasoning": "your engineering justification",
  "baseline_mmi": float_value
}}"""

            api_num = idx * 5 + 5
            print(f"  - [API Call {api_num}/{total * 5}] Querying Single-Prompt Baseline...")
            messages_base = [{"role": "user", "content": baseline_prompt}]
            raw_base = call_llm(messages_base)
            base_report = extract_json(raw_base)
            
            if not base_report or base_report.get("baseline_mmi", 0.0) <= 0.0:
                print(f"  - [Validation Failure] Baseline prediction failed. Retrying ZIP {zip_code} in 20 seconds...")
                time.sleep(20.0)
                continue
                
            print(f"    -> Baseline predicted MMI: {base_report.get('baseline_mmi')}")
            time.sleep(1.0)
            
            # Store results ONLY when all validations pass
            results[zip_code] = {
                "zip": zip_code,
                "actual_mmi": zcta["actual_mmi"],
                "distance_km": dist,
                "population_density": c_data.get("population_density", 0.0),
                "median_income": c_data.get("median_income"),
                "college_educated_pct": c_data.get("college_educated_pct"),
                "personas": persona_reports,
                "judge_reasoning": judge_report.get("reasoning"),
                "judge_mmi": judge_report.get("consolidated_mmi"),
                "baseline_reasoning": base_report.get("reasoning"),
                "baseline_mmi": base_report.get("baseline_mmi")
            }
            
            # Save cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
                
            # Exit the self-healing loop for this ZIP
            break
            
    print("\nSimulation complete! All ZIP codes processed and cached in data/simulation_results.json.")

if __name__ == "__main__":
    run_simulation()
