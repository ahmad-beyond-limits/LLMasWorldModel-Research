import sys
import os

def main():
    print("="*60)
    print("RUNNING MULTI-AGENT SEISMIC SIMULATION PIPELINE")
    print("="*60)
    
    # Check .env existence
    if not os.path.exists(".env"):
        print("Error: .env file is missing. Please create it first by copying .env.example.")
        sys.exit(1)
        
    # Step 1: Fetch USGS Event Data & Sample ZIPs
    print("\n--- STEP 1: Fetching USGS Seismic Data ---")
    import fetch_usgs
    sampled_zctas = fetch_usgs.fetch_usgs_event_and_dyfi()
    if not sampled_zctas:
        print("Failed during USGS data collection. Exiting.")
        sys.exit(1)
        
    # Step 2: Fetch Census Demographics
    print("\n--- STEP 2: Fetching Census Demographics ---")
    import fetch_census
    census_data = fetch_census.fetch_census_demographics()
    if not census_data:
        print("Failed during Census data collection. Exiting.")
        sys.exit(1)
        
    # Step 3: Run Simulation Engine (Sequentially calling OpenRouter LLM)
    print("\n--- STEP 3: Running Simulation Engine (OpenRouter LLM) ---")
    import simulation
    simulation.run_simulation()
    
    # Step 4: Run Statistical Evaluation & Generate Report
    print("\n--- STEP 4: Running Statistical Evaluation & Analysis ---")
    import analyze
    analyze.run_statistical_analysis()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Find your report in results/statistical_summary.txt")
    print("="*60)

if __name__ == "__main__":
    main()
