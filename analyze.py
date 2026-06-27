import os
import json
import math

def pearson_r(x, y):
    n = len(x)
    if n <= 1:
        return 0.0
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_y2 = sum(yi**2 for yi in y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def calculate_rmse(predictions, actuals):
    n = len(predictions)
    if n == 0:
        return 0.0
    squared_errors = [(p - a)**2 for p, a in zip(predictions, actuals)]
    return math.sqrt(sum(squared_errors) / n)

def run_regression(x, y):
    """
    Fits a simple linear regression y = beta_0 + beta_1 * x.
    Returns (beta_0, beta_1, r_squared).
    """
    n = len(x)
    if n <= 1:
        return 0.0, 0.0, 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x)**2 for xi in x)
    
    if denominator == 0:
        beta_1 = 0.0
    else:
        beta_1 = numerator / denominator
        
    beta_0 = mean_y - beta_1 * mean_x
    
    # R squared
    ss_total = sum((yi - mean_y)**2 for yi in y)
    ss_residual = sum((yi - (beta_0 + beta_1 * xi))**2 for xi, yi in zip(x, y))
    r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0
    
    return beta_0, beta_1, r_squared

def classify_text_keywords(text):
    """
    Performs basic keyword counting to quantify psychological vs structural attribution.
    """
    text = text.lower()
    psych_keywords = ["panic", "scared", "frightened", "afraid", "terrified", "dizzy", "anxious", "screamed", "nervous"]
    struct_keywords = ["crack", "broke", "fell", "damage", "masonry", "walls", "windows", "shelf", "foundation", "shattered", "chimney"]
    
    psych_count = sum(text.count(kw) for kw in psych_keywords)
    struct_count = sum(text.count(kw) for kw in struct_keywords)
    
    return psych_count, struct_count

def run_statistical_analysis():
    os.makedirs("results", exist_ok=True)
    
    cache_path = "data/simulation_results.json"
    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} does not exist. Run simulation.py first.")
        return
        
    with open(cache_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    zctas = list(results.values())
    print(f"Loaded simulation results for {len(zctas)} ZIP codes.")
    
    actual_mmi = []
    baseline_predictions = []
    mean_persona_predictions = []
    judge_predictions = []
    
    pop_densities = []
    log_pop_densities = []
    
    # Persona-level analysis
    persona_vulnerabilities = []
    persona_mmis = []
    
    total_psych_words = 0
    total_struct_words = 0
    
    judge_biases = []
    dispersals = []
    
    for z in zctas:
        actual = z["actual_mmi"]
        base = z["baseline_mmi"]
        judge = z["judge_mmi"]
        
        # Calculate mean persona prediction
        personas = z.get("personas", [])
        persona_scores = [p.get("reported_mmi", 0.0) for p in personas]
        mean_persona = sum(persona_scores) / len(persona_scores) if persona_scores else 0.0
        
        # Lists for global metrics
        actual_mmi.append(actual)
        baseline_predictions.append(base)
        mean_persona_predictions.append(mean_persona)
        judge_predictions.append(judge)
        
        # Population density
        density = z.get("population_density", 0.0)
        # Handle zero or negative density in log
        if density <= 0:
            density = 0.1
        pop_densities.append(density)
        log_pop_densities.append(math.log10(density))
        
        # Persona level metrics
        # Persona 1: Elderly Renter (Vulnerability = 1.0)
        # Persona 2: Average Homeowner (Vulnerability = 0.5)
        # Persona 3: Modern Apartment Professional (Vulnerability = 0.0)
        vul_scores = [1.0, 0.5, 0.0]
        for score, p_rep in zip(vul_scores, personas):
            reported_score = p_rep.get("reported_mmi", 0.0)
            persona_vulnerabilities.append(score)
            persona_mmis.append(reported_score)
            
            # Attributions
            desc = p_rep.get("felt_description", "")
            p_cnt, s_cnt = classify_text_keywords(desc)
            total_psych_words += p_cnt
            total_struct_words += s_cnt
            
        # Perceptual dispersal (standard deviation of the ZIP code's personas)
        if len(persona_scores) > 1:
            mean_score = sum(persona_scores) / len(persona_scores)
            var_score = sum((s - mean_score)**2 for s in persona_scores) / (len(persona_scores) - 1)
            dispersals.append(math.sqrt(var_score))
            
        # Judge aggregation bias
        judge_biases.append(judge - mean_persona)

    # 1. Compute correlations against ground truth
    r_baseline = pearson_r(baseline_predictions, actual_mmi)
    rmse_baseline = calculate_rmse(baseline_predictions, actual_mmi)
    
    r_mean = pearson_r(mean_persona_predictions, actual_mmi)
    rmse_mean = calculate_rmse(mean_persona_predictions, actual_mmi)
    
    r_judge = pearson_r(judge_predictions, actual_mmi)
    rmse_judge = calculate_rmse(judge_predictions, actual_mmi)
    
    # 2. Behavioral Metrics
    avg_dispersal = sum(dispersals) / len(dispersals) if dispersals else 0.0
    r_vul_mmi = pearson_r(persona_vulnerabilities, persona_mmis)
    avg_judge_bias = sum(judge_biases) / len(judge_biases) if judge_biases else 0.0
    r_p_s_ratio = total_psych_words / total_struct_words if total_struct_words > 0 else 0.0
    
    # 3. Density Bias Linear Regression (Residuals vs Log Population Density)
    # Residual = Predicted - Actual
    res_base = [p - a for p, a in zip(baseline_predictions, actual_mmi)]
    res_mean = [p - a for p, a in zip(mean_persona_predictions, actual_mmi)]
    res_judge = [p - a for p, a in zip(judge_predictions, actual_mmi)]
    
    b0_base, b1_base, r2_base = run_regression(log_pop_densities, res_base)
    b0_mean, b1_mean, r2_mean = run_regression(log_pop_densities, res_mean)
    b0_judge, b1_judge, r2_judge = run_regression(log_pop_densities, res_judge)

    # Print Summary to Terminal
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*50)
    print(f"Sample size: {len(actual_mmi)} ZCTAs")
    print(f"USGS DYFI Ground Truth Mean MMI: {sum(actual_mmi)/len(actual_mmi):.2f}\n")
    
    print("1. Performance Metrics vs USGS Ground-Truth:")
    print(f"  - Baseline Single Prompt: Pearson r = {r_baseline:.4f}, RMSE = {rmse_baseline:.4f}")
    print(f"  - Persona Mean Average : Pearson r = {r_mean:.4f}, RMSE = {rmse_mean:.4f}")
    print(f"  - LLM-as-Judge Consensus: Pearson r = {r_judge:.4f}, RMSE = {rmse_judge:.4f}")
    print("")
    
    print("2. Behavioral & Aggregation Quantifications:")
    print(f"  - Average Perceptual Dispersal (cross-persona std dev): {avg_dispersal:.4f}")
    print(f"  - Vulnerability-Perception Correlation (r_vul_mmi): {r_vul_mmi:.4f}")
    print(f"  - Psychological-to-Structural Word Ratio (R_P/S): {r_p_s_ratio:.4f}")
    print(f"  - Average Judge Aggregation Bias (Delta_Judge): {avg_judge_bias:.4f}")
    print("    (Interpretation: > 0 means Pessimism Bias, < 0 means Optimism Bias)")
    print("")
    
    print("3. Density Bias Regression (Residual = beta_0 + beta_1 * log10(Density)):")
    print(f"  - Baseline: Slope (beta_1) = {b1_base:.4f}, R2 = {r2_base:.4f}")
    print(f"  - Persona Mean: Slope (beta_1) = {b1_mean:.4f}, R2 = {r2_mean:.4f}")
    print(f"  - LLM-as-Judge: Slope (beta_1) = {b1_judge:.4f}, R2 = {r2_judge:.4f}")
    print("    (Interpretation: Negative slope supports the hypothesis that the model")
    print("     systematically over-predicts in rural areas and under-predicts in urban ones.)")
    print("="*50)
    
    # Save text report
    report_path = "results/statistical_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("ACADEMIC EVALUATION REPORT: SEISMIC SIMULATION BIAS AUDIT\n")
        f.write("="*60 + "\n")
        f.write(f"Sample size: {len(actual_mmi)} ZCTAs\n\n")
        
        f.write("SECTION I: Ground-Truth Alignment & Model Comparison\n")
        f.write("-"*60 + "\n")
        f.write(f"Baseline Single Prompt (EMNLP 25 Replication):\n")
        f.write(f"  Pearson r = {r_baseline:.5f}\n  RMSE = {rmse_baseline:.5f}\n")
        f.write(f"Independent Persona Averaging (MACS Mean):\n")
        f.write(f"  Pearson r = {r_mean:.5f}\n  RMSE = {rmse_mean:.5f}\n")
        f.write(f"LLM-as-Judge Consolidated Consensus:\n")
        f.write(f"  Pearson r = {r_judge:.5f}\n  RMSE = {rmse_judge:.5f}\n\n")
        
        f.write("SECTION II: Behavioral & Cognitive Quantifications\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean Perceptual Dispersal (cross-persona std dev)   : {avg_dispersal:.5f}\n")
        f.write(f"Vulnerability-Perception Correlation (r_vul_mmi)    : {r_vul_mmi:.5f}\n")
        f.write(f"Psychological-to-Structural Attribution Ratio (R_P/S): {r_p_s_ratio:.5f}\n")
        f.write(f"Mean Judge Aggregation Bias (Delta_Judge)           : {avg_judge_bias:.5f}\n\n")
        
        f.write("SECTION III: Population Density Auditing\n")
        f.write("-"*60 + "\n")
        f.write(f"Baseline Residual vs Log10(Density):\n")
        f.write(f"  Intercept = {b0_base:.5f}, Slope = {b1_base:.5f}, R2 = {r2_base:.5f}\n")
        f.write(f"Persona Mean Residual vs Log10(Density):\n")
        f.write(f"  Intercept = {b0_mean:.5f}, Slope = {b1_mean:.5f}, R2 = {r2_mean:.5f}\n")
        f.write(f"LLM-as-Judge Residual vs Log10(Density):\n")
        f.write(f"  Intercept = {b0_judge:.5f}, Slope = {b1_judge:.5f}, R2 = {r2_judge:.5f}\n\n")
        f.write("="*60 + "\n")
        
    print(f"Saved statistical report to: {report_path}")
    
    # 4. Generate visual plots using matplotlib if available
    try:
        import matplotlib.pyplot as plt
        print("Matplotlib available. Generating plots...")
        plt.figure(figsize=(10, 6))
        
        plt.scatter(log_pop_densities, res_base, color='red', alpha=0.6, label='Baseline Residuals')
        plt.scatter(log_pop_densities, res_judge, color='blue', alpha=0.6, label='LLM-as-Judge Residuals')
        
        # Plot regression lines
        x_vals = [min(log_pop_densities), max(log_pop_densities)]
        y_base = [b0_base + b1_base * x for x in x_vals]
        y_judge = [b0_judge + b1_judge * x for x in x_vals]
        
        plt.plot(x_vals, y_base, color='red', linestyle='--', label=f'Baseline Trend (slope: {b1_base:.2f})')
        plt.plot(x_vals, y_judge, color='blue', linestyle='-', label=f'Judge Trend (slope: {b1_judge:.2f})')
        
        plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
        plt.xlabel('Log10 Population Density (people / sq km)')
        plt.ylabel('MMI Prediction Residual (Predicted - Actual)')
        plt.title('MMI Prediction Residuals vs. Population Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/density_bias_analysis.png", dpi=150)
        print("Saved density bias analysis chart to: results/density_bias_analysis.png")
    except ImportError:
        print("Matplotlib is not installed. Skipping PNG generation. You can view the raw numbers in results/statistical_summary.txt")

if __name__ == "__main__":
    run_statistical_analysis()
