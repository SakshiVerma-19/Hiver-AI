import numpy as np
from sklearn.metrics import cohen_kappa_score

def calculate_calibration_metrics(human_scores: list[int], judge_scores: list[int]) -> dict:
    """
    Computes statistical agreement metrics between human ground truth and LLM Judge predictions.
    - Cohen's Kappa (inter-annotator agreement adjusted for chance)
    - Pearson correlation coefficient
    - Mean Absolute Error (MAE)
    - Exact agreement percentage
    """
    if not human_scores or not judge_scores or len(human_scores) != len(judge_scores):
        return {
            "cohen_kappa": 0.0,
            "pearson_r": 0.0,
            "mae": 0.0,
            "exact_accuracy": 0.0,
            "interpretation": "Insufficient data"
        }

    h = np.array(human_scores, dtype=float)
    j = np.array(judge_scores, dtype=float)

    # Cohen's Kappa
    try:
        kappa = cohen_kappa_score(human_scores, judge_scores)
    except Exception:
        kappa = 0.0

    # Pearson r
    if np.std(h) > 0 and np.std(j) > 0:
        pearson_r = float(np.corrcoef(h, j)[0, 1])
    else:
        pearson_r = 1.0 if np.array_equal(h, j) else 0.0

    # MAE and Exact Agreement
    mae = float(np.mean(np.abs(h - j)))
    exact_match = float(np.mean(h == j)) * 100.0

    # Qualitative interpretation based on Landis & Koch (1977)
    if kappa >= 0.81:
        interp = "Almost Perfect Agreement"
    elif kappa >= 0.61:
        interp = "Substantial Agreement"
    elif kappa >= 0.41:
        interp = "Moderate Agreement"
    elif kappa >= 0.21:
        interp = "Fair Agreement"
    else:
        interp = "Slight / Poor Agreement"

    return {
        "cohen_kappa": round(kappa, 3),
        "pearson_r": round(pearson_r, 3),
        "mae": round(mae, 3),
        "exact_accuracy": round(exact_match, 1),
        "interpretation": interp
    }

def print_calibration_report(human_grounding: list[int], judge_grounding: list[int],
                             human_tone: list[int] = None, judge_tone: list[int] = None):
    print("\n" + "=" * 60)
    print("       LLM-AS-A-JUDGE CALIBRATION REPORT (vs. HUMAN)")
    print("=" * 60)
    
    g_metrics = calculate_calibration_metrics(human_grounding, judge_grounding)
    print(f"\n[Grounding Score Alignment] (N = {len(human_grounding)})")
    print(f"  - Cohen's Kappa      : {g_metrics['cohen_kappa']:.2f} ({g_metrics['interpretation']})")
    print(f"  - Pearson Correlation: {g_metrics['pearson_r']:.2f}")
    print(f"  - Mean Absolute Error: {g_metrics['mae']:.2f} points")
    print(f"  - Exact Score Match  : {g_metrics['exact_accuracy']:.1f}%")

    if human_tone and judge_tone:
        t_metrics = calculate_calibration_metrics(human_tone, judge_tone)
        print(f"\n[Tone & Constraint Alignment] (N = {len(human_tone)})")
        print(f"  - Cohen's Kappa      : {t_metrics['cohen_kappa']:.2f} ({t_metrics['interpretation']})")
        print(f"  - Pearson Correlation: {t_metrics['pearson_r']:.2f}")
        print(f"  - Mean Absolute Error: {t_metrics['mae']:.2f} points")
        print(f"  - Exact Score Match  : {t_metrics['exact_accuracy']:.1f}%")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Quick sanity demonstration
    sample_human = [5, 4, 5, 3, 4, 5, 2, 5, 4, 5]
    sample_judge = [5, 4, 4, 3, 4, 5, 2, 5, 5, 5]
    print_calibration_report(sample_human, sample_judge)
