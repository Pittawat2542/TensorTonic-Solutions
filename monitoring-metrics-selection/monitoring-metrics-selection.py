def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    N = len(y_true)
    if system_type == "classification":
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == 1 and y_p == 1: tp += 1
            elif y_t == 1 and y_p == 0: fn += 1
            elif y_t == 0 and y_p == 1: fp += 1
            elif y_t == 0 and y_p == 0: tn += 1
        accuracy = (tp + tn) / N if N > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return [("accuracy", accuracy), ("f1", f1), ("precision", precision), ("recall", recall)]
    elif system_type == "regression":
        import math
        mae = sum([abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)]) / N if N > 0 else 0.0
        rmse = math.sqrt(sum([(y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)]) / N) if N > 0 else 0.0
        return [("mae", mae), ("rmse", rmse)]
    elif system_type == "ranking":
        num_relevant_total = sum(y_true)
        top_three = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)[:3]
        precision_at_3 = sum(1 for y_t, _ in top_three if y_t == 1) / 3
        recall_at_3 = (sum(1 for y_t, _ in top_three if y_t == 1) / num_relevant_total if num_relevant_total > 0 else 0.0)
        return [("precision_at_3", precision_at_3), ("recall_at_3", recall_at_3)]