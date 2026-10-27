from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_best_features(X_train, y_train, k, method='mutual_info'):
    """
    Select top-k features using SelectKBest.
    """
    print(f"  [Feature Selection] Finding top {k} features using mutual info...")
    if method == 'mutual_info':
        score_func = mutual_info_classif
    else:
        raise ValueError(f"Unknown method {method}")
        
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X_train, y_train)
    return selector
