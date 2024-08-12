import beacon
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def main():
    print("Purpose:  Illustrate fitting BEACON models, making NAICS predictions, and using cross-validation to optimize hyperparameters")
    print("")

    # Example using 2017 data
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@    Fit BeaconModel    @@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    X, y, sample_weight = beacon.load_naics_data(vintage="2017")
    mod = beacon.BeaconModel(verbose=1)
    mod.fit(X, y, sample_weight)
    mod.summary()

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@    Apply fitted BeaconModel to example business descriptions    @@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("")

    X_test = [
        "blueberry farm",
        "residential remodeling",
        "retail bakery",
        "sporting goods manufacturing",
        "toy wholesaler",
        "car dealer",
        "new car dealership",
        "apothecary",
        "pharmacy",
        "convenience store",
        "pet store",
        "commercial photography",
        "gasoline station",
        "gift store",
        "florist",
        "consultant",
        "landscaping",
        "landscape architect",
        "medical doctor",
        "fast food restaurant",
        "car repair",
        "gobbledygook",
    ]
    preds = mod.predict(X_test)
    print("{:<35}{:<30}{:<15}".format("BUSINESS DESCRIPTION", "CLEAN TEXT", "PREDICTED NAICS"))
    print("{:<35}{:<30}{:<15}".format("--------------------", "----------", "---------------"))
    for i in range(len(X_test)):
        print("{:<35}{:<30}{:<15}".format(X_test[i], mod.clean_text(X_test[i]), preds[i]))
    print("")

    X_test_restaurant = ["restaurant"]
    probs = mod.predict_proba(X_test_restaurant)
    print("NAICS codes with score > 0.01 for the business description 'restaurant':")
    print("")
    print("{:<10}{}".format("NAICS", "SCORE"))
    print("{:<10}{}".format("-----", "-----"))
    for naics in probs[0]:
        if probs[0][naics] > 0.01:
            print("{:<10}{}".format(naics, probs[0][naics]))
    print("")

    X_test_dealer = ["dealer"]
    preds_top10 = mod.predict_top10(X_test_dealer)
    print("Top 10 highest-scoring NAICS codes for the business description 'dealer':")
    print("")
    print("{:<10}".format("NAICS"))
    print("{:<10}".format("-----"))
    for naics in preds_top10[0]:
        print("{:<10}".format(naics))
    print("")

    # Example using 2022 data
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@    Cross-validation example involving 'freq_thresh'          @@@@")
    print("@@@@    with two folds and three candidate values: 1, 2, and 3    @@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("")
    
    X_2022, y_2022, sample_weight_2022 = beacon.load_naics_data(vintage="2022")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=12345)
    mod = GridSearchCV(estimator=beacon.BeaconModel(), param_grid={'freq_thresh': [1, 2, 3]}, cv=skf.split(X_2022, y_2022), verbose=4)
    mod.fit(X_2022, y_2022)
    print("")

    print("Cross-validation results:")
    print("")
    print("{:<21}{}".format("ATTRIBUTE", "VALUE"))
    print("{:<21}{}".format("---------", "-----"))
    for cv_result in mod.cv_results_:
        print("{:<21}{}".format(cv_result, mod.cv_results_[cv_result]))
    print("")
    print("Best value of 'freq_thresh':  {}".format(mod.best_params_["freq_thresh"]))
    print("Best cross-validation score:  {}".format(mod.best_score_))
    print("")

    return

if __name__ == "__main__":
    main()
