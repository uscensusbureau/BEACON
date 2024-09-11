"""
Business Establishment Automated Classification of NAICS (BEACON)
"""

# Authors:  Brian Dumbacher <brian.dumbacher@census.gov>
#           Daniel Whitehead <daniel.whitehead@census.gov>
#           Jiseok Jeong <jiseok.jeong@census.gov>
#           Sarah Pfeiff <sarah.pfeiff@census.gov>

import io
import numpy as np
import re
import time
from numbers import Integral, Real
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_X_y, check_array, check_random_state, check_is_fitted

def load_naics_data(vintage="2017", shuffle=False, random_state=0):
    """
    Load NAICS data

    This method loads example NAICS datasets for fitting a BEACON classification model.

    Parameters
    ----------
    vintage : str, default="2017"
        Vintage of NAICS data. Valid values are "2017" and "2022".
    
    shuffle : boolean, default=False
        Flag indicating whether to shuffle the observations.
    
    random_state : int or RandomState instance, default=0
        The seed of the pseudo random number generator used to shuffle the observations.
        Pass an int for reproducible output across multiple function calls.
        Used only if shuffle=True.
    
    Returns
    -------
    X : numpy.ndarray
        1D NumPy array of strings representing business descriptions.

    y : numpy.ndarray
        1D NumPy array of strings representing 6-digit NAICS codes.
    
    sample_weight : numpy.ndarray
        1D NumPy array of numerical sample weights.
    """

    # Check vintage
    if not isinstance(vintage, str):
        raise ValueError("Parameter 'vintage' is not a string.")
    if vintage not in ("2017", "2022"):
        raise ValueError("Parameter 'vintage' is invalid. Valid values are '2017' and '2022'.")

    # Prepare for reading data
    file = "example_data_{}.txt".format(vintage)
    data_tups = []
    n_vars = 0
    line_number = 0
    len_error_flag = False

    # Read data line by line
    # io.open() checks for the existence of the data file and raises FileNotFoundError accordingly
    f = io.open(file, "r")
    for line in f:
        line_number += 1
        line_strip = line.strip()
        if line_number == 1:
            if line_strip != "":
                var_names = line_strip.split("|")
                if (var_names != ["TEXT", "NAICS"] and var_names != ["TEXT", "NAICS", "SAMPLE_WEIGHT"]):
                    raise ValueError("Input data file does not have the expected format: TEXT|NAICS|SAMPLE_WEIGHT (SAMPLE_WEIGHT optional).")
                n_vars = len(var_names)
            else:
                raise ValueError("No variable names appear on the first line of input data file.")
        else:
            if line_strip != "":
                row_data = line_strip.split("|")
                if len(row_data) != n_vars:
                    len_error_flag = True
                data_tups.append(tuple(row_data))
    f.close()

    # Check data
    n_obs = len(data_tups)
    if n_obs == 0:
        raise ValueError("Input data file contains zero observations.")
    if len_error_flag:
        raise ValueError("Input data file contains observations with inconsistent numbers of variables.")

    if shuffle:
        # check_random_state() is provided by sklearn.utils.validation
        random_state = check_random_state(random_state)
        uniform_random = random_state.uniform(0, 1, n_obs)
        data_tups_random = [(data_tups[i], uniform_random[i]) for i in range(n_obs)]
        data_tups_random.sort(key=lambda z: z[1])
        data_tups = [z[0] for z in data_tups_random]

    # Create X, y, and sample_weight
    X = np.array([tup[0] for tup in data_tups])
    y = np.array([tup[1] for tup in data_tups])
    if n_vars == 2:
        sample_weight = np.ones(X.shape[0])
    elif n_vars == 3:
        # float() checks whether the sample weights can be converted to type float and raises ValueError accordingly
        sample_weight = np.array([float(tup[2]) for tup in data_tups])

    return X, y, sample_weight

class BeaconModel(BaseEstimator, ClassifierMixin):
    """
    BEACON text classification model for predicting 6-digit NAICS codes

    Business Establishment Automated Classification of NAICS (BEACON) was developed by the U.S. Census Bureau to help
    respondents to economic surveys self-classify their establishment's primary business activity in real time. BEACON is
    based on natural language processing, machine learning, and information retrieval.
    
    The methodology presented here is a simplified version of what the Census Bureau uses in production. The production
    version of BEACON employs a detailed text cleaning algorithm with tens of thousands of rules and a large, rich
    dataset compiled from various public and confidential sources. This Python program demonstrates how one can
    implement a simple BEACON model as an extension of scikit-learn's (Pedregosa et al., 2011) BaseEstimator and
    ClassifierMixin classes. Large sections of BEACON's codebase for processing natural language and cleaning text come
    from Snowball (Porter, 2001) and NLTK (Bird, 2006).

    Parameters
    ----------
    freq_thresh : int, default=1
        Training data frequency threshold for determining whether to include a feature in the dictionary. Does not take
        sample weights into account.
    
    wt_umb : float, default=0.6
        Ensemble weight of the "umbrella" sub-model.
    
    wt_exact : float, default=0.3
        Ensemble weight of the "exact" sub-model.
    
    verbose : int, default=0
        Verbosity indicator. Any positive value turns on messages during model fitting.

    Attributes
    ----------
    naics_ : list
        List of unique 6-digit NAICS codes in data
    
    n_naics_ : int
        Number of unique 6-digit NAICS codes in data
    
    sectors_  : list
        List of unique sectors in data
    
    n_sectors_ : int
        Number of unique sectors in data
    
    sample_sizes_ : dict
        Dictionary of sample sizes by sector
    
    naics_indices_ : dict
        Dictionary of NAICS index mappings by sector

    dict_ncombs_props_ : dict
        Dictionary of n-comb features by sector

    dict_ncombs_weights_ : dict
        Dictionary of n-comb purity weights by sector
        
    dict_ems_props_ : dict
        Dictionary of exact features by sector
        
    dict_ems_weights_ : dict
        Dictionary of exact feature purity weights by sector

    References
    ----------
    Bird, S. (2006). NLTK: The Natural Language Toolkit. Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions.
        Sydney, Australia: Association for Computational Linguistics, 69-72.

    Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R.,
        Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011).
        Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

    Porter, M.F. (2001). Snowball: a language for stemming algorithms. <http://snowball.tartarus.org/texts/introduction.html>.
        [Online; accessed 11 March 2024].
    """

    # Parameter contraints
    # Interval is provided by sklearn.utils._param_validation
    # Integral and Real are provided by numbers
    _parameter_constraints: dict = {
        "freq_thresh": [Interval(Integral, 1, None, closed="left")],
        "wt_umb": [Interval(Real, 0, 1, closed="both")],
        "wt_exact": [Interval(Real, 0, 1, closed="both")],
        "verbose": ["verbose"],
    }

    # Initialize BeaconModel object
    def __init__(self, freq_thresh=1, wt_umb=0.6, wt_exact=0.3, verbose=0):
        self.freq_thresh = freq_thresh
        self.wt_umb = wt_umb
        self.wt_exact = wt_exact
        self.verbose = verbose

    # Validate parameters
    def __validate_parameters(self):
        # BeaconModel inherits the method _validate_params() from the parent class BaseEstimator
        # _validate_params() checks the parameter values against _parameter_constraints defined above
        self._validate_params()
        if (self.wt_umb + self.wt_exact > 1):
            raise ValueError("Sum of parameters 'wt_umb' and 'wt_exact' is greater than 1: {} + {} = {}.".format(self.wt_umb, self.wt_exact, self.wt_umb + self.wt_exact))
        return
    
    # Validate data: X, y, and sample_weight
    """
    Validate data

    This is the main method for validating input data

    Parameters
    ----------
    X : 1D array-like data structure of strings representing business descriptions

    y : 1D array-like data structure of strings representing 6-digit NAICS codes

    sample_weight : 1D array-like data structure of numbers representing sample weights
    
    Returns
    -------
    X : numpy.ndarray
        1D NumPy array of strings representing business descriptions

    y : numpy.ndarray
        1D NumPy array of strings representing 6-digit NAICS codes
    
    sample_weight : numpy.ndarray
        1D NumPy array of numbers representing sample weights
    """
    def __validate_data(self, X, y, sample_weight):
        # check_X_y() is provided by sklearn.utils.validation
        X = self.__validate_data_X(X)
        y = self.__validate_data_y(y)
        X, y = check_X_y(X, y, dtype=str, ensure_2d=False)
        # Check dimensionality of 'X' and 'y' again
        if len(X.shape) > 1:
            raise ValueError("Input 'X' is not 1D.")
        if len(y.shape) > 1:
            raise ValueError("Input 'y' is not 1D.")
        if not isinstance(sample_weight, type(None)):
            sample_weight = self.__validate_data_sample_weight(sample_weight)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("Input 'sample_weight' has an inconsistent number of observations: {}.".format(sample_weight.shape[0]))
        else:
            sample_weight = np.ones(X.shape[0])
        return X, y, sample_weight

    # Validate data: X only
    def __validate_data_X(self, X):
        if isinstance(X, str):
            raise ValueError("Input 'X' must be a 1D array-like data structure of strings.")
        # check_array() is provided by sklearn.utils.validation
        X = check_array(X, dtype=str, ensure_2d=False)
        if len(X.shape) > 1:
            raise ValueError("Input 'X' is not 1D.")
        return X

    # Validate data: y only
    def __validate_data_y(self, y):
        if isinstance(y, str):
           raise ValueError("Input 'y' must be a 1D array-like data structure of strings.")
        # check_array() is provided by sklearn.utils.validation
        y = check_array(y, dtype=str, ensure_2d=False)
        if len(y.shape) > 1:
            raise ValueError("Input 'y' is not 1D.")
        return y
    
    # Validate data: sample_weight only
    def __validate_data_sample_weight(self, sample_weight):
        if isinstance(sample_weight, int) or isinstance(sample_weight, float):
            raise ValueError("Input 'sample_weight' must be a 1D array-like data structure of numbers.")
        # check_array() is provided by sklearn.utils.validation
        sample_weight = check_array(sample_weight, dtype="numeric", ensure_2d=False)
        if len(sample_weight.shape) > 1:
                raise ValueError("Input 'sample_weight' is not 1D.")
        return sample_weight

    # Stop words
    __stop_words = (
        # ADD YOUR OWN STOP WORDS HERE
        # Below are examples based on NLTK
        "a",
        "am",
        "an",
        "and",
        "are",
        "as",
        "but",
        "by",
        "for",
        "from",
        "i",
        "if",
        "in",
        "is",
        "it",
        "on",
        "or",
        "other",
        "since",
        "so",
        "the",
        "this",
        "to",
        "we",
        "with",
        "you",
    )

    # NLTK implementation of the Porter 2/Snowball stemming algorithm with slight modifications
    __vowels = "aeiouy"
    __double_consonants = ("bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt")
    __li_ending = "cdeghkmnrt"
    __step1a_suffixes = ("sses", "ied", "ies", "us", "ss", "s")
    __step1b_suffixes = ("eedly", "ingly", "edly", "eed", "ing", "ed")
    __step2_suffixes = (
        "ization",
        "ational",
        "fulness",
        "ousness",
        "iveness",
        "tional",
        "biliti",
        "lessli",
        "entli",
        "ation",
        "alism",
        "aliti",
        "ousli",
        "iviti",
        "fulli",
        "enci",
        "anci",
        "abli",
        "izer",
        "ator",
        "alli",
        "bli",
        "ogi",
        "li",
    )
    __step3_suffixes = (
        "ational",
        "tional",
        "alize",
        "icate",
        "iciti",
        "ative",
        "ical",
        "ness",
        "ful",
    )
    __step4_suffixes = (
        "ement",
        "ance",
        "ence",
        "able",
        "ible",
        "ment",
        "ant",
        "ent",
        "ism",
        "ate",
        "iti",
        "ous",
        "ive",
        "ize",
        "ion",
        "al",
        "er",
        "ic",
    )
    __step5_suffixes = ("e", "l")
    __step6_suffixes = (
        "curist",
        "graphi",
        "logi",
        "logist",
        "nomi",
        "nomist",
        "pathi",
        "pathet",
        "physicist",
        "scopi",
        "therapeut",
        "therapi",
        "therapist",
        "tomi",
        "tomist",
        "tri",
        "trist",
        "trician",
        "turist",
    )
    __special_words = {
        # ADD YOUR OWN STEMMING RULES HERE FOR CORRECTING OVERSTEMMING ERRORS
        # Below are examples from Porter (2001) and Bird (2006)
        "skis": "ski",
        "skies": "sky",
        "dying": "die",
        "lying": "lie",
        "tying": "tie",
        "idly": "idl",
        "gently": "gentl",
        "ugly": "ugli",
        "early": "earli",
        "only": "onli",
        "singly": "singl",
        "sky": "sky",
        "news": "news",
        "howe": "howe",
        "atlas": "atlas",
        "cosmos": "cosmos",
        "bias": "bias",
        "andes": "andes",
        "inning": "inning",
        "innings": "inning",
        "outing": "outing",
        "outings": "outing",
        "canning": "canning",
        "cannings": "canning",
        "herring": "herring",
        "herrings": "herring",
        "earring": "earring",
        "earrings": "earring",
        "proceed": "proceed",
        "proceeds": "proceed",
        "proceeded": "proceed",
        "proceeding": "proceed",
        "exceed": "exceed",
        "exceeds": "exceed",
        "exceeded": "exceed",
        "exceeding": "exceed",
        "succeed": "success",
        "succeeds": "success",
        "succeeded": "success",
        "succeeding": "success",
    }

    def __r1r2(self, word, vowels):
        r1 = ""
        r2 = ""
        for i in range(1, len(word)):
            if word[i] not in vowels and word[i - 1] in vowels:
                r1 = word[i + 1 :]
                break
        for i in range(1, len(r1)):
            if r1[i] not in vowels and r1[i - 1] in vowels:
                r2 = r1[i + 1 :]
                break
        return (r1, r2)
    
    def __suffix_replace(self, original, old, new):
        return original[: -len(old)] + new
    
    # Stem words
    # Main method of the Porter 2/Snowball stemming algorithm
    def __stem(self, word):
        if word in self.__special_words:
            return self.__special_words[word]
        elif len(word) <= 3:
            return word

        if word.startswith("y"):
            word = "".join(("Y", word[1:]))
        for i in range(1, len(word)):
            if word[i - 1] in self.__vowels and word[i] == "y":
                word = "".join((word[:i], "Y", word[i + 1 :]))

        step1a_vowel_found = False
        step1b_vowel_found = False

        r1 = ""
        r2 = ""

        if word.startswith(("gener", "commun", "arsen")):
            if word.startswith(("gener", "arsen")):
                r1 = word[5:]
            else:
                r1 = word[6:]
            for i in range(1, len(r1)):
                if r1[i] not in self.__vowels and r1[i - 1] in self.__vowels:
                    r2 = r1[i + 1 :]
                    break
        else:
            r1, r2 = self.__r1r2(word, self.__vowels)

        # STEP 1a
        for suffix in self.__step1a_suffixes:
            if word.endswith(suffix):
                if suffix == "sses":
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
                elif suffix in ("ied", "ies"):
                    if len(word[: -len(suffix)]) > 1:
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                    else:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                elif suffix == "s":
                    for letter in word[:-2]:
                        if letter in self.__vowels:
                            step1a_vowel_found = True
                            break
                    if step1a_vowel_found:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                break

        # STEP 1b
        for suffix in self.__step1b_suffixes:
            if word.endswith(suffix):
                if suffix in ("eed", "eedly"):
                    if r1.endswith(suffix):
                        word = self.__suffix_replace(word, suffix, "ee")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "ee")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "ee")
                        else:
                            r2 = ""
                else:
                    for letter in word[: -len(suffix)]:
                        if letter in self.__vowels:
                            step1b_vowel_found = True
                            break
                    if step1b_vowel_found:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        if word.endswith(("at", "bl", "iz")):
                            word = "".join((word, "e"))
                            r1 = "".join((r1, "e"))
                            if len(word) > 5 or len(r1) >= 3:
                                r2 = "".join((r2, "e"))
                        elif word.endswith(self.__double_consonants):
                            word = word[:-1]
                            r1 = r1[:-1]
                            r2 = r2[:-1]
                        elif (
                            r1 == ""
                            and len(word) >= 3
                            and word[-1] not in self.__vowels
                            and word[-1] not in "wxY"
                            and word[-2] in self.__vowels
                            and word[-3] not in self.__vowels
                        ) or (
                            r1 == ""
                            and len(word) == 2
                            and word[0] in self.__vowels
                            and word[1] not in self.__vowels
                        ):
                            word = "".join((word, "e"))
                            if len(r1) > 0:
                                r1 = "".join((r1, "e"))
                            if len(r2) > 0:
                                r2 = "".join((r2, "e"))
                break

        # STEP 1c
        if len(word) > 2 and word[-1] in "yY" and word[-2] not in self.__vowels:
            word = "".join((word[:-1], "i"))
            if len(r1) >= 1:
                r1 = "".join((r1[:-1], "i"))
            else:
                r1 = ""
            if len(r2) >= 1:
                r2 = "".join((r2[:-1], "i"))
            else:
                r2 = ""

        # STEP 2
        for suffix in self.__step2_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if (
                        suffix in ("entli", "fulli", "lessli", "tional")
                        or (suffix == "li" and word[-3] in self.__li_ending)
                    ):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                    elif suffix in ("enci", "anci", "abli"):
                        word = "".join((word[:-1], "e"))
                        if len(r1) >= 1:
                            r1 = "".join((r1[:-1], "e"))
                        else:
                            r1 = ""
                        if len(r2) >= 1:
                            r2 = "".join((r2[:-1], "e"))
                        else:
                            r2 = ""
                    elif suffix in ("izer", "ization"):
                        word = self.__suffix_replace(word, suffix, "ize")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "ize")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "ize")
                        else:
                            r2 = ""
                    elif suffix in ("ational", "ation", "ator"):
                        word = self.__suffix_replace(word, suffix, "ate")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "ate")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "ate")
                        else:
                            r2 = "e"
                    elif suffix in ("alism", "aliti", "alli"):
                        word = self.__suffix_replace(word, suffix, "al")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "al")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "al")
                        else:
                            r2 = ""
                    elif suffix == "fulness":
                        word = word[:-4]
                        r1 = r1[:-4]
                        r2 = r2[:-4]
                    elif suffix in ("ousli", "ousness"):
                        word = self.__suffix_replace(word, suffix, "ous")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "ous")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "ous")
                        else:
                            r2 = ""
                    elif suffix in ("iveness", "iviti"):
                        word = self.__suffix_replace(word, suffix, "ive")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "ive")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "ive")
                        else:
                            r2 = "e"
                    elif suffix in ("biliti", "bli"):
                        word = self.__suffix_replace(word, suffix, "ble")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "ble")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "ble")
                        else:
                            r2 = ""
                    elif suffix == "ogi" and word[-4] == "l":
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                break

        # STEP 3
        for suffix in self.__step3_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix == "tional":
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                    elif suffix == "ational":
                        word = self.__suffix_replace(word, suffix, "ate")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "ate")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "ate")
                        else:
                            r2 = ""
                    elif suffix == "alize":
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                    elif suffix in ("icate", "iciti", "ical"):
                        word = self.__suffix_replace(word, suffix, "ic")
                        if len(r1) >= len(suffix):
                            r1 = self.__suffix_replace(r1, suffix, "ic")
                        else:
                            r1 = ""
                        if len(r2) >= len(suffix):
                            r2 = self.__suffix_replace(r2, suffix, "ic")
                        else:
                            r2 = ""
                    elif suffix in ("ful", "ness"):
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                    elif suffix == "ative" and r2.endswith(suffix):
                        word = word[:-5]
                        r1 = r1[:-5]
                        r2 = r2[:-5]
                break

        # STEP 4
        for suffix in self.__step4_suffixes:
            if word.endswith(suffix):
                if r2.endswith(suffix):
                    if suffix == "ion":
                        if word[-4] in "st":
                            word = word[:-3]
                            r1 = r1[:-3]
                            r2 = r2[:-3]
                    else:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                break

        # STEP 5
        if (
            (r2.endswith("l") and word[-2] == "l")
            or r2.endswith("e")
            or (
                r1.endswith("e")
                and len(word) >= 4
                and (
                    word[-2] in self.__vowels
                    or word[-2] in "wxY"
                    or word[-3] not in self.__vowels
                    or word[-4] in self.__vowels
                )
            )
        ):
            word = word[:-1]

        word = word.replace("Y", "y")

        # STEP 6
        for suffix in self.__step6_suffixes:
            if word.endswith(suffix):
                if (
                    (suffix == "graphi" and len(word) >= 9)
                    or (suffix == "logi" and len(word) >= 7)
                    or (suffix == "nomi" and len(word) >= 7)
                    or (suffix == "pathi" and len(word) >= 6)
                    or (suffix == "scopi" and len(word) >= 8)
                    or (suffix == "therapi")
                    or (suffix == "tomi" and len(word) >= 7)
                    or (suffix == "tri" and len(word) >= 8 and word[-4] in "ae")
                ):
                    word = word[:-1]
                elif suffix == "pathet" and len(word) >= 7:
                    word = word[:-2]
                elif (
                    (suffix == "curist" and len(word) >= 8)
                    or (suffix == "logist" and len(word) >= 9)
                    or (suffix == "nomist" and len(word) >= 9)
                    or (suffix == "therapeut")
                    or (suffix == "therapist")
                    or (suffix == "tomist" and len(word) >= 9)
                    or (suffix == "trist" and len(word) >= 10 and word[-6] in "ae")
                    or (suffix == "turist" and len(word) >= 8)
                ):
                    word = word[:-3]
                elif (
                    (suffix == "physicist")
                    or (suffix == "trician" and len(word) >= 10 and word[-8] in "ae")
                ):
                    word = word[:-5]
                break

        return word
    
    # Mapping rules
    __map_dict = {
        # ADD YOUR OWN MAPPING RULES HERE FOR CORRECTING UNDERSTEMMING ERRORS
        # Below are a few examples
        "auto": "car",
        "automobil": "car",
        "automot": "car",
    }

    # Map stemmed words to other stems
    def __map(self, word):
        return self.__map_dict.get(word, word)
    
    # Clean text
    def clean_text(self, text):
        text = text.lower()

        # ADD YOUR OWN CLEANING RULES HERE
        # Below are a few examples
        text = re.sub(r"\bcarrepair\b", r" car repair ", text)
        text = re.sub(r"\block[ -]+smith", r" locksmith", text)
        text = re.sub(r"\(except.*\)", r" ", text)

        # Remove non-letters and extraneous whitespace
        text = re.sub(r"[^a-z]+", r" " , text)
        text = text.strip()
        # Remove stop words
        text = " ".join(word for word in text.split(" ") if word not in self.__stop_words)
        # Stem words
        text = " ".join(self.__stem(word) for word in text.split(" ") if word != "")
        # Map stemmed words to other stems
        text = " ".join(self.__map(word) for word in text.split(" ") if word != "")
        return text

    # Given tokenized clean text and a value of n, return a list of n-combs
    # BeaconModel considers only 1-, 2-, and 3-combs
    def __get_ncombs(self, tokens, n):
        tokens = sorted(set(tokens))
        n_tokens = len(tokens)
        ncombs = []
        if n == 1 and n_tokens >= 1 and tokens[0] != "":
            for i in range(n_tokens):
                ncomb = tokens[i]
                ncombs.append(ncomb)
        elif n == 2 and n_tokens >= 2:
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    ncomb = "_".join(sorted([tokens[i], tokens[j]]))
                    ncombs.append(ncomb)
        elif n == 3 and n_tokens >= 3:
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    for k in range(j + 1, n_tokens):
                        ncomb = "_".join(sorted([tokens[i], tokens[j], tokens[k]]))
                        ncombs.append(ncomb)
        return ncombs
    
    # Given a 6-digit NAICS code, return the sector
    def __get_sector(self, naics):
        sector = naics[:2]
        # Manufacturing
        if sector in ("32", "33"):
            return "31"
        # Retail Trade
        elif sector == "45":
            return "44"
        # Transportation and Warehousing
        elif sector == "49":
            return "48"
        return sector

    # Fit BeaconModel for a given sector
    # The method fit() calls __fit_sector() multiple times
    def __fit_sector(self, X_clean, y, sample_weight, sector):
        # Preparation
        sample_size = len(X_clean)
        naics_unique = sorted(set(y))
        naics_index = {}
        for i in range(len(naics_unique)):
            naics_index[naics_unique[i]] = i
        n_naics = len(naics_unique)

        # Dictionary of n-combs
        ncombs_list = []
        ncombs_freqs_raw = {}
        ncombs_freqs_sw = {}
        for i in range(len(X_clean)):
            x_clean_temp = X_clean[i]
            naics_temp = y[i]
            sample_weight_temp = sample_weight[i]
            if x_clean_temp != "":
                tokens = x_clean_temp.split(" ")
                ncs = self.__get_ncombs(tokens, 1) + self.__get_ncombs(tokens, 2) + self.__get_ncombs(tokens, 3)
                ncs_unique = list(set(ncs))
                for nc in ncs_unique:
                    if nc not in ncombs_freqs_raw:
                        ncombs_freqs_raw[nc] = [0] * n_naics
                        ncombs_freqs_sw[nc] = [0] * n_naics
                    ncombs_freqs_raw[nc][naics_index[naics_temp]] += 1
                    ncombs_freqs_sw[nc][naics_index[naics_temp]] += sample_weight_temp
                ncombs_list.extend(ncs_unique)
        ncombs_unique = sorted(set(ncombs_list))

        # Apply freq_thresh to determine n-comb features
        top_ncombs = [nc for nc in ncombs_unique if sum(ncombs_freqs_raw[nc]) >= self.freq_thresh]

        # N-comb proportions
        top_ncombs_props = {}
        for nc in top_ncombs:
            row_total_freq_sw = 1.0 * sum(ncombs_freqs_sw[nc])
            top_ncombs_props[nc] = [freq_sw / row_total_freq_sw for freq_sw in ncombs_freqs_sw[nc]]

        # N-comb purity weights
        top_ncombs_weights = {}
        for nc in top_ncombs:
            weight_temp = max(top_ncombs_props[nc]) - 1/n_naics
            weight_temp_norm = (n_naics/(n_naics - 1)) * weight_temp
            weight_temp_round = round(weight_temp_norm, 4)
            if weight_temp_round < 0.0001:
                weight_temp_round = 0.0001
            top_ncombs_weights[nc] = weight_temp_round
        
        # Dictionary of exact features
        ems_list = []
        ems_freqs_raw = {}
        ems_freqs_sw = {}
        for i in range(len(X_clean)):
            x_clean_temp = X_clean[i]
            naics_temp = y[i]
            sample_weight_temp = sample_weight[i]
            if x_clean_temp != "":
                tokens = x_clean_temp.split(" ")
                em = "_".join(sorted(set(token for token in tokens if token in top_ncombs_weights)))
                if em != "":
                    if em not in ems_freqs_raw:
                        ems_freqs_raw[em] = [0] * n_naics
                        ems_freqs_sw[em] = [0] * n_naics
                    ems_freqs_raw[em][naics_index[naics_temp]] += 1
                    ems_freqs_sw[em][naics_index[naics_temp]] += sample_weight_temp
                    ems_list.append(em)
        ems_unique = sorted(set(ems_list))

        # Apply freq_thresh to determine exact features
        top_ems = [em for em in ems_unique if sum(ems_freqs_raw[em]) >= self.freq_thresh]

        # Exact feature proportions
        top_ems_props = {}
        for em in top_ems:
            row_total_freq_sw = 1.0 * sum(ems_freqs_sw[em])
            top_ems_props[em] = [freq_sw / row_total_freq_sw for freq_sw in ems_freqs_sw[em]]

        # Exact feature purity weights
        top_ems_weights = {}
        for em in top_ems:
            weight_temp = max(top_ems_props[em]) - 1/n_naics
            weight_temp_norm = (n_naics/(n_naics - 1)) * weight_temp
            weight_temp_round = round(weight_temp_norm, 4)
            if weight_temp_round < 0.0001:
                weight_temp_round = 0.0001
            top_ems_weights[em] = weight_temp_round

        # Populate BeaconModel dictionaries with sector-specific information
        self.sample_sizes_[sector] = sample_size
        self.naics_indices_[sector] = naics_index
        self.dict_ncombs_props_[sector] = top_ncombs_props
        self.dict_ncombs_weights_[sector] = top_ncombs_weights
        self.dict_ems_props_[sector] = top_ems_props
        self.dict_ems_weights_[sector] = top_ems_weights
        return

    # Fit BeaconModel
    def fit(self, X, y, sample_weight=None):
        if self.verbose:
            print("")
            print("Parameter and data validation")
        # Validate parameters and data
        # __validate_parameters() and __validate_data() raise ValueErrors if the input does not have the expected format
        self.__validate_parameters()
        X, y, sample_weight = self.__validate_data(X, y, sample_weight)
 
        # Determine sectors and NAICS codes
        naics_unique = sorted(set(y))
        n_naics = len(naics_unique)
        y_sector = [self.__get_sector(naics) for naics in y]
        sectors_unique = sorted(set(y_sector))
        n_sectors = len(sectors_unique)
        if n_sectors == 1:
            raise ValueError("BeaconModel requires data from at least two sectors.")
        in_sector = {sector: [] for sector in sectors_unique}
        for i in range(len(y_sector)):
            in_sector[y_sector[i]].append(i)
        for sector in sectors_unique:
            if len(set(y[in_sector[sector]])) == 1:
                raise ValueError("BeaconModel requires data from at least two NAICS codes within each sector.")

        # Clean text
        t0 = time.time()
        X_clean = np.array([self.clean_text(x) for x in X])
        t1 = time.time()
        if self.verbose:
            print("Text cleaning (time = {}s)".format(round(t1 - t0, 3)))

        # Initialize BeaconModel attributes
        # As per Python nomenclature conventions, class attributes estimated from the sample data end in "_"
        # The trailing "_" is used to check whether the model has been fitted (see comments on the check_is_fitted() method)
        self.naics_ = naics_unique
        self.n_naics_ = n_naics
        self.sectors_ = sectors_unique
        self.n_sectors_ = n_sectors
        self.sample_sizes_ = {}
        self.naics_indices_ = {}
        self.dict_ncombs_props_ = {}
        self.dict_ncombs_weights_ = {}
        self.dict_ems_props_ = {}
        self.dict_ems_weights_ = {}

        # Populate BeaconModel dictionary, one sector at a time
        if self.verbose:
            print("Dictionary creation")
        sector_count = 1
        t0 = time.time()
        self.__fit_sector(X_clean, y_sector, sample_weight, "00")
        t1 = time.time()
        if self.verbose:
            print("[{:>2}/{}] ..... Sector 00 (time = {}s)".format(sector_count, n_sectors + 1, round(t1 - t0, 3)))
        for sector in sectors_unique:
            sector_count += 1
            t0 = time.time()
            self.__fit_sector(X_clean[in_sector[sector]], y[in_sector[sector]], sample_weight[in_sector[sector]], sector)
            t1 = time.time()
            if self.verbose:
                print("[{:>2}/{}] ..... Sector {} (time = {}s)".format(sector_count, n_sectors + 1, sector, round(t1 - t0, 3)))

        return self

    # Return list of n-comb features used by the "umbrella" sub-model
    def __get_feats_umb(self, nc1s, nc2s, nc3s):
        nc2_sets = [set(nc2.split("_")) for nc2 in nc2s]
        nc3_sets = [set(nc3.split("_")) for nc3 in nc3s]
        nc1s_umb = [nc1 for nc1 in nc1s if all([not set([nc1]) < nc2_set for nc2_set in nc2_sets])]
        nc2s_umb = [nc2 for nc2 in nc2s if all([not set(nc2.split("_")) < nc3_set for nc3_set in nc3_sets])]
        return nc1s_umb + nc2s_umb + nc3s

    # Normalize scores so their sum across 6-digit NAICS codes equals one
    def __norm_scores(self, scores_raw):
        score_total = sum([scores_raw[naics] for naics in scores_raw])
        if score_total > 0.0:
            scores_raw = {naics: scores_raw[naics] / score_total for naics in scores_raw}
        return scores_raw

    # Calculate scores for either the "standard" or "umbrella" sub-model
    def __calc_scores_nonexact(self, feats, sector):
        scores_raw = {naics: 0.0 for naics in self.naics_indices_[sector]}
        for nc in feats:
            for naics in self.naics_indices_[sector]:
                scores_raw[naics] += self.dict_ncombs_weights_[sector][nc] * self.dict_ncombs_props_[sector][nc][self.naics_indices_[sector][naics]]
        return self.__norm_scores(scores_raw)

    # Calculate scores for the "exact" sub-model
    def __calc_scores_exact(self, feats, x_exact, sector):
        scores_raw = {naics: 0.0 for naics in self.naics_indices_[sector]}
        if x_exact in self.dict_ems_weights_[sector]:
            for naics in self.naics_indices_[sector]:
                scores_raw[naics] = self.dict_ems_props_[sector][x_exact][self.naics_indices_[sector][naics]]
        else:
            for em in feats:
                if em in self.dict_ems_weights_[sector]:
                    for naics in self.naics_indices_[sector]:
                        scores_raw[naics] += self.dict_ems_weights_[sector][em] * self.dict_ems_props_[sector][em][self.naics_indices_[sector][naics]]
        return self.__norm_scores(scores_raw)

    # Calculate ensemble scores
    def __calc_scores_ensemble(self, tokens, sector):
        nc1s = [nc for nc in self.__get_ncombs(tokens, 1) if nc in self.dict_ncombs_props_[sector]]
        nc2s = [nc for nc in self.__get_ncombs(tokens, 2) if nc in self.dict_ncombs_props_[sector]]
        nc3s = [nc for nc in self.__get_ncombs(tokens, 3) if nc in self.dict_ncombs_props_[sector]]
        x_exact      = "_".join(sorted(set(nc1s)))
        feats_stand  = nc1s + nc2s + nc3s
        feats_umb    = self.__get_feats_umb(nc1s, nc2s, nc3s)
        feats_exact  = feats_stand
        scores_stand = self.__calc_scores_nonexact(feats_stand, sector)
        scores_umb   = self.__calc_scores_nonexact(feats_umb, sector)
        scores_exact = self.__calc_scores_exact(feats_exact, x_exact, sector)
        # Calculate a weighted average of the "standard", "umbrella", and "exact" scores using the ensemble weights
        scores_ensemble = {naics: (1.0 - self.wt_umb - self.wt_exact) * scores_stand[naics] + self.wt_umb * scores_umb[naics] + self.wt_exact * scores_exact[naics] for naics in scores_stand}
        return self.__norm_scores(scores_ensemble)

    # Calculate hierarchical scores
    def __calc_scores_hier(self, x):
        x_clean = self.clean_text(x)
        tokens = x_clean.split(" ")
        scores_dict = {}
        scores_dict["00"] = self.__calc_scores_ensemble(tokens, "00")
        for sector in self.sectors_:
            if scores_dict["00"][sector] > 0.0:
                scores_dict[sector] = self.__calc_scores_ensemble(tokens, sector)
            else:
                scores_dict[sector] = {naics: 0.0 for naics in self.naics_indices_[sector]}
        scores_hier = {}
        for sector in self.sectors_:
            for naics in scores_dict[sector]:
                # Conditional probability formula
                scores_hier[naics] = scores_dict["00"][sector] * scores_dict[sector][naics]
        return self.__norm_scores(scores_hier)

    # Return dictionary of scores for all 6-digit NAICS codes
    # Main prediction method on which other prediction methods are based
    def predict_proba(self, X):
        # check_is_fitted() is provided by sklearn.utils.validation
        # check_is_fitted() checks for the existence of a BeaconModel attribute ending in "_" and raises NotFittedError accordingly
        check_is_fitted(self)
        return np.array([self.__calc_scores_hier(x) for x in X])

    # Given a dictionary of scores, return the highest-scoring NAICS code
    # Return an empty string if there is no NAICS code with a positive score
    def __argmax_scores(self, scores_hier):
        tups_hier = [(naics, scores_hier[naics]) for naics in scores_hier if scores_hier[naics] > 0.0]
        if len(tups_hier) == 0:
            return ""
        tups_hier.sort(key=lambda x: x[1], reverse=True)
        return tups_hier[0][0]

    # Given a dictionary of scores, return a list of the 10 highest-scoring NAICS codes
    # Pad the list with empty strings so it has length 10
    # Return a list of empty strings if there is no NAICS code with a positive score
    def __argmax_scores_top10(self, scores_hier):
        tups_hier = [(naics, scores_hier[naics]) for naics in scores_hier if scores_hier[naics] > 0.0]
        if len(tups_hier) == 0:
            return [""] * 10
        tups_hier.sort(key=lambda x: x[1], reverse=True)
        naics_top10 = [tup[0] for tup in tups_hier[:10]]
        if len(naics_top10) < 10:
            naics_top10 += [""] * (10 - len(naics_top10))
        return naics_top10

    # Return the highest-scoring NAICS code
    def predict(self, X):
        # check_is_fitted() is provided by sklearn.utils.validation
        # check_is_fitted() checks for the existence of a BeaconModel attribute ending in "_" and raises NotFittedError accordingly
        check_is_fitted(self)
        X = self.__validate_data_X(X)
        return np.array([self.__argmax_scores(scores_hier) for scores_hier in self.predict_proba(X)])

    # Return the 10 highest-scoring NAICS codes
    def predict_top10(self, X):
        # check_is_fitted() is provided by sklearn.utils.validation
        # check_is_fitted() checks for the existence of a BeaconModel attribute ending in "_" and raises NotFittedError accordingly
        check_is_fitted(self)
        X = self.__validate_data_X(X)
        return np.array([self.__argmax_scores_top10(scores_hier) for scores_hier in self.predict_proba(X)])

    # Note: BeaconModel inherits the method score() from the parent class ClassifierMixin
    #       score(X, y[, sample_weight]) returns the mean accuracy on the given test data 'X' and NAICS codes 'y' using optional sample weights 'sample_weight'

    # Print summary of the fitted model
    def summary(self):
        # check_is_fitted() is provided by sklearn.utils.validation
        # check_is_fitted() checks for the existence of a BeaconModel attribute ending in "_" and raises NotFittedError accordingly
        check_is_fitted(self)
        print("")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@    BeaconModel Summary    @@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("")
        print("Sample size:               {}".format(self.sample_sizes_["00"]))
        print("Number of sectors:         {}".format(self.n_sectors_))
        print("Number of NAICS codes:     {}".format(self.n_naics_))
        print("Dictionary size:           {}".format(len(self.dict_ncombs_weights_["00"]) + len(self.dict_ems_weights_["00"])))
        print("Frequency threshold:       {}".format(self.freq_thresh))
        print("Umbrella ensemble weight:  {}".format(self.wt_umb))
        print("Exact ensemble weight:     {}".format(self.wt_exact))
        print("")
        return
