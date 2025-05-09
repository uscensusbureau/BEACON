# BEACON

<p>
<img src="https://img.shields.io/badge/repo_status-active-brightgreen?style=for-the-badge" alt="Repo Status: Active">
<img src="https://img.shields.io/badge/license-cc0_1.0_universal-brightgreen?style=for-the-badge" alt="License: CC0 1.0 Universal">
<img src="https://img.shields.io/badge/python-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/scikit--learn-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white&logoSize=auto" alt="scikit-learn">
<img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas">
</p>

## Introduction

BEACON (Business Establishment Automated Classification of NAICS) is a machine learning tool developed to help respondents self-designate their 6-digit NAICS (North American Industry Classification System) code on the Economic Census (EC). BEACON’s methodology is based on machine learning, natural language processing, and information retrieval.

The EC is conducted every five years. In particular, the years ending in “2” or “7”. This survey represents approximately eight million establishments, covering most industries and all geographic areas of the United States.

The NAICS is a hierarchical 6-digit coding structure. The first two digits represent the economic sector and the additional non-zero digits as industry detail. The U.S. Census Bureau classifies establishments by NAICS industry based on the primary business of activity of the establishment. NAICS is utilized throughout the survey life cycle: sample selection, data collection, analytical review, and publication.

On the EC, respondents are asked to describe their business. There are prelisted descriptions corresponding to a suggested NAICS code, but the respondent can also type in a description. Clerical analysis of this write-in text is a resource intensive process.

The general idea of BEACON is the respondent inputs a business description and BEACON returns a ranked list of 6-digit NAICS code with matching industry descriptions.

1. Respondent provides write-in description.
2. Text is outputted to BEACON API.
3. API returns most relevant NAICS codes to respondent.

The goals are to help respondents properly self-designate their NAICS code, send respondents down correct EC questionnaire path, and reduce clerical work associated with write-ins.

This makes the questionnaire more dynamic. Overall, BEACON leads to less clerical work associated with analyzing NAICS write-ins.

## Usage

First step is the text cleaning process: convert to lower and account for numbers and punctuation, remove common “stop” words, stem words to reduce the number of word variations, and correct common misspellings.

Example: Input Text: ```This is a convenience store.``` Clean Text: ```conveni store```

Underlying BEACON is a dictionary of text that occurs frequently in the cleaned training data. It consists of words, word combinations, and full-length/exact descriptions. These pieces of text serve as the model features. These features contain NAICS distributions and associated purity weights that measure how concentrated, or pure, the distribution is for each word or word combination.

Information retrieval models look at how words, word combinations, and entire descriptions are distributed across NAICS codes. Each type (word, word combination, and entire description) has their relevant scores calculated by using their NAICS distribution and their purity weights. The individual scores are averaged, yielding relevance scores. These relevance scores range in value between 0 and 100. The scores reflect how confident BEACON is that the NAICS code is correct.

## Repository Contents

This section serves as a guide to the repository contents.

| File                                        | Description                                                    |
| ------------------------------------------- | -------------------------------------------------------------- |
| ```create_example_data.py```                | Program for creating example datasets using public NAICS files |
| ```create_example_data_output.txt```        | Output of ```create_example_data.py```                         |
| ```beacon.py```                             | Codebase for implementing a simplified version of BEACON       |
| ```beacon_example.py```                     | Program for illustrating the use of ```beacon.py```            |
| ```beacon_example_output.txt```             | Output of ```beacon_example.py```                              |
| ```eurostat_BEACON_Whitehead_Pfeiff.pdf```  | 2024 Eurostat industry coding webinar presentation on BEACON   |
| ```2023-FCSM-BEACON-Model-Stacking.pdf```   | 2023 FCSM presentation on BEACON and applying model stacking   |
| ```2022-FCSM-Wiley-Whitehead.pdf```         | 2022 FCSM presentation on BEACON and SINCT                     |
| ```JSM_Dumbacher_Whitehead.pdf```           | 2022 JSM presentation on BEACON                                |

## Credits

For more information, please see BEACON conference presentations and papers attached to the repository. If you have any questions or comments, please reach out to the BEACON team:

* Brian Dumbacher ([@brian-dumbacher](https://www.github.com/brian-dumbacher))
* Daniel Whitehead ([@DanWhiteheadCensus](https://www.github.com/DanWhiteheadCensus))
* Sarah Pfeiff ([@sdpfeiff](https://www.github.com/sdpfeiff))

## References

* Dumbacher, B., Whitehead, D., Jeong, J., and Pfeiff, S. (2025). <b>BEACON: A Tool for Industry Self-Classification in the Economic Census</b>. <i>Journal of Data Science</i>, <i>23</i>(2): 429–448. [https://doi.org/10.6339/25-JDS1180](https://doi.org/10.6339/25-JDS1180)
* Dumbacher, B. and Whitehead, D. (2024). <b>Industry Self-Classification in the Economic Census</b>. <i>U.S. Census Bureau ADEP Working Paper Series</i>, ADEP-WP-2024-04. [https://www2.census.gov/library/working-papers/2024/econ/industry-self-classification-economic-census.pdf](https://www2.census.gov/library/working-papers/2024/econ/industry-self-classification-economic-census.pdf)
* Dumbacher, B. and Whitehead, D. (2024). <b>Ranked short text classification using co-occurrence features and score functions</b>. <i>U.S. Census Bureau ADEP Working Paper Series</i>, ADEP-WP-2024-06. [https://www2.census.gov/library/working-papers/2024/econ/ranked-short-text-classification-using-co-occurrence-features-and-score-functions.pdf](https://www2.census.gov/library/working-papers/2024/econ/ranked-short-text-classification-using-co-occurrence-features-and-score-functions.pdf)
* U.S. Census Bureau. (2024). <b>Economic Census</b>. Online; accessed 5 August 2024. [https://www.census.gov/programs-surveys/economic-census.html](https://www.census.gov/programs-surveys/economic-census.html)
* U.S. Census Bureau. (2024). <b>North American Industry Classification System</b>. Online; accessed 5 August 2024. [https://www.census.gov/naics/](https://www.census.gov/naics/)
* Whitehead, D. and Dumbacher, B. (2024). <b>Ensemble Modeling Techniques for NAICS Classification in the Economic Census</b>. <i>U.S. Census Bureau ADEP Working Paper Series</i>, ADEP-WP-2024-03. [https://www2.census.gov/library/working-papers/2024/econ/ensemble-modeling-techniques-for-naics-classification-economic-census.pdf](https://www2.census.gov/library/working-papers/2024/econ/ensemble-modeling-techniques-for-naics-classification-economic-census.pdf)
* Wiley, E. and Whitehead, D. (2024). <b>Implementing Interactive Classification Tools in the 2022 Economic Census</b>. <i>U.S. Census Bureau ADEP Working Paper Series</i>, ADEP-WP-2024-05. [https://www2.census.gov/library/working-papers/2024/econ/implementing-interactive-classification-tools-2022-economic-census.pdf](https://www2.census.gov/library/working-papers/2024/econ/implementing-interactive-classification-tools-2022-economic-census.pdf)

# License

As a work of the United States government, this project is in the public domain within the United States.

Additionally, we waive copyright and related rights in the work worldwide through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).

## CC0 1.0 Universal Summary

This is a human-readable summary of the
[Legal Code (read the full text)](https://creativecommons.org/publicdomain/zero/1.0/legalcode).

### No copyright

The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.

You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

### Other information

In no way are the patent or trademark rights of any person affected by CC0, nor are the rights that other persons may have in the work or in how the work is used, such as publicity or privacy rights.

Unless expressly stated otherwise, the person who associated a work with this deed makes no warranties about the work, and disclaims liability for all uses of the work, to the fullest extent permitted by applicable law. When using or citing the work, you should not imply endorsement by the author or the affirmer.
