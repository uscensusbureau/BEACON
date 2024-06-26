# BEACON

## Introduction

BEACON (Business Establishment Automated Classification of NAICS) is a machine learning tool developed to help respondents self-designate their 6-digit NAICS (North American Industry Classification System) code on the Economic Census (EC). BEACON’s methodology is based on machine learning, natural language processing and information retrieval.

The EC is conducted every five years. In particular, the years ending in “2” or “7”. This survey represents approximately eight million establishments, covering most industries and all geographic areas of the United States.

The NAICS is a hierarchical 6-digit coding structure. The first two digits represent the economic sector and the additional non-zero digits as industry detail. The U.S. Census Bureau classifies establishments by NAICS industry based on the primary business of activity of the establishment. NAICS is utilized throughout the survey life cycle: sample selection, data collection, analytical review, and publication.

On the EC, respondents are asked to describe their business. There are prelisted descriptions corresponding to a suggested  NAICS code, but the respondent can also type in a description. Clerical analysis of this write-in text is a resource intensive process.

The general idea of BEACON is the respondent inputs a business description and BEACON returns a ranked list of 6-digit NAICS code with matching industry descriptions.

1. Respondent provides write-in description.
2. Text is outputted to BEACON API.
3. API returns most relevant NAICS codes to respondent.

The goals are to help respondents properly self-designate their NAICS code, send respondents down correct EC questionnaire path, and reduce clerical work associated with write-ins.

This makes the questionnaire more dynamic. Overall, BEACON leads to less clerical work associated with analyzing NAICS write-ins.

## Usage

First step is the text cleaning process: convert to lower and account for numbers and punctuation, remove common “stop” words, stem words to reduce the number of word variation and correct common misspellings.

Example: Input Text: This is a convenience store. Clean Text: conveni store

Underlying BEACON is a dictionary of text that occurs frequently in the cleaned training data. It consists of words, word combinations, and full-length/exact descriptions. These pieces of text serve as the model features. These features contain NAICS distributions and associated purity weights that measure how concentrated, or pure, the distribution is for each word or word combination.

Information retrieval models look at how words, word combinations and entire descriptions are distributed across NAICS codes. Each type (word, word combination and entire description) has their relevant scores calculated by using their NAICS distribution and their purity rates.  The individual scores are averaged, yielding relevance scores. These relevance scores range in value between 0 and 100. The scores reflect how confident BEACON is that the NAICS code is correct.

## Repository Contents

This section serves as a guide to the repository contents.

## Credits

For more information, please see BEACON conference presentations and papers attached to the repository. If you have any questions or comments, please reach out to @uscensusbureau/BEACON team.

Brian Dumbacher, [@brian-dumbacher](https://www.github.com/brian-dumbacher)

Daniel Whitehead, [@DanWhiteheadCensus](https://www.github.com/DanWhiteheadCensus)

Sarah Pfeiff, [@sdpfeiff](https://www.github.com/sdpfeiff)

## References

Economic Census: [https://www.census.gov/programs-surveys/economic-census.html](https://www.census.gov/programs-surveys/economic-census.html)

NAICS: [https://www.census.gov/naics/](https://www.census.gov/naics/)

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
