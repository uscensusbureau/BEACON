import io
import pandas as pd

def format_naics(naics_temp):
    return str(int(naics_temp))

def format_descr(descr_temp):
    return str(descr_temp).upper().strip()

def get_sector(naics):
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

def main():
    print("Purpose:  Create pipe-delimited example dataset using publicly available 2017 NAICS files")
    print("          Create pipe-delimited example dataset using publicly available 2022 NAICS files")
    print("")
    print("Note:     Download the following files before running this program")
    print("")
    print("Sources:  https://www.census.gov/naics/2017NAICS/2017_NAICS_Index_File.xlsx")
    print("          https://www.census.gov/naics/2022NAICS/2022_NAICS_Index_File.xlsx")
    print("          https://www.census.gov/naics/2017NAICS/6-digit_2017_Codes.xlsx")
    print("          https://www.census.gov/naics/2022NAICS/6-digit_2022_Codes.xlsx")
    print("")

    # 2017 data WITH sample weights
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@    2017 data WITH sample weights    @@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("")

    index17_df = pd.read_excel("2017_NAICS_Index_File.xlsx")
    index17_tups = []
    for i in range(len(index17_df)):
        naics_temp = index17_df.at[i, "NAICS17"]
        descr_temp = index17_df.at[i, "INDEX ITEM DESCRIPTION"]
        if not (pd.isna(naics_temp) or naics_temp == "******" or pd.isna(descr_temp)):
            index17_tups.append((format_naics(naics_temp), format_descr(descr_temp)))

    codes17_df = pd.read_excel("6-digit_2017_Codes.xlsx")
    codes17_tups = []
    for i in range(len(codes17_df)):
        naics_temp = codes17_df.at[i, "2017 NAICS Code"]
        descr_temp = codes17_df.at[i, "2017 NAICS Title"]
        if not (pd.isna(naics_temp) or pd.isna(descr_temp)):
            codes17_tups.append((format_naics(naics_temp), format_descr(descr_temp)))

    combined17_tups = sorted(list(set(index17_tups + codes17_tups)))

    print("The sample weights are for illustration only and are based loosely on the number of")
    print("    establishments in each 6-digit NAICS code according to the 2017 Economic Census")
    print("    (Source: EC1700BASIC table available at https://data.census.gov)")
    print("")

    weights17 = {
        "111110": "813.23",
        "111120": "180.72",
        "111130": "232.35",
        "111140": "813.23",
        "111150": "542.15",
        "111160": "813.23",
        "111191": "813.23",
        "111199": "180.72",
        "111211": "271.08",
        "111219": "23.92",
        "111310": "1626.46",
        "111320": "203.31",
        "111331": "1626.46",
        "111332": "542.15",
        "111333": "1626.46",
        "111334": "135.54",
        "111335": "162.65",
        "111336": "1626.46",
        "111339": "56.08",
        "111411": "406.61",
        "111419": "77.45",
        "111421": "85.6",
        "111422": "108.43",
        "111910": "813.23",
        "111920": "542.15",
        "111930": "542.15",
        "111940": "271.08",
        "111991": "1626.46",
        "111992": "1626.46",
        "111998": "54.22",
        "112111": "162.65",
        "112112": "271.08",
        "112120": "406.61",
        "112130": "1626.46",
        "112210": "180.72",
        "112310": "232.35",
        "112320": "271.08",
        "112330": "542.15",
        "112340": "325.29",
        "112390": "147.86",
        "112410": "271.08",
        "112420": "271.08",
        "112511": "116.18",
        "112512": "135.54",
        "112519": "180.72",
        "112910": "203.31",
        "112920": "271.08",
        "112930": "271.08",
        "112990": "73.93",
        "113110": "1626.46",
        "113210": "67.77",
        "113310": "147.86",
        "114111": "41.7",
        "114112": "116.18",
        "114119": "271.08",
        "114210": "232.35",
        "115111": "813.23",
        "115112": "54.22",
        "115113": "116.18",
        "115114": "37.82",
        "115115": "325.29",
        "115116": "813.23",
        "115210": "54.22",
        "115310": "135.54",
        "211120": "274.01",
        "211130": "64.26",
        "212111": "75.42",
        "212112": "257.48",
        "212113": "111.74",
        "212210": "102.56",
        "212221": "145.22",
        "212222": "408.64",
        "212230": "62.05",
        "212291": "135.54",
        "212299": "25.41",
        "212311": "58.38",
        "212312": "157.91",
        "212313": "230.2",
        "212319": "70.48",
        "212321": "348.07",
        "212322": "117.44",
        "212324": "183.23",
        "212325": "47.04",
        "212391": "58.55",
        "212392": "272.29",
        "212393": "50.17",
        "212399": "20.16",
        "213111": "159.21",
        "213112": "76.51",
        "213113": "66.65",
        "213114": "98.98",
        "213115": "111.68",
        "221111": "669.07",
        "221112": "892.66",
        "221113": "587.93",
        "221114": "446.35",
        "221115": "653.74",
        "221116": "551.29",
        "221117": "579.11",
        "221118": "551.82",
        "221121": "458.77",
        "221122": "1215.97",
        "221210": "218.89",
        "221310": "334.18",
        "221320": "346.51",
        "221330": "123.22",
        "236115": "477.63",
        "236116": "196.07",
        "236117": "192.33",
        "236118": "571.45",
        "236210": "72.15",
        "236220": "103",
        "237110": "174.88",
        "237120": "116.85",
        "237130": "139.58",
        "237210": "329.03",
        "237310": "136.55",
        "237990": "48.44",
        "238110": "473.61",
        "238120": "217.48",
        "238130": "359.63",
        "238140": "340.51",
        "238150": "353.86",
        "238160": "431.76",
        "238170": "371.4",
        "238190": "189.76",
        "238210": "293.27",
        "238220": "212.93",
        "238290": "98.11",
        "238310": "193.41",
        "238320": "614.01",
        "238330": "446.08",
        "238340": "445.16",
        "238350": "290.69",
        "238390": "151.16",
        "238910": "160.9",
        "238990": "190.13",
        "311111": "159.62",
        "311119": "58.67",
        "311211": "53.69",
        "311212": "105.64",
        "311213": "138.49",
        "311221": "58.12",
        "311224": "63.05",
        "311225": "68.23",
        "311230": "99.66",
        "311313": "103.12",
        "311314": "83.41",
        "311340": "64.75",
        "311351": "99.15",
        "311352": "148.6",
        "311411": "119.03",
        "311412": "90.42",
        "311421": "52.57",
        "311422": "86.1",
        "311423": "90.23",
        "311511": "65.08",
        "311512": "333.32",
        "311513": "169.45",
        "311514": "50.46",
        "311520": "120.92",
        "311611": "86.38",
        "311612": "72.05",
        "311613": "119.32",
        "311615": "80.26",
        "311710": "72.37",
        "311811": "2287.76",
        "311812": "156.04",
        "311813": "163.72",
        "311821": "190.79",
        "311824": "80.73",
        "311830": "957.06",
        "311911": "227.85",
        "311919": "214.67",
        "311920": "118.57",
        "311930": "134.4",
        "311941": "99.98",
        "311942": "80.76",
        "311991": "121.77",
        "311999": "58.99",
        "312111": "106.54",
        "312112": "370.03",
        "312113": "641.48",
        "312120": "305.75",
        "312130": "168.07",
        "312140": "167.06",
        "312230": "125.45",
        "313110": "29.36",
        "313210": "37.8",
        "313220": "46.42",
        "313230": "90.08",
        "313240": "64.4",
        "313310": "56.86",
        "313320": "87.78",
        "314110": "227.67",
        "314120": "132.34",
        "314910": "120.38",
        "314994": "158.08",
        "314999": "51.65",
        "315110": "71.98",
        "315190": "21.04",
        "315210": "12.87",
        "315220": "17.64",
        "315240": "16.22",
        "315280": "29.6",
        "315990": "35.81",
        "316110": "28.02",
        "316210": "19.97",
        "316992": "424.91",
        "316998": "18.11",
        "321113": "118.83",
        "321114": "74.41",
        "321211": "200.91",
        "321212": "188.33",
        "321213": "122.51",
        "321214": "363.69",
        "321219": "89.2",
        "321911": "124.55",
        "321912": "56.46",
        "321918": "74.47",
        "321920": "49.85",
        "321991": "258.32",
        "321992": "139.03",
        "321999": "24.88",
        "322110": "206.83",
        "322121": "42.85",
        "322122": "148.52",
        "322130": "97.15",
        "322211": "202.67",
        "322212": "244.65",
        "322219": "80.29",
        "322220": "51.24",
        "322230": "76.37",
        "322291": "133.11",
        "322299": "78.21",
        "323111": "67.24",
        "323113": "76.2",
        "323117": "66.91",
        "323120": "61.08",
        "324110": "34.17",
        "324121": "312.13",
        "324122": "89.68",
        "324191": "133.58",
        "324199": "140.4",
        "325110": "59.32",
        "325120": "86.9",
        "325130": "27.63",
        "325180": "10.45",
        "325193": "357.38",
        "325194": "16.7",
        "325199": "14.24",
        "325211": "28.23",
        "325212": "41.62",
        "325220": "34.59",
        "325311": "137.05",
        "325312": "187.2",
        "325314": "247.17",
        "325320": "40.96",
        "325411": "32.57",
        "325412": "32.87",
        "325413": "129.47",
        "325414": "82.26",
        "325510": "75.49",
        "325520": "119.27",
        "325611": "131.52",
        "325612": "43.69",
        "325613": "133.29",
        "325620": "53.33",
        "325910": "117.8",
        "325920": "73.93",
        "325991": "385.71",
        "325992": "81.22",
        "325998": "27.53",
        "326111": "268.66",
        "326112": "480.08",
        "326113": "167.27",
        "326121": "317.86",
        "326122": "326.87",
        "326130": "300.88",
        "326140": "130.16",
        "326150": "174.37",
        "326160": "985.99",
        "326191": "145.26",
        "326199": "51.9",
        "326211": "191.17",
        "326212": "381.58",
        "326220": "96.74",
        "326291": "482.66",
        "326299": "51.3",
        "327110": "28.01",
        "327120": "38.8",
        "327211": "283.92",
        "327212": "47.09",
        "327213": "186.32",
        "327215": "53.92",
        "327310": "443.94",
        "327320": "705.48",
        "327331": "191.93",
        "327332": "179.72",
        "327390": "107.37",
        "327410": "189.11",
        "327420": "68.38",
        "327910": "102.91",
        "327991": "148.02",
        "327992": "82.92",
        "327993": "364.85",
        "327999": "107.79",
        "331110": "20.09",
        "331210": "258.94",
        "331221": "104.63",
        "331222": "72.18",
        "331313": "109.94",
        "331314": "99.75",
        "331315": "80.38",
        "331318": "32.61",
        "331410": "36.74",
        "331420": "36.35",
        "331491": "30.05",
        "331492": "54.33",
        "331511": "69.47",
        "331512": "428.78",
        "331513": "222.8",
        "331523": "101",
        "331524": "274.63",
        "331529": "91.32",
        "332111": "146.76",
        "332112": "185.71",
        "332114": "643.31",
        "332117": "866.38",
        "332119": "251.29",
        "332215": "44.73",
        "332216": "20.14",
        "332311": "126.17",
        "332312": "146.04",
        "332313": "62.31",
        "332321": "89.46",
        "332322": "79.91",
        "332323": "69.89",
        "332410": "93.54",
        "332420": "65.05",
        "332431": "126.52",
        "332439": "73.86",
        "332510": "86.23",
        "332613": "86.72",
        "332618": "39.99",
        "332710": "2376.64",
        "332721": "3568.58",
        "332722": "113.98",
        "332811": "217.18",
        "332812": "146.79",
        "332813": "181.36",
        "332911": "66.16",
        "332912": "105.26",
        "332913": "131.58",
        "332919": "100.58",
        "332991": "125.45",
        "332992": "87.28",
        "332993": "52.13",
        "332994": "33.85",
        "332996": "164.32",
        "332999": "29.14",
        "333111": "35.55",
        "333112": "73.15",
        "333120": "35.86",
        "333131": "53.89",
        "333132": "188.72",
        "333241": "43.76",
        "333242": "174.8",
        "333243": "52.82",
        "333244": "87.63",
        "333249": "25.33",
        "333314": "56.9",
        "333316": "55.35",
        "333318": "28.5",
        "333413": "131.82",
        "333414": "68.66",
        "333415": "63.1",
        "333511": "317.77",
        "333514": "198.25",
        "333515": "65.63",
        "333517": "24.13",
        "333519": "74.23",
        "333611": "109.63",
        "333612": "199.69",
        "333613": "120.89",
        "333618": "103.97",
        "333912": "231.54",
        "333914": "157.51",
        "333921": "251.86",
        "333922": "166.92",
        "333923": "116.04",
        "333924": "78.3",
        "333991": "64.09",
        "333992": "111.57",
        "333993": "143.18",
        "333994": "85.68",
        "333995": "266.55",
        "333996": "194.22",
        "333997": "130.26",
        "333999": "120.62",
        "334111": "123.87",
        "334112": "141.57",
        "334118": "88.1",
        "334210": "77.72",
        "334220": "61.07",
        "334290": "81.58",
        "334310": "70.57",
        "334412": "492.66",
        "334413": "62.62",
        "334416": "190.72",
        "334417": "254.74",
        "334418": "243.13",
        "334419": "71.77",
        "334510": "41.11",
        "334511": "34.34",
        "334512": "32.21",
        "334513": "29.98",
        "334514": "42.35",
        "334515": "24.46",
        "334516": "32.96",
        "334517": "87.02",
        "334519": "19.36",
        "334613": "119.44",
        "334614": "137.12",
        "335110": "45.45",
        "335121": "87.01",
        "335122": "93.65",
        "335129": "63.45",
        "335210": "18.89",
        "335220": "62.64",
        "335311": "47.7",
        "335312": "49.75",
        "335313": "66",
        "335314": "96.81",
        "335911": "146.04",
        "335912": "112.21",
        "335921": "849.04",
        "335929": "360.91",
        "335931": "47.96",
        "335932": "124.75",
        "335991": "87.32",
        "335999": "53.71",
        "336111": "219.91",
        "336112": "128.87",
        "336120": "140.99",
        "336211": "71.43",
        "336212": "162.31",
        "336213": "416.06",
        "336214": "153.38",
        "336310": "80.1",
        "336320": "64.75",
        "336330": "152.2",
        "336340": "117.94",
        "336350": "123.37",
        "336360": "129.35",
        "336370": "239.48",
        "336390": "85.79",
        "336411": "143.36",
        "336412": "148.83",
        "336413": "161.65",
        "336414": "331.73",
        "336415": "330.13",
        "336419": "331.57",
        "336510": "106.11",
        "336611": "78.06",
        "336612": "110.85",
        "336991": "277.95",
        "336992": "416.26",
        "336999": "160.54",
        "337110": "390.64",
        "337121": "101.52",
        "337122": "49.73",
        "337124": "45.9",
        "337125": "83.2",
        "337127": "64.31",
        "337211": "186.79",
        "337212": "575.68",
        "337214": "179.72",
        "337215": "87.31",
        "337910": "241.41",
        "337920": "144.23",
        "339112": "51.69",
        "339113": "32.5",
        "339114": "72.9",
        "339115": "152.92",
        "339116": "859.66",
        "339910": "56.12",
        "339920": "47.12",
        "339930": "46.37",
        "339940": "34.94",
        "339950": "526.21",
        "339991": "225.5",
        "339992": "42.09",
        "339993": "131.75",
        "339994": "160.75",
        "339995": "341.17",
        "339999": "75.34",
        "423110": "138.37",
        "423120": "149.32",
        "423130": "345.69",
        "423140": "384.82",
        "423210": "226.69",
        "423220": "123.32",
        "423310": "140.08",
        "423320": "117.31",
        "423330": "169.46",
        "423390": "116.57",
        "423410": "99.79",
        "423420": "159.35",
        "423430": "163.76",
        "423440": "98.83",
        "423450": "121.52",
        "423460": "169.79",
        "423490": "109.85",
        "423510": "85.92",
        "423520": "110.71",
        "423610": "119.42",
        "423620": "54.34",
        "423690": "92.23",
        "423710": "201.84",
        "423720": "113.8",
        "423730": "230.02",
        "423740": "121.44",
        "423810": "171.59",
        "423820": "164.41",
        "423830": "79.97",
        "423840": "85.01",
        "423850": "94.85",
        "423860": "97.59",
        "423910": "93.29",
        "423920": "125.49",
        "423930": "223.09",
        "423940": "190.36",
        "423990": "115.63",
        "424110": "317.99",
        "424120": "84.19",
        "424130": "91.19",
        "424210": "95.82",
        "424310": "104.58",
        "424320": "144.73",
        "424330": "129.29",
        "424340": "360.93",
        "424410": "768.92",
        "424420": "189.24",
        "424430": "148.81",
        "424440": "177.55",
        "424450": "163.57",
        "424460": "312.12",
        "424470": "269.78",
        "424480": "491.99",
        "424490": "86.78",
        "424510": "440.22",
        "424520": "261.71",
        "424590": "59.58",
        "424610": "269.55",
        "424690": "69.58",
        "424710": "181.59",
        "424720": "159.83",
        "424810": "344.72",
        "424820": "222.38",
        "424910": "153.11",
        "424920": "213.14",
        "424930": "344.49",
        "424940": "286.25",
        "424950": "146.1",
        "424990": "77.2",
        "425110": "286.65",
        "425120": "1435.9",
        "441110": "2579.47",
        "441120": "2092.38",
        "441210": "346.06",
        "441222": "464.64",
        "441228": "254.37",
        "441310": "841.8",
        "441320": "1253.3",
        "442110": "1345.26",
        "442210": "1418.61",
        "442291": "1364.67",
        "442299": "525.78",
        "443141": "940.65",
        "443142": "420.05",
        "444110": "1431.01",
        "444120": "1534.2",
        "444130": "2171.86",
        "444190": "407.71",
        "444210": "912.03",
        "444220": "887.39",
        "445110": "2204.58",
        "445120": "4413.97",
        "445210": "669.84",
        "445220": "934.8",
        "445230": "446.39",
        "445291": "1585.27",
        "445292": "856.89",
        "445299": "335.38",
        "445310": "1613.76",
        "446110": "2214.8",
        "446120": "1727.8",
        "446130": "1478.98",
        "446191": "1083.27",
        "446199": "768.24",
        "447110": "5415.77",
        "447190": "1363.94",
        "448110": "1515.99",
        "448120": "2377.35",
        "448130": "1140.48",
        "448140": "2218.15",
        "448150": "456.53",
        "448190": "370.12",
        "448210": "1378.07",
        "448310": "1535.53",
        "448320": "764.42",
        "451110": "427.8",
        "451120": "716.42",
        "451130": "610.37",
        "451140": "694.4",
        "451211": "2231.72",
        "451212": "457.03",
        "452210": "3756.38",
        "452311": "1233.46",
        "452319": "1321.36",
        "453110": "3005.65",
        "453210": "1049.27",
        "453220": "516.46",
        "453310": "357.46",
        "453910": "1347.61",
        "453920": "983.47",
        "453930": "938.72",
        "453991": "1373.79",
        "453998": "275.32",
        "454110": "530.71",
        "454210": "1180.35",
        "454310": "543.09",
        "454390": "406.86",
        "481111": "307.59",
        "481112": "238.23",
        "481211": "284.94",
        "481212": "178.47",
        "481219": "661.78",
        "482111": "271.08",
        "482112": "203.31",
        "483111": "305.13",
        "483112": "280.01",
        "483113": "170.42",
        "483114": "158.29",
        "483211": "169.17",
        "483212": "183.51",
        "484110": "1545.75",
        "484121": "1667.24",
        "484122": "995.2",
        "484210": "1000.18",
        "484220": "381.59",
        "484230": "216.15",
        "485111": "236.27",
        "485112": "234.2",
        "485113": "231.15",
        "485119": "186.32",
        "485210": "292.98",
        "485310": "478.38",
        "485320": "490.56",
        "485410": "943.92",
        "485510": "823.64",
        "485991": "570.68",
        "485999": "448.55",
        "486110": "437.76",
        "486210": "361.41",
        "486910": "298.19",
        "486990": "336.48",
        "487110": "126.31",
        "487210": "211.69",
        "487990": "228.12",
        "488111": "887.13",
        "488119": "177.36",
        "488190": "326.67",
        "488210": "235.23",
        "488310": "157.36",
        "488320": "330.11",
        "488330": "258.12",
        "488390": "312.16",
        "488410": "865.58",
        "488490": "210.08",
        "488510": "1285.86",
        "488991": "660.67",
        "488999": "205.93",
        "491110": "271.08",
        "492110": "1085.66",
        "492210": "469.28",
        "493110": "991.82",
        "493120": "246.72",
        "493130": "290.3",
        "493190": "313.15",
        "511110": "777.66",
        "511120": "90.6",
        "511130": "72.05",
        "511140": "134.78",
        "511191": "343.18",
        "511199": "77.58",
        "511210": "474.29",
        "512110": "318.64",
        "512120": "188.77",
        "512131": "380.5",
        "512132": "295.83",
        "512191": "161.88",
        "512199": "103.88",
        "512230": "193.3",
        "512240": "550.41",
        "512250": "192.22",
        "512290": "216.71",
        "515111": "269.41",
        "515112": "352.51",
        "515120": "479.05",
        "515210": "212.44",
        "517311": "505.78",
        "517312": "467.22",
        "517410": "278.92",
        "517911": "370.78",
        "517919": "264.22",
        "518210": "319.64",
        "519110": "176.32",
        "519120": "269.53",
        "519130": "57.84",
        "519190": "331.7",
        "521110": "185.8",
        "522110": "2241.47",
        "522120": "147.86",
        "522130": "1466.05",
        "522190": "232.35",
        "522210": "418.86",
        "522220": "336.98",
        "522291": "794.36",
        "522292": "649.93",
        "522293": "195.14",
        "522294": "90.36",
        "522298": "95.67",
        "522310": "477.61",
        "522320": "250.94",
        "522390": "953.23",
        "523110": "187.41",
        "523120": "616.68",
        "523130": "182.35",
        "523140": "190.72",
        "523210": "206.93",
        "523910": "502.39",
        "523920": "977.12",
        "523930": "1042.63",
        "523991": "325.99",
        "523999": "165.89",
        "524113": "335.61",
        "524114": "412.14",
        "524126": "254.63",
        "524127": "599.44",
        "524128": "199.2",
        "524130": "320.4",
        "524210": "2725.42",
        "524291": "480.12",
        "524292": "544.07",
        "524298": "240.4",
        "525110": "203.31",
        "525120": "325.29",
        "525190": "325.29",
        "525910": "271.08",
        "525920": "203.31",
        "525990": "101.65",
        "531110": "730.57",
        "531120": "256.73",
        "531130": "678.99",
        "531190": "254.29",
        "531210": "751.31",
        "531311": "770.76",
        "531312": "474.26",
        "531320": "1227.56",
        "531390": "480.21",
        "532111": "537.38",
        "532112": "252.13",
        "532120": "260.16",
        "532210": "287.86",
        "532281": "161.94",
        "532282": "333.01",
        "532283": "286.52",
        "532284": "128.99",
        "532289": "280.99",
        "532310": "1068.47",
        "532411": "135.54",
        "532412": "277.12",
        "532420": "182.85",
        "532490": "127.86",
        "533110": "286.71",
        "541110": "893.87",
        "541120": "1626.46",
        "541191": "1052.11",
        "541199": "404.74",
        "541211": "882.21",
        "541213": "2264.84",
        "541214": "1405.94",
        "541219": "956.12",
        "541310": "1114.22",
        "541320": "191.12",
        "541330": "460.61",
        "541340": "770.1",
        "541350": "639.2",
        "541360": "158.23",
        "541370": "300.82",
        "541380": "135.79",
        "541410": "1022.93",
        "541420": "371.18",
        "541430": "301.04",
        "541490": "272.4",
        "541511": "1218.56",
        "541512": "696.09",
        "541513": "963.27",
        "541519": "1830.12",
        "541611": "1205.53",
        "541612": "406.86",
        "541613": "2015.42",
        "541614": "262.1",
        "541618": "1573.18",
        "541620": "909.22",
        "541690": "364.61",
        "541713": "1256.93",
        "541714": "157.85",
        "541715": "162.71",
        "541720": "185.32",
        "541810": "3116.06",
        "541820": "832.78",
        "541830": "1146.91",
        "541840": "273.71",
        "541850": "221.22",
        "541860": "763.2",
        "541870": "192.73",
        "541890": "598.32",
        "541910": "448.24",
        "541921": "610.17",
        "541922": "531.32",
        "541930": "386.02",
        "541940": "443.48",
        "541990": "295.68",
        "551111": "948.53",
        "551112": "459.29",
        "551114": "1430.59",
        "561110": "686.62",
        "561210": "501.86",
        "561311": "139.17",
        "561312": "863.46",
        "561320": "719.83",
        "561330": "507.41",
        "561410": "238.3",
        "561421": "302.37",
        "561422": "365.93",
        "561431": "653.51",
        "561439": "373.91",
        "561440": "399.99",
        "561450": "199.46",
        "561491": "1095.9",
        "561492": "418.93",
        "561499": "441.84",
        "561510": "3126.54",
        "561520": "806.7",
        "561591": "382.4",
        "561599": "120.9",
        "561611": "284.92",
        "561612": "438.29",
        "561613": "2200.23",
        "561621": "514.27",
        "561622": "718.08",
        "561710": "800.01",
        "561720": "579",
        "561730": "417.25",
        "561740": "338.57",
        "561790": "311.67",
        "561910": "176.41",
        "561920": "136.28",
        "561990": "203.97",
        "562111": "324.01",
        "562112": "325.18",
        "562119": "219.18",
        "562211": "123.46",
        "562212": "181.25",
        "562213": "167.53",
        "562219": "307.29",
        "562910": "217.78",
        "562920": "419.16",
        "562991": "346.82",
        "562998": "217.26",
        "611110": "54.22",
        "611210": "180.72",
        "611310": "70.72",
        "611410": "241.8",
        "611420": "383.06",
        "611430": "904.29",
        "611511": "267.06",
        "611512": "325.21",
        "611513": "219.66",
        "611519": "96.18",
        "611610": "281.44",
        "611620": "202.42",
        "611630": "433.98",
        "611691": "526.2",
        "611692": "766.98",
        "611699": "298.1",
        "611710": "390.3",
        "621111": "565.59",
        "621112": "506.38",
        "621210": "1191.76",
        "621310": "2583.04",
        "621320": "1966.52",
        "621330": "1192.94",
        "621340": "477.01",
        "621391": "817.91",
        "621399": "222.11",
        "621410": "328.79",
        "621420": "534.37",
        "621491": "660.79",
        "621492": "962.83",
        "621493": "601.46",
        "621498": "695.87",
        "621511": "275.61",
        "621512": "191.3",
        "621610": "956.86",
        "621910": "689.3",
        "621991": "216.95",
        "621999": "314.12",
        "622110": "804",
        "622210": "139.49",
        "622310": "148.2",
        "623110": "687.34",
        "623210": "1071.37",
        "623220": "324.95",
        "623311": "1368.01",
        "623312": "827.36",
        "623990": "222.46",
        "624110": "421.97",
        "624120": "662.64",
        "624190": "269.49",
        "624210": "558.17",
        "624221": "285.19",
        "624229": "476.15",
        "624230": "257.67",
        "624310": "337.27",
        "624410": "959.88",
        "711110": "134.03",
        "711120": "128.1",
        "711130": "126.66",
        "711190": "195.98",
        "711211": "86.68",
        "711212": "157.46",
        "711219": "87.45",
        "711310": "42.02",
        "711320": "47.45",
        "711410": "122.57",
        "711510": "110.79",
        "712110": "155.01",
        "712120": "337.19",
        "712130": "122.75",
        "712190": "131.17",
        "713110": "315.2",
        "713120": "362.41",
        "713210": "303.68",
        "713290": "158.17",
        "713910": "1400.61",
        "713920": "208.88",
        "713930": "588.68",
        "713940": "311.89",
        "713950": "388.14",
        "713990": "66.5",
        "721110": "548.17",
        "721120": "387.49",
        "721191": "1074.09",
        "721199": "252.91",
        "721211": "636.48",
        "721214": "204.73",
        "721310": "197.4",
        "722310": "883.03",
        "722320": "1923.99",
        "722330": "311.8",
        "722410": "1039.65",
        "722511": "2346.47",
        "722513": "1439.75",
        "722514": "1410.74",
        "722515": "814.8",
        "811111": "1353.13",
        "811112": "653.75",
        "811113": "1255.76",
        "811118": "227.16",
        "811121": "702.03",
        "811122": "538.9",
        "811191": "1237.2",
        "811192": "753.42",
        "811198": "277.4",
        "811211": "303.21",
        "811212": "514.57",
        "811213": "348.84",
        "811219": "226.88",
        "811310": "189.28",
        "811411": "241.11",
        "811412": "318.87",
        "811420": "388.31",
        "811430": "379.12",
        "811490": "220.24",
        "812111": "1346.24",
        "812112": "1038.37",
        "812113": "1656.46",
        "812191": "649.12",
        "812199": "394.08",
        "812210": "667.64",
        "812220": "368.94",
        "812310": "464.86",
        "812320": "280.34",
        "812331": "170.49",
        "812332": "185.19",
        "812910": "422.09",
        "812921": "484.2",
        "812922": "344.1",
        "812930": "841.61",
        "812990": "148.3",
        "813110": "125.11",
        "813211": "585.71",
        "813212": "939.27",
        "813219": "711.79",
        "813311": "425.98",
        "813312": "603.55",
        "813319": "215.89",
        "813410": "179.26",
        "813910": "223.58",
        "813920": "175.34",
        "813930": "147.86",
        "813940": "180.72",
        "813990": "622.03",
        "814110": "325.29",
        "921110": "203.31",
        "921120": "162.65",
        "921130": "116.18",
        "921140": "542.15",
        "921150": "180.72",
        "921190": "116.18",
        "922110": "180.72",
        "922120": "95.67",
        "922130": "203.31",
        "922140": "162.65",
        "922150": "232.35",
        "922160": "203.31",
        "922190": "203.31",
        "923110": "180.72",
        "923120": "116.18",
        "923130": "108.43",
        "923140": "542.15",
        "924110": "203.31",
        "924120": "108.43",
        "925110": "406.61",
        "925120": "203.31",
        "926110": "95.67",
        "926120": "101.65",
        "926130": "108.43",
        "926140": "162.65",
        "926150": "73.93",
        "927110": "325.29",
        "928110": "125.11",
        "928120": "95.67",
    }

    n_dup17 = 2
    f = io.open("example_data_2017.txt", "w")
    f.write("|".join(["TEXT", "NAICS", "SAMPLE_WEIGHT"]) + "\n")
    for tup in combined17_tups:
        # Output duplicate observations to assist with illustrating cross-validation
        for i in range(n_dup17):
            f.write("|".join([tup[0], tup[1], weights17[tup[0]]]) + "\n")
    f.close()

    print("Sample size:            {}".format(len(combined17_tups) * n_dup17))
    print("Number of sectors:      {}".format(len(set(get_sector(tup[0]) for tup in combined17_tups))))
    print("Number of NAICS codes:  {}".format(len(set(tup[0] for tup in combined17_tups))))
    print("")

    # 2022 data WITHOUT sample weights
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@    2022 data WITHOUT sample weights    @@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("")

    index22_df = pd.read_excel("2022_NAICS_Index_File.xlsx")
    index22_tups = []
    for i in range(len(index22_df)):
        naics_temp = index22_df.at[i, "NAICS22"]
        descr_temp = index22_df.at[i, "INDEX ITEM DESCRIPTION"]
        if not (pd.isna(naics_temp) or naics_temp == "******" or pd.isna(descr_temp)):
            index22_tups.append((format_naics(naics_temp), format_descr(descr_temp)))

    codes22_df = pd.read_excel("6-digit_2022_Codes.xlsx")
    codes22_tups = []
    for i in range(len(codes22_df)):
        naics_temp = codes22_df.at[i, "2022 NAICS Code"]
        descr_temp = codes22_df.at[i, "2022 NAICS Title"]
        if not (pd.isna(naics_temp) or pd.isna(descr_temp)):
            codes22_tups.append((format_naics(naics_temp), format_descr(descr_temp)))

    combined22_tups = sorted(list(set(index22_tups + codes22_tups)))

    n_dup22 = 2
    f = io.open("example_data_2022.txt", "w")
    f.write("|".join(["TEXT", "NAICS"]) + "\n")
    for tup in combined22_tups:
        # Output duplicate observations to assist with illustrating cross-validation
        for i in range(n_dup22):
            f.write("|".join([tup[0], tup[1]]) + "\n")
    f.close()

    print("Sample size:            {}".format(len(combined22_tups) * n_dup22))
    print("Number of sectors:      {}".format(len(set(get_sector(tup[0]) for tup in combined22_tups))))
    print("Number of NAICS codes:  {}".format(len(set(tup[0] for tup in combined22_tups))))
    print("")

    return

if __name__ == "__main__":
    main()
