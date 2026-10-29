"""Column removal lists and missing-code exceptions for BRFSS 2015.
This file was created using an LLM.
"""

# Does not describe the health of the participant
ADMIN = [
    # ID
    "Id",
    # interview date and identifiers
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    # phone screening questions (landline / cell phone, household size)
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "NUMADULT", "NUMMEN", "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT",
    "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT",
    # phone ownership, only used for weighting
    "NUMHHOL2", "NUMPHON2", "CPDEMO1",
    # location and questionnaire version / language
    "_STATE", "QSTVER", "QSTLANG", "_GEOSTR", "_DENSTR2",
    # survey weights and strata (how the sample was drawn)
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT",
    "_DUALUSE", "_DUALCOR", "_LLCPWT",
    # CDC flags for missing / out-of-range values in other columns
    "_MISFRTN", "_MISVEGN", "_FRTRESP", "_VEGRESP", "_FRT16", "_VEG23",
    "_FRUITEX", "_VEGETEX", "PAMISS1_",
    # child-survey variables (from memory, mostly empty)
    "_CHISPNC", "_CRACE1", "_CPRACE", "_IMPCAGE", "_IMPCRAC", "_IMPCSEX",
]

# Answers that have been translated into calculated variables
RAW_DUPLICATES = [
    "WEIGHT2", "HEIGHT3",                    # -> WTKG3, HTIN4, HTM4, _BMI5
    "ALCDAY5",                               # -> DROCDY3_, _DRNKWEK
    "FRUITJU1", "FRUIT1", "FVBEANS",         # -> FTJUDA1_, FRUTDA1_, BEANDAY_
    "FVGREEN", "FVORANG", "VEGETAB1",        # -> GRENDAY_, ORNGDAY_, VEGEDA1_
    "EXEROFT1", "EXERHMM1", "STRENGTH",      # -> PAFREQ1_, PADUR1_, STRFREQ_
    "EXEROFT2", "EXERHMM2",                  # -> PAFREQ2_, PADUR2_
    "_IMPAGE",                               # imputed age, redundant with AGE / _AGE80
]

# Dates of flu shots and HIV test (MMYYYY).
DATES = ["FLSHTMY2", "HIVTSTD3"]

# Coded as unit + amount (1xx = day, 2xx = week, 3xx = month, 4xx = year),
# no calculated version
MIXED_UNITS = ["BLDSUGAR", "FEETCHK2", "LONGWTCH"]

# Optional: raw/calculated pairs that stay in the data (collinear). Not in REMOVE.
OPTIONAL_REDUNDANT = [
    "_RFHLTH",    # GENHLTH
    "_EDUCAG",    # EDUCA
    "_INCOMG",    # INCOME2
    "_CHLDCNT",   # CHILDREN
    "_AGEG5YR", "_AGE65YR", "_AGE_G",   # AGE / _AGE80
    "_TOTINDA",   # EXERANY2
    "_BMI5CAT", "_RFBMI5",              # _BMI5
    "_RFSMOK3",   # _SMOKER3
    "_RFDRHV5",   # _DRNKWEK
    "_LTASTH1", "_CASTHM1",             # _ASTHMS1
    "_RFSEAT3",   # _RFSEAT2
    "_FRTLT1", "_VEGLT1",               # _FRUTSUM, _VEGESUM
    "_PA150R2", "_PA300R2", "_PA30021", "_PAINDX1", "_PAREC1", "_PASTAE1",
    "_RACEG21", "_RACEGR3", "_RACE_G1", "_PRACE1", "_MRACE1", "_HISPANC",
]

REMOVE = ADMIN + RAW_DUPLICATES + DATES + MIXED_UNITS + OPTIONAL_REDUNDANT


# (missing_codes, zero_code). Include 9 / 99 even if absent from train:
# test data may contain them.
CODE_EXCEPTIONS = {
    # 7, 8, 77 or 88 is a real answer
    "AGE":      ([7, 9], None),       # 7 = don't know, 9 = refused; 18-99 = age (77, 99 real)
    "EMPLOY1":  ([9], None),          # 7 = retired, 9 = refused
    "DIABAGE2": ([98, 99], None),     # 1-97 = age (77, 88 are ages), 98 = don't know
    "CHILDREN": ([99], 88),           # 1-87 = count, 88 = none, 99 = refused
    # Design choice: 8 = never drives/rides, off the always..never scale.
    # The CDC's own _RFSEAT2 counts it as "always"; alternative is to keep 8 as a category.
    "SEATBELT": ([7, 8, 9], None),
    "SCNTMNY1": ([7, 8, 9], None),    # 8 = not applicable
    "SCNTMEL1": ([7, 8, 9], None),

    # calculated variables with their own missing code
    "_RACE":    ([9], None),          # 7 = multiracial, 8 = Hispanic, 9 = don't know
    "_PRACE1":  ([77, 99], None),     # 7, 8 are real categories
    "_MRACE1":  ([77, 99], None),
    "_AGEG5YR": ([14], None),         # 1-13 = age groups, 14 = don't know/refused
    "_AGE65YR": ([3], None),          # 1-2 = age groups, 3 = don't know/refused
    "DROCDY3_": ([900], None),        # stored x100
    "_DRNKWEK": ([99900], None),      # stored x100
    "MAXVO2_":  ([99900], None),      # stored x100 (was 999)
    "FC60_":    ([99900], None),      # stored x100 (was 999)
    "HTIN4":    ([999], None),
    "HTM4":     ([999], None),
    "WTKG3":    ([99999], None),

    # numeric calculated variables where NaN is the only missing marker
    "_BMI5":    ([], None),
    "_FRUTSUM": ([], None),
    "_VEGESUM": ([], None),
    "_MINAC11": ([], None),
    "_MINAC21": ([], None),
    "PA1MIN_":  ([], None),
    "PA1VIGM_": ([], None),
    "PAMIN11_": ([], None),
    "PAMIN21_": ([], None),
    "PAVIG11_": ([], None),
    "PAVIG21_": ([], None),
    "PADUR1_":  ([], None),
    "PADUR2_":  ([], None),
    "METVL11_": ([], None),
    "METVL21_": ([], None),
    # times per week / per day: 77, 88 and 99 are real values
    "PAFREQ1_": ([], None),
    "PAFREQ2_": ([], None),
    "STRFREQ_": ([], None),
    "FTJUDA1_": ([], None),
    "FRUTDA1_": ([], None),
    "BEANDAY_": ([], None),
    "GRENDAY_": ([], None),
    "ORNGDAY_": ([], None),
    "VEGEDA1_": ([], None),

    # optional modules (mostly > 90 % NaN, only asked in a few states)
    # diabetes
    "CHKHEMO3": ([77, 98, 99], 88),   # 98 = never heard of the A1C test
    # visual impairment (design choice: 8 = blind treated as missing; could be kept)
    "VIDFCLT2": ([6, 7, 8, 9], None), # 6 = unable for other reasons
    "VIREDIF3": ([6, 7, 9], None),
    "VIEYEXM2": ([7, 8, 9], None),    # 5 = never, 8 = blind
    "VINOCRE2": ([8, 77, 99], None),  # 8 = blind
    "VIINSUR2": ([7, 8, 9], None),
    "VICTRCT4": ([7, 9], None),       # 2 = had them removed, 3 = no
    "VIGLUMA2": ([7, 9], None),
    "VIMACDG2": ([7, 9], None),
    # asthma call-back: counts use 88 = none, 98 = don't know (77 is a real count / age)
    "ASTHMAGE": ([98, 99], None),     # 11-96 = age, 97 = age 10 or younger
    "ASATTACK": ([7, 9], None),
    "ASERVIST": ([98, 99], 88),
    "ASDRVIST": ([98, 99], 88),
    "ASRCHKUP": ([98, 99], 88),
    "ASYMPTOM": ([7, 9], 8),          # 8 = not at any time
    "ASNOSLEP": ([7, 9], 8),          # 8 = none
    "ASTHMED3": ([7, 9], 8),          # 8 = never
    "ASINHALR": ([7, 9], 8),          # 8 = never
    # cardiovascular
    "RDUCHART": ([7, 9], None),
    # HPV vaccination (design choice: 3 = doctor refused, treated as missing)
    "HPVADVC2": ([3, 7, 9], None),
    # social context: hours worked per week.
    # VERIFY against the codebook: 97 may mean "97 or more hours", not don't know.
    "SCNTWRK1": ([97, 99], 98),       # 1-96 = hours, 98 = zero
    "SCNTLWK1": ([97, 99], 98),
}


CONT_CALCULATED = {
    "_AGE80", "_BMI5", "_DRNKWEK", "_FRUTSUM", "_VEGESUM", "_MINAC11", "_MINAC21",
    "HTIN4", "HTM4", "WTKG3", "DROCDY3_",
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_",
    "MAXVO2_", "FC60_", "METVL11_", "METVL21_",
    "PADUR1_", "PADUR2_", "PAFREQ1_", "PAFREQ2_", "STRFREQ_",
    "PAMIN11_", "PAMIN21_", "PA1MIN_", "PAVIG11_", "PAVIG21_", "PA1VIGM_",
}
 
CONT_RAW = {  # VERIFY each against the codebook
    "AGE",                                   # 18-99
    "PHYSHLTH", "MENTHLTH", "POORHLTH",      # days 1-30, 88 = none -> 0
    "CHILDREN",                              # count, 88 = none -> 0
    "SLEPTIM1",                              # hours
    "AVEDRNK2", "DRNK3GE5", "MAXDRNKS",      # drinks
    "DOCTDIAB", "CHKHEMO3",                  # counts (diabetes module)
    "DIABAGE2", "ASTHMAGE",                  # age at diagnosis
    "ASERVIST", "ASDRVIST", "ASRCHKUP",      # counts (asthma call-back)
    "SCNTWRK1", "SCNTLWK1",                  # hours worked
    "JOINPAIN",                              # 0-10 scale
}
 
CONT_VARS = CONT_CALCULATED | CONT_RAW