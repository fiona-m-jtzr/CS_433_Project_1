
#Does not describe the health of the participant
ADMIN = [
    # interview date and identifiers
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    # phone screening questions (landline / cell phone, household size)
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "NUMADULT", "NUMMEN", "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT",
    "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT",
    # phone ownership, only used for weighting
    "NUMHHOL2", "NUMPHON2", "CPDEMO1",
    # location and questionnaire version / language
    "_STATE", "QSTVER", "QSTLANG",
    # survey weights and strata (how the sample was drawn)
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT",
    # CDC flags for missing / out-of-range values in other columns
    "_MISFRTN", "_MISVEGN", "_FRTRESP", "_VEGRESP", "_FRT16", "_VEG23",
    "_FRUITEX", "_VEGETEX", "PAMISS1_",
]

#Answers that have been translated into calculated variables 
RAW_DUPLICATES = [
    "WEIGHT2", "HEIGHT3",                    # -> WTKG3, HTIN4, HTM4, _BMI5
    "ALCDAY5",                               # -> DROCDY3_, _DRNKWEK
    "FRUITJU1", "FRUIT1", "FVBEANS",         # -> FTJUDA1_, FRUTDA1_, BEANDAY_
    "FVGREEN", "FVORANG", "VEGETAB1",        # -> GRENDAY_, ORNGDAY_, VEGEDA1_
    "EXEROFT1", "EXERHMM1", "STRENGTH",      # -> PAFREQ1_, PADUR1_, STRFREQ_
    "EXEROFT2", "EXERHMM2",                  # -> PAFREQ2_, PADUR2_
]

#Dates of flu shots and HIV test 
DATES = ["FLSHTMY2", "HIVTSTD3"]

#Coded as unit + amount (1xx = day, 2xx = week, 3xx = month, 4xx = year), no calculated version
MIXED_UNITS = ["BLDSUGAR", "FEETCHK2", "LONGWTCH"]

REMOVE = ADMIN + RAW_DUPLICATES + DATES + MIXED_UNITS

CODE_EXCEPTIONS = {
#7, 8, 77 or 88 is a real answer
    "EMPLOY1":  ([9], None),          # 7 = retired, 9 = refused
    "DIABAGE2": ([98, 99], None),     # 1-97 = age (77, 88 are ages), 98 = don't know
    "SEATBELT": ([7, 8, 9], None),    # 8 = never drives, off the always..never scale
    "SCNTMNY1": ([7, 8, 9], None),    # 8 = not applicable, off the always..never scale
    "SCNTMEL1": ([7, 8, 9], None),
    # calculated variables with their own missing code
    "_RACE":    ([9], None),          # 7 = multiracial, 8 = Hispanic, 9 = don't know
    "_AGEG5YR": ([14], None),         # 1-13 = age groups, 14 = don't know/refused
    "_AGE65YR": ([3], None),          # 1-2 = age groups, 3 = don't know/refused
    "DROCDY3_": ([900], None),        # 900 = don't know/refused
    "_DRNKWEK": ([99900], None),      # 99900 = don't know/refused
    "MAXVO2_":  ([999], None),        # 999 = don't know/refused
    "FC60_":    ([999], None),
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
    # visual impairment: nobody refused, so the max value is 7 or 8 instead of 9
    "VIDFCLT2": ([6, 7, 8], None),    # 6 = unable for other reasons, 8 = blind, off the no..unable scale
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
    # HPV vaccination
    "HPVADVC2": ([3, 7, 9], None),    # 3 = doctor refused when asked
    # social context: hours worked per week
    "SCNTWRK1": ([97, 99], 98),       # 1-96 = hours, 97 = don't know, 98 = zero
    "SCNTLWK1": ([97, 99], 98),
}
