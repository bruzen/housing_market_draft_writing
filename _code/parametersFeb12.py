default_parameters = {
    'run_notes': 'Exploring behavior.',
    'subfolder': None,
    'width':     50, #30,
    'height':    50, #30,

    # FLAGS
    'demographics_on': True,  # Set flag to False for debugging to check firm behaviour without demographics or housing market
    'center_city':     False, # Flag for city center in center if True, or bottom corner if False
    # 'random_init_age': False,  # Flag for randomizing initial age. If False, all workers begin at age 0
    'random_init_age': True,  # Flag for randomizing initial age. If False, all workers begin at age 0

    # DATA STORAGE
    'store_agent_data': True,
    'no_decimals':      1,

    # STORAGE VALUES, JUST FOR MODEL FAST
    'distances': None,
    'newcomer_savings': None,

    # LABOUR MARKET AND FIRM PARAMETERS
    'subsistence_wage': 40000., # psi
    'init_city_extent': 2.,    # CUT OR CHANGE?
    'seed_population': 400,
    'init_wage_premium_ratio': 0.2, # 1.2, ###

    # PARAMETERS MOST LIKELY TO AFFECT SCALE
    'c': 300.0,                            ### transportation costs
    'price_of_output': 10,                 ######
    'density': 400,                        #####
    'A': 3000,                             ### 
    'alpha': 0.18,
    'beta':  0.75,
    'gamma': 0.1, ### reduced from .14
    'overhead': .5,
    'mult': 1.0,
    'adjN': 0.15,
    'adjk': 0.10,
    'adjn': 0.15,
    'adjF': 0.02,
    'adjw': 0.02, 
    'adjs': 0.2,  #Adjust worker Supply 
    'adjd': 0.2,  #Adjust worker Demand 
    'adjp': 0.2,  #Adjust agglom_pop -Population
    'dist': 1, 
    'init_F': 100.0,
    'init_k': 500000.0,
    'init_n': 100.0,
    'animal_spirits': 1.,

    # HOUSING AND MORTGAGE MARKET PARAMETERS
    'mortgage_period': 5.0,       # T, in years
    'working_periods': 40,        # in years
    'savings_rate': 0.3,
    'discount_rate': 0.07,        # 1/delta
    'r_prime': 0.05,
    'r_margin': 0.01,
    'r_investor': 0.05,            # Next best alternative return for investor
    'property_tax_rate': 0.04,     # tau, annual rate, was c
    'housing_services_share': 0.3, # a
    'maintenance_share': 0.2,      # b
    'max_mortgage_share': 0.9,
    'ability_to_carry_mortgage': 0.28,
    'wealth_sensitivity': 0.1,
    'cg_tax_per':   0.01, # share 0-1
    'cg_tax_invest': 0.15, # share 0-1
}