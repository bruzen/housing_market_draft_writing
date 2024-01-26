parameters = {
            'width':     [10,20,100],
            'height':    20,

            # LABOUR MARKET AND FIRM PARAMETERS
            'subsistence_wage': 40000., # psi
            'init_city_extent': 10.,    # CUT OR CHANGE?
            'seed_population': 400,
            'init_wage_premium_ratio': 0.2, # 1.2, ###

            # PARAMETERS MOST LIKELY TO AFFECT SCALE
            'c':  [1000, 300, 100],     #300.0, TRANSPORT COST PER UNIT DISTANCE
            'price_of_output': [12, 10, 8], #10, WORLD PRICE FOR FIRM  
            'density': [2000, 600, 100],#600,  WORKERS PER BLOCK
            'A': [6000, 3000,],         #3000, SCALE FACTOR 
            'alpha': [2.0, 1.8, 1.6],   #0.18, EXPONENT ON FIRM CAPITAL
            'beta':  [0.8, 0.7, 0.6],   #0.75, EXPONENT ON FIRM LABOUR
            'gamma': [0.2, .12, 0.8],   #0.12, EXPONENT ON AGGLOMERATION POPULATION
            'overhead': [1.0, 0.5, 0.0,],#1,   FIRM OVERHEAD COST AS WAGE FRACTION
            'mult': [4, 2, 1.2],        #1.2,  AGGLOM POP IS MULT for x N
            'adjN': [.4, .15, .1],      #0.15  ADJUSTMENT RATE WORKER COUNT
            'adjk': [.15, .1, .05],     #0.10  ADJUSTMENT RATE FIRM CAPITAL
            'adjn': [0.5, 0.25, 0.1],   #0.25  ADJUSTMENT RATE FIRM WORKFORCE
            'adjF': [.4, .15, .1],      #0.15  ADJUSTMENT RATE NO OF FIRMS
            'adjw': [0.2, 0.12, 0.5],   #0.02  ADJUSTMENT RATE WAGE 
            'dist': [1], #1, 
            'init_F': [33, 100, 300],    #100.0, INITIAL NUMBER OF FIRMS
            'init_k': [500, 5000, 50000],#500.0, INITIAL CAPITAL STOCK OF FIRMS
            'init_n': [100,1000],        #100.0, INITIAL LABOURFORCE OF FIRMS

            # HOUSING AND MORTGAGE MARKET PARAMETERS
            'mortgage_period': 5.0,       # T, YEARS TO RENEGOTIATION OF MORGAGE
            'working_periods': [40 ,20],  # YEARS IN THE WORKFORCE
            'savings_rate': 0.3,
            'discount_rate': [0.15, 0.07, 0.0],# DISCOUNT RATE
            'r_prime': 0.05,              # BANK RATE          
            'r_margin': 0.01,             # BANK PROFIT
            'r_investor': 0.05,           # Next best alternative return for investor
            'property_tax_rate': [10.0, 0.04, 0],     # tau, ANNUAL PROPERTY TAX RATE
            'housing_services_share': 0.3, # a
            'maintenance_share': 0.2,      # b
            'max_mortgage_share': 0.9,
            'ability_to_carry_mortgage': 0.28,# MAX SHARE OF INCOME  FOR MORTGAGE PAYMENTS
            'wealth_sensitivity': 0.1,     # OF INTEREST CHARGE
            'cg_tax_per':   [.50, .25, 0.0],     # 0.0 CAPITAL GAINS TAX OWNER OCCUPIERS
            'cg_tax_invest': [1.0, .50, .25,.],  # 1.0   CAPITAL GAINS TAX INVESTORS
        }