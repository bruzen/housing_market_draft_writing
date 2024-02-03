import math
import logging
# import numpy as np
from typing import Union
from collections import defaultdict
from scipy.spatial import distance

from mesa import Agent

class Land(Agent):
    """Land parcel.

    :param unique_id: An integer identifier.
    :param model: The main city model.
    :param pos: The land parcel's location on the spatial grid.
    :param resident: The agent who resides at this land parcel.
    :param owner: The agent who owns this land parcel.
    """

    @property 
    def market_rent(self):
        return self.warranted_rent

    @property
    def net_rent(self):
        return self.warranted_rent - self.maintenance - self.property_tax # TODO check we don't use net rent for warranted_price
    
    @property
    def appraised_price(self):
        return self.warranted_price

    @property
    def property_tax(self):
        tau              = self.property_tax_rate
        appraised_price  = self.appraised_price
        return tau * appraised_price

    def __init__(self, unique_id, model, pos, 
                 property_tax_rate = 0., 
                 resident = None, owner = None):
        super().__init__(unique_id, model)
        self.pos                  = pos
        self.property_tax_rate    = property_tax_rate
        self.resident             = resident
        self.owner                = owner
        self.distance_from_center = self.calculate_distance_from_center()
        self.transport_cost           = self.calculate_transport_cost()
        self.warranted_rent           = self.get_warranted_rent()
        self.warranted_price          = self.get_warranted_price()
        self.maintenance              = self.get_maintenance()
        self.realized_price           = - 1
        self.realized_all_steps_price = - 1
        self.ownership_type           = - 1
        self.p_dot                    = None

    def step(self):
        self.warranted_rent  = self.get_warranted_rent()
        self.warranted_price = self.get_warranted_price()

        # self.p_dot = None # Calculate when properties are listed for sale
        if (self.model.firm.wage_premium > self.transport_cost):
            self.p_dot       = self.model.firm.p_dot
        else:
            self.p_dot       = None 

        # TODO Flip
        self.realized_price         = - 1 # Reset to show realized price in just this time_step
        # self.realized_all_steps_price = - 1 # 

        if self.resident is None:
            self.model.logger.warning(f'Land has no resident: {self.unique_id}, pos {self.pos}, resident {self.resident}, owner {self.owner}')

    def calculate_distance_from_center(self, method='euclidean'):
        if method == 'euclidean':
            return distance.euclidean(self.pos, self.model.center)
        elif method == 'cityblock':
            return distance.cityblock(self.pos, self.model.center)
        else:
            raise ValueError("Invalid distance calculation method."
                            "Supported methods are 'euclidean' and 'cityblock'.")

    def calculate_transport_cost(self, dist=None):
        if dist:
            dist = dist
        else:
            dist = self.distance_from_center
        cost = self.distance_from_center * self.model.transport_cost_per_dist
        return cost

    def change_owner(self, new_owner, old_owner):
        if not self.check_owners_match:
            owner_properties = ' '.join(str(prop.unique_id) if hasattr(prop, 'unique_id') else str(prop) for prop in self.owner.properties_owned)
            self.model.logger.error(f'In change_owner, property owner does not match an owner in owners properties_owned list: property {self.unique_id}, property\'s owner {self.owner.unique_id}, owner\'s properties {owner_properties} ')

        if not self.owner == old_owner:
            self.model.logger.error(f'In change_owner, the old_owner must own a property in order to transfer ownership: property {self.unique_id}, old_owner {old_owner.unique_id}, property\'s owner {self.owner.unique_id}')

        self.owner = new_owner

        # Remove the land from the old owner's properties_owned list
        old_owner.properties_owned.remove(self)

        # Add the land to the new owner's properties_owned list
        new_owner.properties_owned.append(self)

        # # Update the owner type
        # if isinstance(self.owner, Person):
        #     self.ownership_type = 1 # 'Person'
        # elif isinstance(self.owner, Investor):
        #     self.ownership_type = 2 # 'Investor'
        # else:
        #     self.ownership_type = 3 # 'Other'
        #     self.model.logger.warning(f'Land {self.unique_id} owner not a person or investor. Owner: {self.owner}')

    def get_maintenance(self):
        a                 = self.model.housing_services_share
        b                 = self.model.maintenance_share
        subsistence_wage  = self.model.firm.subsistence_wage # subsistence_wage
        return a * b * subsistence_wage

    def get_warranted_rent(self):  ####   ADD AMENITY HERE
        wage_premium     = self.model.firm.wage_premium
        subsistence_wage = self.model.firm.subsistence_wage
        a                = self.model.housing_services_share
        # return max(wage_premium - self.transport_cost, 0)
        # Note, this is the locational_rent. It is the warranted level of extraction. It is the economic_rent when its extracted.
        # We set the rural land value to zero to study the urban land market, with the agricultural price, warranted rent would be:
        return max(wage_premium - self.transport_cost + a * subsistence_wage, 0)
        # TODO add amenity + A
        # TODO should it be positive outside the city? How to handle markets outside the city if it is?
        # TODO but outside the city, any concern with transportation cost is only speculative - how to handle

    def get_warranted_price(self):
        return self.warranted_rent / self.model.r_prime

    def check_owners_match(self):
        for owned_property in self.owner.properties_owned:
            if self.unique_id == owned_property.unique_id:
                return True
        return False
 
    def change_dist(self, dist):
        self.model.logger.warning(f'Note land has set rather than calculated distance. Could introduce errors. {self.unique_id}.')
        self.distance_from_center     = dist # self.calculate_distance_from_center()
        self.transport_cost           = self.calculate_transport_cost()
        self.warranted_rent           = self.get_warranted_rent()
        self.warranted_price          = self.get_warranted_price()
        self.maintenance              = self.get_maintenance()
        self.p_dot                    = self.model.firm.p_dot

    def __str__(self):
        return f'Land {self.unique_id} (Dist. {self.distance_from_center}, Pw {self.warranted_price})'

class Person(Agent):
    @property
    def borrowing_rate(self):
        """Borrowing rate of the person.

        Returns:
        The borrowing rate calculated based on the model's  \
        target rate and individual wealth adjustment.
        """
        return self.model.r_target + self.individual_wealth_adjustment

    @property
    def individual_wealth_adjustment(self):
        """Individual wealth adjustment. Added on to the agent's mortgage 
        borrowing rate. It depends on the agent's wealth.

        # TODO: Fix
 
        Formula for interest rate they get: r_target + K/(W-W_min) - K/(W_avg-W_min)
        Formula for adjustment: K/(W-W_min) - K/(W_avg-W_min)
        K is wealth sensitivity parameter

        Returns:
        The individual wealth adjustment value.
        """
        # r_target = self.model.r_target
        K        = self.model.wealth_sensitivity
        W        = self.get_wealth() 
        W_min    = 10000. # TODO could be 0 or 20K
        W_avg    = self.model.bank.get_average_wealth()
        return K / (W - W_min) - K / (W_avg - W_min)
        # return 0.002

    def __init__(self, unique_id, model, pos, init_working_period = 0,
                 savings = 0., debt = 0.,
                 capital_gains_tax = None,
                 residence_owned = None,):
        super().__init__(unique_id, model)
        self.pos = pos
        # self.model.workforce = self.model.workforce

        self.init_working_period = init_working_period
        self.working_period      = init_working_period
        self.savings             = savings

        self.properties_owned    = []
        self.residence           = residence_owned
        if not capital_gains_tax:
            self.model.logger.warning(f'No capital gains tax for person {self.unique_id}.')
        self.capital_gains_tax   = capital_gains_tax

        self.bank                = self.model.bank 
        self.amenity             = 0.
        self.purchased_property  = False

        # If the agent initially owns a property, set residence and owners
        if self.residence:
            self.properties_owned.append(self.residence)
            if self.residence.owner is not None:
                self.model.logger.warning(f'Overrode property {self.residence.unique_id} owner {self.residence.owner} in init. Property now owned by {self.unique_id}.')
            self.residence.owner = self

            if self.residence.resident is not None:
                self.model.logger.warning(f'Overrode property {self.residence.unique_id} resident {self.residence.resident} in init. Property resident now {self.unique_id}.')
            self.residence.resident = self

        # Count time step and track whether agent is working
        self.count               = 0 # TODO check if we still use this
        self.is_working_check    = 0 # TODO delete?

        # else:
        #     self.model.workforce.remove(self, self.model.workforce.workers)

    def step(self):
        self.count              += 1
        self.working_period     += 1
        premium                  = self.model.firm.wage_premium

        # Non-residents
        if not isinstance (self.residence, Land):
            # Newcomers who don't find a home leave the city
            if (self.unique_id in self.model.workforce.newcomers):
                # if (self.residence == None):
                if self.count > 0:
                    self.model.logger.debug(f'Newcomer removed {self.unique_id}') #  removed, who owns {self.properties_owned}, count {self.count}')
                    self.remove()
                    return  # Stop execution of the step function after removal
            # Everyone who is not a newcomer leaves if they have no residence
            else:
                self.model.logger.warning(f'Non-newcomer with no residence removed: {self.unique_id}, count {self.count}')
                self.remove()
                return  # Stop execution of the step function after removal

        # Urban workers
        elif premium > self.residence.transport_cost:
            self.model.workforce.add(self, self.model.workforce.workers)
            # Agents list properties a step before they stop working for consistent worker numbers
            if self.working_period >= self.model.working_periods:
                if self.residence in self.properties_owned:
                    self.model.workforce.add(self, self.model.workforce.retiring_urban_owner)

                    # P_bid    = self.model.bank.get_max_desired_bid(R_N, r, r_target, m, listing.sale_property.p_dot, self.capital_gains_tax, listing.sale_property.transport_cost)            
                    reservation_price = self.model.bank.get_reservation_price(
                        R_N = self.residence.net_rent, 
                        r = self.model.r_prime, 
                        r_target = self.model.r_target, 
                        m =  0.8, 
                        p_dot =  self.residence.p_dot, 
                        capital_gains_tax = self.capital_gains_tax,
                        transport_cost = self.residence.transport_cost)
                    self.model.realtor.list_property_for_sale(self, self.residence, reservation_price)
                    # TODO Contact bank. Decide: sell, rent or keep empty
                    self.model.logger.debug(f'Agent is retiring: {self.unique_id}, period {self.working_period}')

            if self.working_period > self.model.working_periods:
                if self.residence in self.properties_owned:
                    self.model.workforce.remove(self, self.model.workforce.workers)
                    self.model.logger.warning(f'Urban homeowner still in model: {self.unique_id}, working_period {self.working_period}')
                else:
                    self.working_period = 1
                    self.savings        = 0
                    # TODO what should reset savings/age be for new renters? This simply resets keeping the initial distribution of ages cycling outside the city

        # Rural population
        else:
            self.model.workforce.remove(self, self.model.workforce.workers)
            if self.working_period > self.model.working_periods:
                self.working_period = 1
                self.savings        = 0
                # TODO what should reset savings/age be? This simply resets keeping the initial distribution of ages cycling outside the city
                
        # Update savings and wealth
        self.savings += self.model.savings_per_step # TODO debt, wealth
        # self.wealth  = self.get_wealth()

        # TODo consider doing additional check for retiring agents who are still in model
        # if self.unique_id in self.model.workforce.retiring_urban_owner:        
        #     if (self.residence):
        #         if premium > self.residence.transport_cost:
        #             self.model.logger.warning(f'Removed retiring_urban_owner agent {self.unique_id} in step, working, with residence, properties owned {len(self.properties_owned)} which was still in model.')
        #             self.remove()
        #         else:
        #             self.model.logger.warning(f'retiring_urban_owner agent {self.unique_id}, not working, with residence, still in model.')
        #     else:
        #         self.model.logger.warning(f'retiring_urban_owner agent {self.unique_id}, witout residence, still in model.')
        # else:
        #     self.model.logger.debug(f'Agent {self.unique_id} has no residence.')

        # TODO TEMP? use is_working_check to perform any checks
        if self.residence:
            if premium > self.residence.transport_cost:
                self.is_working_check = 1
            else:
                self.is_working_check = 0

            # TODO Temp
            if isinstance(self.residence.owner, Person):
                if self.residence.owner.purchased_property:
                    self.residence.ownership_type = 1 # 'Person' True newcomer purchased property
                else:
                    self.residence.ownership_type = 0 # 'Person' False original owner
                if premium > self.residence.transport_cost:
                    self.model.urban_resident_owners_count += 1
            elif isinstance(self.residence.owner, Investor):
                self.residence.ownership_type = 2 # 'Investor'
                if premium > self.residence.transport_cost:
                    self.model.urban_investor_owners_count += 1
            else:
                self.residence.ownership_type = 3 #'Other' including retiree ownership
                if premium > self.residence.transport_cost:
                    self.model.urban_other_owners_count += 1

    def work_if_worthwhile_to_work(self):
        premium = self.model.firm.wage_premium
        if premium > self.residence.transport_cost:
            self.model.workforce.add(self, self.model.workforce.workers)
        else:
            self.model.workforce.remove(self, self.model.workforce.workers)

    def bid_on_properties(self):
        """Newcomers bid on properties for use or investment value."""
        
        # self.model.logger.debug(f'Newcomer bids: {self.unique_id}, count {self.count}')
        # m = max_mortgage_share - lenders_wealth_sensitivity * average_wealth / W # TODO adjust max mortgage share
        max_mortgage_share  = self.model.max_mortgage_share # m # TODO make wealth dependant?
        max_mortgage         = self.get_max_mortgage()      # M

        for listing in self.model.realtor.bids:
            net_rent        = listing.sale_property.net_rent # Net rent
            p_dot           = listing.sale_property.p_dot
            transport_cost  = listing.sale_property.transport_cost
            P_bid, bid_type = self.get_max_bid(m     = max_mortgage_share, 
                                               M     = max_mortgage,
                                               R_N   = net_rent, 
                                               p_dot = p_dot, 
                                               transport_cost = transport_cost)

            self.model.realtor.add_bid(self, listing, P_bid, bid_type)

    def get_max_mortgage(self, savings = None):
        # W = self.savings # TODO fix self.get_wealth()
        S = savings if savings is not None else self.savings
        r = self.borrowing_rate
        r_prime  = self.model.r_prime
        r_target = self.model.r_target # TODO this is personal but uses same as bank. Clarify.        
        wage     = self.model.firm.wage
        # average_wealth = # TODO? put in calculation

        # Max mortgage
        M = 0.28 * (wage + r * S) / r_prime
        return M

    def get_max_bid(self, m, M, R_N, p_dot, transport_cost, savings = None):
        # m = mortgage_share
        # M = max_mortgage # TODO M why not used?
        # W = self.savings # TODO fix self.get_wealth()
        S = savings if savings is not None else self.savings
        r = self.borrowing_rate
        r_prime  = self.model.r_prime
        r_target = self.model.r_target # TODO this is personal but uses same as bank. Clarify.        
        wage     = self.model.firm.wage
        # average_wealth = # TODO? put in calculation

        # First Calculate value of purchase (max bid)
        
        bid_type = 'value_limited'
        P_bid    = self.model.bank.get_max_desired_bid(R_N, r, r_target, m, p_dot, self.capital_gains_tax, transport_cost)
        # self.model.logger.warning(f'Max bid: {self.unique_id}, bid {P_bid}, R_N {R_N}, r {r}, r {r_target}, m {m}, transport_cost {transport_cost}')

        if S/(1-m) <= P_bid:
            bid_type = 'equity_limited'
            P_bid = S/(1-m)
            self.model.logger.warning(f'Newcomer bid EQUITY LIMITED: {self.unique_id}, bid {P_bid}') #, S {S}, m {m}, .. ')

        if (0.28 * (wage + r * S) / r_prime)  <= P_bid:
            bid_type = 'income_limited'
            P_bid = 0.28 * (wage + r * S) / r_prime
            self.model.logger.warning(f'Newcomer bid INCOME LIMITED: {self.unique_id}, bid {P_bid}')

        if P_bid < 0:
            bid_type = 'negative'
            # P_bid = 0
            self.model.logger.warning(f'Newcomer bid is NEGATIVE: {self.unique_id}, bid {P_bid}')

        else:
            bid_type = 'none'
            self.model.logger.warning(f'Newcomer doesnt bid: {self.unique_id}, bid {P_bid}') #, sale_property {listing.sale_property.unique_id}')
        return P_bid, bid_type

        # # Old logic, replaced by version above
        # # Max desired bid
        # # P_max_bid = self.model.bank.get_max_desired_bid(R_N, r, r_target, m, transport_cost)

        # mortgage_share_max = m * P_max_bid # TODO this should have S in it. 
        # mortgage_total_max = M

        # # Agents cannot exceed any of their constraints
        # if mortgage_share_max < mortgage_total_max:
        #     # Mortgage share limited
        #     if mortgage_share_max + S < P_max_bid:
        #         P_bid = mortgage_share_max + S
        #         bid_type = 'mortgage_share_limited'
        #     # Max bid limited
        #     else:
        #         P_bid = P_max_bid
        #         bid_type = 'max_bid_limited'
        #     mortgage = P_bid - S # TODO is this right? what about savings. 
        # else:
        #     mortgage = M
        #     # Mortgage total limited
        #     if mortgage_total_max + S < P_max_bid:
        #         P_bid = mortgage_total_max + S
        #         bid_type = 'mortgage_total_limited'
        #     # Max bid limited
        #     else:
        #         P_bid = P_max_bid
        #         bid_type = 'max_bid_limited'

    def get_wealth(self):
        # TODO Wealth is properties owned, minus mortgages owed, plus savings.
        # W = self.resident_savings + self.P_expected - self.M
        return self.savings

    def remove(self):
        self.model.removed_agents += 1
        self.model.workforce.remove_from_all(self)
        # self.model.grid.remove(self)
        self.model.schedule.remove(self)
        # x, y = self.pos
        # self.model.grid.remove_agent(x,y,self)
        self.model.grid.remove_agent(self)
        # self.model.logger.debug(f'Person {self.unique_id} removed from model')
        # TODO If agent owns property, get rid of property

    def __str__(self):
        return f'Person {self.unique_id}'

class Firm(Agent):
    """Firm.

    :param unique_id: An integer identifier.
    :param model: The main city model.
    :param pos: The firms's location on the spatial grid.
    :param init_wage_premium: initial urban wage premium.
    """

    # # TODO include seed population?
    # @property
    # def N(self):
    #     """total_no_workers"""
    #     total_no_workers = self.model.workforce.get_agent_count(self.model.workforce.workers)
    #     return total_no_workers * self.density + self.seed_population

    @property
    def p_dot(self):
        try:
            p_dot = (self.model.firm.wage_premium / self.model.firm.old_wage_premium)**self.model.mortgage_period - 1

            # # Handle the case where the result is negative
            # if p_dot < 0:
            #     p_dot = 0.

        except ZeroDivisionError:
            # Handle division by zero
            p_dot = None
            logging.error(f"ZeroDivisionError at time_step {self.model.time_step} for Land ID {self.unique_id}, old_wage_premium {self.model.firm.old_wage_premium}")
        except Exception as e:
            # Handle other exceptions
            self.model.logger.error(f"An error occurred: {str(e)}")
            p_dot = None
        return p_dot

    def __init__(self, unique_id, model, pos, 
                 subsistence_wage,
                 init_wage_premium_ratio,
                 alpha, beta, gamma,
                 price_of_output, r_prime,
                #  wage_adjustment_parameter,
                #  firm_adjustment_parameter,
                 seed_population,
                 density,
                 A,
                 overhead,
                 mult,
                 adjN,
                 adjk,
                 adjn,
                 adjF,
                 adjw,
                 dist,
                 init_F,
                 init_k,
                 init_n,
                 ):
        super().__init__(unique_id, model)
        self.pos             = pos

        # Old initialization calculations
        # # Calculate scale factor A for a typical urban firm
        # Y_R      = n_R * subsistence_wage / beta_F
        # Y_U      = self.n * self.wage / beta_F
        # k_R      = alpha_F * Y_R / self.r
        # self.k   = alpha_F * Y_U / self.r
        # self.A_F = 3500 # Y_R/(k_R**alpha_F * n_R * self.subsistence_wage**beta_F)

        # TEMP New parameter values
        self.subsistence_wage = subsistence_wage # subsistence_wage
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma
        self.price_of_output  = price_of_output
        self.seed_population  = seed_population
        self.density  = density
        self.A        = A
        self.overhead = overhead    # labour overhead costs for firm
        self.mult     = mult
        # self.c        = c
        self.adjN     = adjN
        self.adjk     = adjk
        self.adjn     = adjn
        self.adjF     = adjF
        self.adjw     = adjw
        self.dist     = dist
        # agent_count = 50 # TODO comes from agents deciding
        self.r        = r_prime # Firm cost of capital

        # Initial values # TODO do we need all these initial values?
        self.y        = 100000
        self.Y        = 0
        self.F        = init_F
        self.k        = init_k #1.360878e+09 #100
        self.n        = init_n
        self.F_target = init_F
        #self.k_target = 10000      
        #self.y_target = 10000
        self.N = self.F * self.n
        self.wage_premium = init_wage_premium_ratio * self.subsistence_wage 
        self.wage         = self.wage_premium + self.subsistence_wage
        self.MPL          = self.beta  * self.y / self.n  # marginal value product of labour known to firms
        self.wage_delta   = 0.0
        self.old_wage_premium = init_wage_premium_ratio * self.subsistence_wage   ### REVISED should remove inital problems

    def step(self):
        # GET POPULATION AND OUTPUT TODO replace N with agent count
        # self.N = self.get_N()
        self.n =  self.N / self.F # distribute workforce across firms
        self.y = self.A * self.N**self.gamma *  self.k**self.alpha * self.n**self.beta

        # SET TARGET WAGE EQUAL VALUE OF MARGINAL PRODUCT OF LABOUR
        self.MPL = self.beta  * self.y / self.n  # marginal value product of labour known to firms
        self.wage_target = self.price_of_output * self.MPL / (1 + self.overhead) # 
        # ADJUST WAGE: 
        self.wage = (1 - self.adjw) * self.wage + self.adjw * self.wage_target # partial adjustment process
        
        # FIND NEW WAGE PREMIUM
        self.old_wage_premium  = self.wage_premium
        self.wage_premium = self.wage /(1+self.overhead) - self.subsistence_wage # find wage available for transportation


        # FIND POPULATION AT NEW WAGE
        #self.dist = self.wage_premium / self.c  # find calculated extent of city at wage
        #self.N = self.dist * self.model.height * self.density / self.mult # calculate total firm population from city size # TODO make this expected pop
        #self.n =  self.N / self.F # distribute workforce across firms

        # ADJUST NUMBER OF FIRMS
        self.F_target = self.F * self.wage_target/self.wage  # this is completely arbitrary but harmless
        self.F = (1 - self.adjF) * self.F + self.adjF * self.F_target
 
        # ADJUST CAPITAL STOCK 
        self.y_target = self.price_of_output * self.A * self.N**self.gamma *  self.k**self.alpha * self.n**self.beta
        self.k_target = self.alpha * self.y_target/self.r
        self.k = (1 - self.adjk) * self.k + self.adjk * self.k_target
    
        # CALCULATE P_DOT
        self.wage_delta = (self.wage_premium - self.old_wage_premium ) #  -1 ???
        #self.F_target = self.F * self.n_target/self.n  #this is completely argbitrary but harmless
        # self.F_target = self.F*(self.n_target/self.n)**.5 # TODO name the .5
        ####self.F_target = (1-self.adjF)*self.F + self.adjF*self.F*(self.n_target/self.n) 
        #self.N_target = self.F_target * self.n_target
        #self.N = (1 - self.adjN) * self.N + self.adjN * self.N_target
        #####self.N = self.N*1.02
        #self.F = (1 - self.adjF) * self.F + self.adjF * self.F_target
        #self.k = (1 - self.adjk) * self.k + self.adjk * self.k_target
        # n = N/F 
        #self.wage_premium = self.c * math.sqrt(self.mult * self.N / (2 * self.density)) # TODO check role of multiplier
        #self.wage = self.wage_premium + self.subsistence_wage

        # # TODO Old firm implementation. Cut.
        # # Calculate wage, capital, and firm count given number of urban workers
        # self.n = self.N/self.F
        # self.y = self.output(self.N, self.k, self.n)

        # self.n_target = self.beta_F * self.y / self.wage
        # self.y_target = self.output(self.N, self.k, self.n_target)
        # self.k_target = self.alpha_F * self.y_target / self.r

        # # N_target_exist = n_target/self.n * self.N
        # adj_f = self.firm_adjustment_parameter # TODO repeats
        # self.F_target = self.n_target/self.n * self.F
        # self.F_next = (1 - adj_f) * self.F + adj_f * self.F_target
        # self.N_target_total = self.F_next * self.n_target
        # self.F_next_total = self.N_target_total / self.n_target

        # # adj_l = 1.25 # TODO self.labor_adjustment_parameter
        # # N_target_total = adj_l * n_target/self.n * self.N
        # # N_target_new = n_target * self.Z * (MPL - self.wage)/self.wage * self.F # TODO - CHECK IS THIS F-NEXT?

        # c = self.model.transport_cost_per_dist
        # self.wage_premium_target = c * math.sqrt(self.N_target_total/(2*self.density))        

        # k_next = self.k_target # TODO fix

        # adj_w = self.wage_adjustment_parameter
        # # self.wage_premium = self.wage_premium_target # TODO add back in wage adjusment process
        # # self.wage_premium = (1-adj_w) * self.wage_premium + adj_w * self.wage_premium_target
        # if self.model.time_step < 3:
        #     self.wage_premium = (1-adj_w)*self.wage_premium + adj_w * self.wage_premium_target
        # else:
        #     self.wage_premium += 100
        # self.k = k_next
        # self.F = self.F_next_total # OR use F_total

    # def output(self, N, k, n):
    #     A_F     = self.A_F
    #     alpha_F = self.alpha_F
    #     beta_F  = self.beta_F
    #     gamma   = self.model.gamma
    #     return A_F * N**gamma * k**alpha_F * n**beta_F

    def get_N(self):
        # If the city is in the bottom corner center_city is false, and effective population must be multiplied by 4
        # TODO think about whether this multiplier needs to come in elsewhere
        worker_agent_count = self.model.workforce.get_agent_count(self.model.workforce.workers)
        if self.model.center_city:
            N = self.density * worker_agent_count
        else:
            N = 4 * self.density * worker_agent_count
        # TODO handle divide by zero errors
        if N == 0:
            N = 1
        # TODO make sure all relevant populations are tracked - n, N, N adjusted x 4/not, agent count, N
        agglomeration_population = self.mult * N + self.seed_population
        return agglomeration_population

    def get_N_from_city_extent(self, city_extent):
        # agent_count = math.pi * (city_extent ** 2) #  Euclidian radius of the circular city
        agent_count = 2 * (city_extent ** 2)         #  Block metric radius of the circular city
        agglomeration_population = self.mult * self.density * agent_count + self.seed_population
        return agglomeration_population

class Bank(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos

    def get_reservation_price(self, R_N, r, r_target, m, p_dot, capital_gains_tax, transport_cost):
        # TODO is it the same as max bid?
        return self.get_max_desired_bid(R_N, r, r_target, m, p_dot, capital_gains_tax, transport_cost)

    def get_max_desired_bid(self, R_N, r, r_target, m, p_dot, capital_gains_tax, transport_cost):
        T      = self.model.mortgage_period
        delta  = self.model.delta
        # capital_gains_tax = self.model.capital_gains_tax # person and investor send.

        if R_N is not None and r is not None and r_target is not None and m is not None and p_dot is not None:
            R_NT   = ((1 + r)**T - 1) / r * R_N
            # return R_NT / ((1 - m) * r_target/(delta**T) - p_dot) 
            return (1 - capital_gains_tax) * R_NT / ((1 - m) * r_target/(delta**T) - p_dot +(1+r)**T*m) # Revised denominator from eqn 6:20

        else:
            self.model.logger.error(f'Get_max_desired_bid None error Rn {R_N}, r {r}, r_target {r_target}, m {m}, p_dot {p_dot}')
            return 0. # TODO Temp

    def get_average_wealth(self):
        rural_home_value     = self.get_rural_home_value()
        avg_locational_value = self.model.firm.wage_premium / (3 * self.model.r_prime)
        if not self.model.center_city:
            avg_locational_value = avg_locational_value/4
        return rural_home_value + avg_locational_value
    #   Should this be randomized? not everyone s at the same distance
    
        # AVERAGE_WEALTH_CALCULATION
        # The value of average_wealth.
        # # value of a home + savings half way through a lifespan.
        # # Value of house on average in the city - know the area and volume of a cone. Cone has weight omega, the wage_premium
        # avg_wealth = rural_home_value + avg_locational_value + modifier_for_other_cities_or_capital_derived_wealth
        # where:
        # avg_locational_value = omega / (3 * r_prime)
        # TODO consider adding modifier_for_other_cities_or_capital_derived_wealth
        # TODO check if we need to adjust if not center_city

    def get_rural_home_value(self):
        # r is the bank rate ussed to capitalie the value of housing services. a is the housing share, and a * subsistence_wage is the value of the housing services since we've fixed the subsistence wage and all houses are the same.
        a                = self.model.housing_services_share
        subsistence_wage = self.model.firm.subsistence_wage
        r                = self.model.r_prime
        return a * subsistence_wage / r

class Investor(Agent):

    # @property
    # def borrowing_rate(self):
    #     self.model.r_target
    
    def __init__(self, unique_id, model, pos, r_investor, capital_gains_tax, properties_owned = []):
        super().__init__(unique_id, model)
        self.pos = pos
        self.borrowing_rate = r_investor # self.model.r_target

        # Properties for bank as an asset holder
        # self.property_management_costs = property_management_costs # TODO 
        self.properties_owned      = properties_owned
        if not capital_gains_tax:
            self.model.logger.warning(f'No capital gains tax for investor {self.unique_id}.')
        self.capital_gains_tax     = capital_gains_tax

    def bid_on_properties(self):
        # """Investors bid on investment properties."""
        m = self.model.max_mortgage_share # mortgage share

        for listing in self.model.realtor.bids:
            R_N             = listing.sale_property.net_rent
            p_dot           = listing.sale_property.p_dot
            transport_cost  = listing.sale_property.transport_cost
            P_bid, bid_type = self.get_max_bid(m = m,
                                               R_N   = R_N, 
                                               p_dot = p_dot, 
                                               transport_cost = transport_cost)
            # P_bid, bid_type = get_max_bid(R_N=R_N, p_dot, transport_cost)
            # mortgage = m * P_bid
            self.model.logger.debug(f'Investor {self.unique_id} bids {P_bid} for property {listing.sale_property.unique_id}, if val is positive.')
            if P_bid > 0:
                self.model.realtor.add_bid(self, listing, P_bid, bid_type)
            else:
                self.model.logger.debug(f'Investor doesn\'t bid: {self.unique_id}')

    def get_max_bid(self, m, R_N, p_dot, transport_cost):
        r = self.borrowing_rate
        r_target = self.model.r_target
        P_bid    = self.model.bank.get_max_desired_bid(R_N, r, r_target, m, p_dot, self.capital_gains_tax, transport_cost)
        bid_type = 'investor'
        return P_bid, bid_type

    def __str__(self):
        return f'Investor {self.unique_id}'

class Realtor(Agent):
    """Realtor agents connect sellers, buyers, and renters."""
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        # self.model.workforce = self.model.workforce

        # self.sale_listings = []
        self.rental_listings = []

        self.bids = defaultdict(list)

    def step(self):
        pass

    def list_property_for_sale(self, seller, sale_property, reservation_price):
        listing = Listing(seller, sale_property, reservation_price)
        self.bids[listing] = []
        sale_property.p_dot

    def add_bid(self, bidder, listing, price, bid_type= ''):
        # Type check for bidder and property
        if not isinstance(bidder, (Person, Investor)):
            self.model.logger.error(f'Bidder in add_bid {bidder.unique_id} is not a Person or Investor. {bidder}')
        if not isinstance(listing, Listing):
            self.model.logger.error(f'Listing in add_bid is not a Listing instance.')
        if not isinstance(price, (int, float)):
            self.model.logger.error(f'Price in add_bid must be a numeric value (int or float).')
        if not isinstance(bid_type, (str)):
            self.model.logger.error(f'Bid type in add_bid must be a string.')
    
        bid = Bid(bidder, price, bid_type)
        self.bids[listing].append(bid)

    def sell_homes(self):
        # Allocate properties based on bids
        allocations = []

        self.model.logger.debug(f'Number of listed properties to sell: {len(self.bids)}')
        for listing, property_bids in self.bids.items():
            # # Look at all bids for debugging
            # bid_info = [f'bidder {bid.bidder} bid {bid.bid_price}' for bid in property_bids]
            # self.model.logger.debug(f'Listed property: {listing.sale_property.unique_id} has bids: {len(property_bids)}, {", ".join(bid_info)}')
            final_price = None
            reservation_price = listing.reservation_price
            if property_bids:
                property_bids.sort(key=lambda x: x.bid_price, reverse=True)
                highest_bid              = property_bids[0]
                highest_bid_price        = property_bids[0].bid_price
                second_highest_bid_price = property_bids[1].bid_price if len(property_bids) > 1 else 0

                using_reservation_price = False
                if using_reservation_price:
                    if highest_bid_price > reservation_price:
                        final_price = reservation_price + 0.5 * (highest_bid_price - reservation_price)
                        # TODO add more conditions: If 1 bid go half way between bid and reservation_price. If more bids go to 2nd highest bid
                    else:
                        self.model.logger.debug('Reservation price above bid for property {listing.sale_property.unique_id}')
                else:
                    final_price = highest_bid_price

            if final_price:
                allocation = Allocation(
                                        buyer                    = highest_bid.bidder,
                                        seller                   = listing.seller,
                                        sale_property            = listing.sale_property,
                                        final_price              = final_price,
                                        highest_bid_price        = highest_bid.bid_price,
                                        second_highest_bid_price = second_highest_bid_price,
                                        )
                allocations.append(allocation)

                # If buyer is a Person, they only buy one property and are removed from bidding on other properties
                if isinstance(allocation.buyer, Person):
                    self.model.logger.debug(f'Person has purchased, remove bids from other properties, {allocation.buyer.unique_id} {allocation.sale_property.pos}')
                    for property_bids in self.bids.values():
                        property_bids[:] = [bid for bid in property_bids if bid.bidder != allocation.buyer]

            else:
                # If no allocation rent homes TODO they could keep the home on the market
                # self.model.logger.debug('No allocation')
                self.model.logger.debug(f'Property {listing.sale_property.unique_id}, {listing.sale_property.pos} NOT sold by seller {listing.seller}')
                # List property  to rent it to a newcomer
                self.rental_listings.append(listing.sale_property)
                # Track ownership with retired_agents
                self.model.retired_agents.add_property(listing.seller.unique_id, listing.sale_property)
                listing.sale_property.owner = self.model.retired_agents
                # Remove retiring agent from the model
                listing.seller.remove()

        # Complete transactions for all listings and clear bids
        self.complete_transactions(allocations)
        self.bids.clear()
        # self.sale_listings.clear() # TODO check do we clear listings here?
        return allocations # TODO returning for testing. Do we need this? Does it interfere with main code?

    def complete_transactions(self, allocations):
        for allocation in allocations:
            self.model.logger.debug(f'Property {allocation.sale_property.unique_id}, {allocation.sale_property.pos} sold by seller {allocation.seller} to {allocation.buyer} for {allocation.final_price}')
            # if not isinstance(allocation.seller, Person):
            #     self.model.logger.debug(f'Seller not a Person in complete_transaction: Seller {allocation.seller.unique_id}')

            # Record data for data_collection
            allocation.sale_property.realized_price = allocation.final_price
            allocation.sale_property.realized_all_steps_price = allocation.final_price # Record that it sold

            # Record data for forecasting
            new_row = {
            'land_id':        allocation.sale_property.unique_id,
            'realized_price': allocation.final_price,
            'time_step':      self.model.time_step,
            'transport_cost': allocation.sale_property.transport_cost,
            'wage':           self.model.firm.wage,
            }

            # if isinstance(allocation.seller, Person):
            #     pass
            #     # self.handle_seller_departure(allocation)
            # elif isinstance(allocation.seller, Investor):
            #     self.model.logger.debug(f'In complete_transaction, before purchase, seller is Investor, id {allocation.seller.unique_id}.')
            # else:
            #     self.model.logger.debug(f'In complete_transaction, before purchase, seller {allocation.seller.unique_id} was not a person or investor. Seller {allocation.seller}.')

            # self.model.logger.debug(f'Time {self.model.time_step}, Property {allocation.property.unique_id}, Price {allocation.property.realized_price}')
            if isinstance(allocation.buyer, Investor):
                self.handle_investor_purchase(allocation)
            elif isinstance(allocation.buyer, Person):
                self.handle_person_purchase(allocation)
            else:
                self.model.logger.warning('In complete_transaction, buyer was neither a person nor an investor.')

            self.handle_seller_departure(allocation)
            # if isinstance(allocation.seller, Person):
            #     self.model.logger.debug(f'Seller departure {allocation.seller}')
                
            # elif isinstance(allocation.seller, Investor):
            #     self.model.logger.debug(f'In complete_transaction, after purchase, seller is Investor, id {allocation.seller.unique_id}.')
            # else:
            #     self.model.logger.debug(f'In complete_transaction, after purchase, seller {allocation.seller.unique_id} was not a person or investor. Seller {allocation.seller}.')                

    def handle_investor_purchase(self, allocation):
        """Handles the purchase of a property by an investor."""
        allocation.sale_property.change_owner(allocation.buyer, allocation.seller)
        allocation.sale_property.resident = None
        allocation.buyer.residence = None
        self.model.logger.debug('Property %s sold to investor.', allocation.sale_property.unique_id)
        self.rental_listings.append(allocation.sale_property)

    def handle_person_purchase(self, allocation):
        """Handles the purchase of a property by a person."""
        allocation.sale_property.change_owner(allocation.buyer, allocation.seller)
        # if not allocation.sale_property.check_residents_match:
        #     self.model.logger.error(f'Sale property residence doesn\'t match before transfer: seller {seller.unique_id}, buyer {buyer.unique_id}')
        allocation.sale_property.resident = allocation.buyer
        allocation.buyer.residence = allocation.sale_property
        self.model.grid.move_agent(allocation.buyer, allocation.sale_property.pos)
        allocation.buyer.purchased_property = True
        # if not allocation.sale_property.check_residents_match:
        #     self.model.logger.error(f'Sale property residence doesn\'t match after transfer: seller {seller.unique_id}, buyer {buyer.unique_id}')
        self.model.logger.debug('Property %s sold to newcomer %s, new loc %s.', allocation.sale_property.unique_id, allocation.buyer.unique_id, allocation.buyer.pos)

        # if self.residence: # Already checked since residence assigned
        if self.model.firm.wage_premium > allocation.sale_property.transport_cost:
            self.model.workforce.add(allocation.buyer, self.model.workforce.workers)
            self.model.logger.debug(f'Person purchase: add person to workforce')
        else:
            self.model.logger.debug(f'Person purchase: don\'t add person to workforce')

        if allocation.buyer.unique_id in self.model.workforce.newcomers:
            self.model.logger.debug(f'Remove from newcomers list {allocation.buyer.unique_id}')
            self.model.workforce.remove(allocation.buyer, self.model.workforce.newcomers)
        else:
            self.model.logger.warning(f'Person buyer was not a newcomer: {allocation.buyer.unique_id}')
        # self.model.logger.debug(f'Time {self.model.time_step} New worker {buyer.unique_id} Loc {sale_property}') # TEMP

    def handle_seller_departure(self, allocation):
        """Handles the departure of a selling agent."""
        if isinstance(allocation.seller, Person):
            if allocation.seller.unique_id in self.model.workforce.retiring_urban_owner:
                self.model.logger.debug(f'Removing seller {self.unique_id}')
                allocation.seller.remove()
            else:
                self.model.logger.warning(f'Seller a Person but not retiring, so not removed: Seller {allocation.seller.unique_id}')
        elif isinstance(allocation.seller, Investor):
            self.model.logger.warning(f'Seller an Investor in handle_seller_departure: Seller {allocation.seller.unique_id}')
        else:
            self.model.logger.warning(f'Seller not an Investor or a Person in handle_seller_departure: Seller {allocation.seller.unique_id}')

    def rent_homes(self):
        """Rent homes listed by investors to newcomers."""
        self.model.logger.debug(f'{len(self.rental_listings)} properties to rent.')
        for rental in self.rental_listings:
            renter = self.model.create_newcomer(pos = rental.pos)
            rental.resident = renter
            renter.residence = rental
            self.model.workforce.remove(renter, self.model.workforce.newcomers)
            self.model.logger.debug(f'Newly created renter {renter.unique_id} lives at '
                         f'property {renter.residence.unique_id} which has '
                         f'resident {rental.resident.unique_id}, owner {rental.owner.unique_id}, pos {renter.pos}.')
            
            renter.work_if_worthwhile_to_work()
        self.rental_listings.clear()

class Bid_Storage(Agent):
    """Stores bids in fast run."""
    # Must be created after firm
    def __init__(self, unique_id, model, pos,
                 bidder_name,
                 distance_from_center,
                 bidder_savings = 0,
                ):
        super().__init__(unique_id, model) 
        self.bidder_name    = bidder_name
        self.bidder_savings = bidder_savings
        self.distance_from_center = distance_from_center
        self.transport_cost = self.model.property.calculate_transport_cost(self.distance_from_center)
        
        self.bid_type  = None # TODO this would stay constant
        self.bid_value = 0
        self.R_N       = 0
        self.density   = self.model.firm.density
         
        # m
        # M
        # p_dot
        # transport_cost, savings_value
    def step(self):
        # TODO this part could be done just once for each dist in each time step to speed up..
        self.model.property.change_dist(self.distance_from_center)
        self.R_N             = self.model.property.net_rent
        self.p_dot           = self.model.property.p_dot
        transport_cost  = self.model.property.transport_cost
        m  = self.model.max_mortgage_share
        if self.bidder_name == 'Investor':
            self.bid_value,  self.bid_type = self.model.investor.get_max_bid(m = m,
                            R_N   = self.R_N,
                            p_dot = self.p_dot,
                            transport_cost = transport_cost)
        else: # TODO CHECK THIS IS actually person type and catch errors
            M     = self.model.person.get_max_mortgage(self.bidder_savings)
            self.bid_value,  self.bid_type = self.model.person.get_max_bid(m, M, self.R_N, self.p_dot, transport_cost, self.bidder_savings)
            # self.step_data["newcomer_bid"].append((round(newcomer_bid, self.no_decimals), round(dist, self.no_decimals), round(savings_value, self.no_decimals)))
            # dist += 1

        # TODO dynamically generate for all the savings bids
        # savings_1_bid = 

class Listing:
    def __init__(
        self, 
        seller: Union[Person, Investor],
        sale_property: Land, 
        reservation_price: Union[float, int] = 0.0,
    ):
        if not isinstance(seller, (Person, Investor)):
            self.model.logger.error(f'Bidder in Listing {seller.unique_id} is not a Person or Investor, {seller}')
        if not isinstance(sale_property, Land):
            self.model.logger.error(f'sale_property in Listing {sale_property.unique_id} is not Land, {sale_property}')
        if not isinstance(reservation_price, (float, int)):
            self.model.logger.error(f'reservation_price in Listing must be a numeric value.')
               
        self.seller = seller
        self.sale_property = sale_property
        self.reservation_price = reservation_price

    def __str__(self):
        return f'Seller: {self.seller}, Property: {self.sale_property}, List Price: {self.reservation_price}'

class Bid:
    def __init__(
        self, 
        bidder: Union[Person, Investor],
        bid_price: Union[float, int],
        bid_type: str = '',
    ):
        if not isinstance(bidder, (Person, Investor)):
            self.model.logger.error(f'Bidder in Bid {bidder.unique_id} is not a Person or Investor, {bidder}')
        if not isinstance(bid_price, (float, int)):
            self.model.logger.error(f'Price in Bid must be a numeric value.')
        if not isinstance(bid_type, (str)):
            self.model.logger.error(f'Bid type must be a string.')
               
        self.bidder      = bidder
        self.bid_price   = bid_price
        self.bid_type    = bid_type

    def __str__(self):
        return f'Bidder: {self.bidder.unique_id}, Price: {self.price}, Type: {self.bid_type}'

class Allocation:
    def __init__(
        self, 
        buyer: Union[Person, Investor],
        seller: Union[Person, Investor],
        sale_property: Land,
        final_price: Union[float, int] = 0.0,
        highest_bid_price: Union[float, int] = 0.0,
        second_highest_bid_price: Union[float, int] = 0.0,
    ):
        if not isinstance(buyer, (Person, Investor)):
            self.model.logger.error(f'Successful buyer {buyer.unique_id} in Allocation must be of type Person or Investor.')
        if not isinstance(seller, (Person, Investor)):
            self.model.logger.error(f'Seller {seller.unique_id} in Allocation must be of type Person or Investor.')
        if not isinstance(sale_property, Land):
            self.model.logger.error(f'Property {property.unique_id} in Allocation must be of type Land.')
        if not isinstance(final_price, (float, int)):
            self.model.logger.error(f'Final price in Allocation must be a numeric value.')
        if not isinstance(highest_bid_price, (float, int)):
            self.model.logger.error(f'Highest bid in Allocation must be a numeric value.')
        if not isinstance(second_highest_bid_price, (float, int)):
            self.model.logger.error(f'Second highest bid in Allocation must be a numeric value.')

        self.buyer                    = buyer
        self.seller                   = seller
        self.sale_property            = sale_property
        self.final_price              = final_price
        self.highest_bid_price        = highest_bid_price
        self.second_highest_bid_price = second_highest_bid_price

    def __str__(self):
        return f'Buyer: {self.buyer}, Seller {self.seller.unique_id}, Property: {self.sale_property}, Final Price: {self.final_price}, Highest Bid: {self.highest_bid_price}, Second Highest Bid: {self.second_highest_bid_price}'
