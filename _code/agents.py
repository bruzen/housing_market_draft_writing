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
        return self.warranted_rent - self.maintenance - self.property_tax
    
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

        if (self.model.firm.wage_premium > self.transport_cost):
            self.p_dot       = self.model.firm.p_dot
        else:
            self.p_dot       = None 

        self.realized_price  = - 1 # Reset to show realized price in just this time_step

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
        subsistence_wage  = self.model.firm.subsistence_wage
        return a * b * subsistence_wage

    def get_warranted_rent(self):
        wage_premium     = self.model.firm.wage_premium
        subsistence_wage = self.model.firm.subsistence_wage
        a                = self.model.housing_services_share
        # Note, this is the locational_rent. It is the warranted level of extraction. It is the economic_rent when its extracted.
        # We set the rural land value to zero to study the urban land market, with the agricultural price, warranted rent would be:
        warranted_rent = wage_premium - self.transport_cost + a * subsistence_wage #### TODO add amenity here + A
        return max(warranted_rent, 0)

    def get_warranted_price(self):
        return self.warranted_rent / self.model.r_prime

    def check_owners_match(self):
        for owned_property in self.owner.properties_owned:
            if self.unique_id == owned_property.unique_id:
                return True
        return False
 
    def change_dist(self, dist):
        self.model.logger.warning(f'Change distance from center for land {self.unique_id}') # Used in model_fast
        self.distance_from_center     = dist
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
        borrowing rate. It depends on the agent's wealth. A resonable value
        might be around 0.002.
 
        Formula for interest rate they get: r_target + K/(W-W_min) - K/(W_avg-W_min)
        Formula for adjustment: K/(W-W_min) - K/(W_avg-W_min)
        K is wealth sensitivity parameter

        Returns:
        The individual wealth adjustment value.
        """
        K        = self.model.wealth_sensitivity
        W        = self.get_wealth() 
        W_min    = 10000. # TODO could make a parameter be 0 or 20K
        W_avg    = self.model.bank.get_average_wealth()
        return K / (W - W_min) - K / (W_avg - W_min)

    def __init__(self, unique_id, model, pos, init_working_period = 0,
                 savings = 0., debt = 0.,
                 capital_gains_tax = None,
                 residence_owned = None,):
        super().__init__(unique_id, model)
        self.pos = pos

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
        # self.is_working_check    = 0 # TODO delete?
        self.expectations        = 1.0 # Set parameter for investor_expectations, constant for person


    def step(self):
        self.count              += 1
        self.working_period     += 1
        premium                  = self.model.firm.wage_premium

        # Non-residents
        if not isinstance (self.residence, Land):
            # Newcomers who don't find a home leave the city
            if (self.unique_id in self.model.workforce.newcomers):
                if self.count > 0:
                    self.model.logger.debug(f'Newcomer removed {self.unique_id}')
                    self.remove()
                    return
            # Everyone who is not a newcomer leaves if they have no residence
            else:
                self.model.logger.warning(f'Non-newcomer with no residence removed: {self.unique_id}, count {self.count}')
                self.remove()
                return

        # Urban workers
        elif premium > self.residence.transport_cost:
            self.model.workforce.add(self, self.model.workforce.workers)
            # Agents list properties a step before they stop working for consistent worker numbers
            if self.working_period >= self.model.working_periods:
                if self.residence in self.properties_owned:
                    self.model.workforce.add(self, self.model.workforce.retiring_urban_owner)
                    reservation_price = self.model.bank.get_reservation_price(
                        R_N = self.residence.net_rent, 
                        r = self.model.r_prime, 
                        r_target = self.model.r_target, 
                        m =  self.model.max_mortgage_share,
                        p_dot =  self.residence.p_dot, 
                        capital_gains_tax = self.capital_gains_tax,
                        transport_cost = self.residence.transport_cost,
                        expectations   = self.expectations)
                    self.model.realtor.list_property_for_sale(self, self.residence, reservation_price)
                    # TODO Extend so agents can sell, rent or keep empty
                    self.model.logger.debug(f'Person retiring: {self.unique_id}, {self.residence.unique_id}-{self.residence.pos}, reservation price {reservation_price}')

            if self.working_period > self.model.working_periods:
                if self.residence in self.properties_owned:
                    self.model.workforce.remove(self, self.model.workforce.workers)
                    self.model.logger.warning(f'Urban homeowner still in model: {self.unique_id}, working_period {self.working_period}')
                # Renters: This resets keeping the initial distribution of ages cycling for renters in the city, assuming savings built over the lifetime, no inheritance
                else:
                    self.working_period = 1
                    self.savings        = 0

        # Rural population: This resets keeping the initial distribution of ages cycling for renters in the city, assuming savings built over the lifetime, no inheritance
        else:
            self.model.workforce.remove(self, self.model.workforce.workers)
            if self.working_period > self.model.working_periods:
                self.working_period = 1
                self.savings        = 0
                
        # Update savings # TODO update wealth, mortgages, etc.
        self.savings += self.model.savings_per_step

        # TODO consider doing additional check for retiring agents who are still in model
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
        #     if premium > self.residence.transport_cost:
        #         self.is_working_check = 1
        #     else:
        #         self.is_working_check = 0

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
        # TODO can use wealth in place of savings
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

        # Calculate bid value under each constraint        
        value_bid = self.model.bank.get_max_desired_bid(R_N, r, r_target, m, p_dot, self.capital_gains_tax, transport_cost, self.expectations)
        equity_bid = S/(1-m)
        income_bid = 0.28 * (wage + r * S) / r_prime
        #income_bid = 0.28 * (wage + r * S) /( 1+0.28*r) * m   #BETTER see chapter model  line 444 

        # Determine bid type
        P_bid = value_bid
        if equity_bid <= P_bid:
            bid_type = 'value_limited'
            P_bid    = value_bid

        if income_bid  <= P_bid:
            bid_type = 'equity_limited'
            P_bid    = income_bid

        else:
            bid_type = 'value_limited' # and bid remains value_limited

        bid_dict = {'value': value_bid, 'equity': equity_bid, 'income': income_bid}
        self.model.logger.debug(f'get_max_bid returns: {bid_type} {P_bid} bid, for agent {self.unique_id}, \nbids: {bid_dict} \n')
        return P_bid, bid_type

        # Old
        # bid_type = 'value_limited'
        # P_bid    = self.model.bank.get_max_desired_bid(R_N, r, r_target, m, p_dot, self.capital_gains_tax, transport_cost, self.expectations)
        # # self.model.logger.warning(f'Max bid: {self.unique_id}, bid {P_bid}, R_N {R_N}, r {r}, r {r_target}, m {m}, transport_cost {transport_cost}')

        # if S/(1-m) <= P_bid:
        #     bid_type = 'equity_limited'
        #     P_bid = S/(1-m)
        #     self.model.logger.warning(f'Newcomer bid EQUITY LIMITED: {self.unique_id}, bid {P_bid}') #, S {S}, m {m}, .. ')

        # if (0.28 * (wage + r * S) / r_prime)  <= P_bid:
        #     bid_type = 'income_limited'
        #     P_bid = 0.28 * (wage + r * S) / r_prime
        #     self.model.logger.warning(f'Newcomer bid INCOME LIMITED: {self.unique_id}, bid {P_bid}')

        # # TODO do we want to not bid if value is negative?
        # # if P_bid < 0:
        # #     bid_type = 'negative'
        # #     # P_bid = 0
        # #     self.model.logger.warning(f'Newcomer bid is NEGATIVE: {self.unique_id}, bid {P_bid}')

        # else:
        #     # bid_type = 'none'
        #     self.model.logger.warning(f'Newcomer bid is VALUE LIMITED: {self.unique_id}, bid {P_bid}') #, sale_property {listing.sale_property.unique_id}')
        # return P_bid, bid_type
    


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
        # Wealth is properties owned, minus mortgages owed, plus savings.
        # Assume newcomers arrive not owning property. TODO update if property owners bid
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
                 adjs,
                 adjd,
                 adjp,
                 dist,
                 init_F,
                 init_k,
                 init_n,
                 ):
        super().__init__(unique_id, model)
        self.pos             = pos

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
        self.adjN     = adjN
        self.adjk     = adjk
        self.adjn     = adjn
        self.adjF     = adjF
        self.adjw     = adjw
        self.adjd     = adjd
        self.adjs     = adjs
        self.adjp     = adjp
        self.dist     = dist
        self.r        = r_prime # Firm cost of capital

        # Initial values # TODO do we need all these initial values?
        self.y        = 100000
        self.Y        = 0
        self.F        = init_F
        self.k        = init_k #1.360878e+09 #100
        self.n        = init_n
        self.F_target = init_F
        self.n_target = init_n
        self.wage_premium     = init_wage_premium_ratio * self.subsistence_wage 
        self.old_wage_premium = 8000 # -1 # init_wage_premium_ratio * self.subsistence_wage   ### REVISED should remove inital problems
        self.wage             = (1 + init_wage_premium_ratio) * self.subsistence_wage
        self.wage_target      = self.wage
        self.MPL              = 7200 # self.beta  * self.y / self.n  # marginal value product of labour known to firms
        self.worker_demand    = self.F * self.n
        self.worker_supply    = self.F * self.n
        self.agglom_pop       = self.F * self.n 
        self.p_dot            = 0 # TODO fix init p_dot
        self.N                = self.F * self.n
        self.A_time           = self.model.schedule.time
        # N_target   = self.N #  this is an initializaton it can't be in step or N_tarfget is meaningless 

        
    def step(self):
       
        # STORE INITIAL VALUES FOR CALCULATING CHANGES
        self.A_time = self.model.schedule.time
        self.y          = self.A * self.agglom_pop**self.gamma *  self.k**self.alpha * self.n**self.beta
        self.MPL        =  self.beta + self.y /self.n 
        N_target   = self.N #  this is an initializaton it can't be in step or N_tarfget is meaningless 

        self.N        = (1 - self.adjN) * self.N + self.adjN * N_target   #
        n_old        = self.n
        wage_old     = self.wage
        ov         = 1 + self.overhead  # overhead ratio
        VMPL       = self.price_of_output * self.MPL
        revenue    = self.price_of_output * self.y 
        cost       = self.r * self.k + self.wage * self.n
        profit     = revenue - cost
        profit_ratio          = revenue / cost
        self.old_wage_premium = self.wage_premium
        #N_target   = self.N #        NEW    *****
        # self.N        = (1 - self.adjN) * self.N + self.adjN * N_target   #
        self.model.model_description = "Checking bidding behaviour"

 ############   MODEL 1  --- MARCH 1   ############ Use this version
        self.model.model_description = "March 2 ensure we still have the model working"
    
        self.k_target = self.price_of_output * self.alpha * self.y/self.r     #(old optimal version)
        self.n_target   = 5 * (self.beta* self.A * self.agglom_pop**self.gamma *  self.k**self.alpha )**(1-self.beta) 
        self.worker_demand = self.n_target * self.F # self.n_target * self.F_target
        edr = (self.worker_demand - self.worker_supply) / max(abs(self.worker_demand), abs(self.worker_supply)) #positive or negative 
        self.F = self.worker_supply / self.n  #moved this up two lines
        self.wage_target =  VMPL / ov
        self.wage        = self.wage_target  #(1 - self.adjw) * self.wage_target + self.adjw * self.wage_target
        
        #INCREMENT STATE VARIABLES TOWARDS TARGETS
        self.n        = (1 - self.adjn) * self.n + self.adjn * self.n_target
        self.k        = (1 - self.adjk) * self.k + self.adjk * self.k_target 
        # self.F      = (1 - self.adjF) * self.F + self.adjF * self.F_target
        self.wage     = (1 - self.adjw) * self.wage + self.adjw * self.wage_target  #reintroduced  - didn't help
        #self.y       = (1 - self.adjy) * self.y + self.adjy * self.y*F_target 

        self.wage_premium     = self.wage - self.subsistence_wage # find wage available for transportation
        self.p_dot            = self.get_p_dot()
        # COMMENT: wage goes to 100,000 with 5 in n_target  Mult=1, but n=12
      
     ############   MODEL 2  --- MARCH 3   ############ 
        # self.model.model_description = "March 4 experimensts wit wage"
        # self.k_target = self.price_of_output * self.alpha * self.y/self.r     #(old optimal version)
        # self.n_target   = 5 * (self.beta* self.A * self.agglom_pop**self.gamma *  self.k**self.alpha )**(1-self.beta) 
        # self.worker_demand = self.n_target * self.F # self.n_target * self.F_target
        # edr = (self.worker_demand - self.worker_supply) / max(abs(self.worker_demand), abs(self.worker_supply)) #positive or negative 
        # self.F = self.worker_supply / self.n  #moved this up two lines
        # # self.wage_target = (1+ edr) * VMPL / ov  # this is INNOVATION 1:  March 3 It pushes demand and supply up and extent
        # self.wage_target =  VMPL / ov
        # # self.wage        = self.wage_target  #(1 - self.adjw) * self.wage_target + self.adjw * self.wage_target
        # self.wage        = (1 - self.adjw) * self.wage + self.adjw * self.wage_target 
        # #                                       this is the INNOVATION 2 seems to have no effect
        # # I thought adjw -> 1-adjw makes this the standard case when adjw is 0.002. 
        
        # # INCREMENT STATE VARIABLES TOWARDS TARGETS 
        # self.n        = (1 - self.adjn) * self.n + self.adjn * self.n_target
        # self.k        = (1 - self.adjk) * self.k + self.adjk * self.k_target 
        # # self.F        = (1 - self.adjF) * self.F + self.adjF * self.F_target
        # #self.wage     = (1 - self.adjw) * self.wage + self.adjw * self.wage_target  #reintroduced  - didn't help
        # #self.y        = (1 - self.adjy) * self.y + self.adjy * self.y*F_target 

        # self.wage_premium     = self.wage - self.subsistence_wage # find wage available for transportation
        # self.p_dot            = self.get_p_dot()
        # # COMMENT: mult=1 1-> 1.2 no dif? density increases F, beta[0.73, .78]increases wneFN low alpha very inhibiting
        # # COMMENT:adjN no effect if adj k high, overshoot adjn no effect


     ############   MODEL 3  --- MARCH 2   ############ 
        #  self.model.model_description = "March 2 ensure we still have the modle working"        self.k_target = self.price_of_output * self.alpha * self.y/self.r     #(old optimal version)
        # self.n_target   = 1 * (self.beta* self.A * self.agglom_pop**self.gamma *  self.k**self.alpha )**(1-self.beta) 
        # self.worker_demand = self.n_target * self.F # self.n_target * self.F_target
        # edr = (self.worker_demand - self.worker_supply) / max(abs(self.worker_demand), abs(self.worker_supply)) #positive or negative 
        # self.F = self.worker_supply / self.n  #moved this up two lines
        # self.wage_target =  VMPL / ov
        # self.wage        = self.wage_target  #(1 - self.adjw) * self.wage_target + self.adjw * self.wage_target
        
        # #TODO INCREMENT STATE VARIABLES TOWARDS TARGETS     NOT USED  REMOVE?
        # self.n        = (1 - self.adjn) * self.n + self.adjn * self.n_target
        # self.k        = (1 - self.adjk) * self.k + self.adjk * self.k_target 
        # # self.F        = (1 - self.adjF) * self.F + self.adjF * self.F_target
        # self.wage     = (1 - self.adjw) * self.wage + self.adjw * self.wage_target  #reintroduced  - didn't help
        # #self.y        = (1 - self.adjy) * self.y + self.adjy * self.y*F_target 

        # self.wage_premium     = self.wage - self.subsistence_wage # find wage available for transportation
        # self.p_dot            = self.get_p_dot()


        ##      k target _____________________________________________________________#  -> #* in model 1
        #      kopt) --- Optimal k calculation (two versions)
        #* self.k_target = self.price_of_output * self.alpha * self.y/self.r     #(old optimal version)
        # self.k_target = (self.r/(self.price_of_output * self.alpha * self.A * self.agglom_pop**self.gamma *  self.n**self.beta) )**(1-self.alpha)        
        
        #     kprofit) --- Profit-based adjustment
        # self.k_target =  profit_ratio * self.k 
        
        #     kold) --- Profit-based adjustment
        # self.k_target = (self.alpha * self.y) /self.r

        # #     WAGE OFFER ________ 2 versions______________moving this up makes demand respond instantly
        # ##     w1) --- BASED ON EXCESS DEMAND
        # # self.wage = (1 + self.adjw *edr)*self.wage
        # ##     w2) --- BASED ON MPL  (respond directly to flaw in behavour??)
        # self.wage_target =  VMPL / ov
        #* self.wage        = (1 - self.adjw) * self.wage_target + self.adjw * self.wage_target
        

        ##     n_target _________ 3 versions_________________________________________

        #     nopt 1) --- setting  the optimal number of worker(s using wage=vMPL 
        # self.n_target   = self.beta*
        # self.n_target   =  (self.beta * revenue)/(1+ self.overhead)*self.wage  # This explodes
        #     nopt 2) --- setting  the optimal number of worker(s using wage=vMPL 
        #* self.n_target   = 5 * (self.beta* self.A * self.agglom_pop**self.gamma *  self.k**self.alpha )**(1-self.beta)         

        #     n1) --- Profit-ratio-based adjustment     
        # self.n_target = profit_ratio * self.n #  THIS gives us F crashing  try longer run - may work out
        # change_n = n_target - self.n
        # self.n_target      = (1 - self.adjn) * self.n + self.adjn * self.n_target # Firm plans a partial adjustment and posts employment target
        
        #     n2) --- Profit-based adjustment provisionaly using all profit for new labour. Both updated to end of previous period   
     
        # self.n_target        = self.n + change_n
        # _________  # same adjustment for all 3 versions of n_target ______
       
        ##     F target ___________ 3 versions________________________________________
        ##     F1) --- Entreprenur uses profit signal measured as new labour  in n1 for entry/exit decisions. 
        # self.F_target = self.F * (1 + change_n / self.n)
        # self.F_target = (1 - self.adjF) * self.F + self.adjF * self.F_target # Entrepreneur  plans a partial adjustment and posts employment target 
        ##     F2) --- Entreprenur grows firm to set P*MPL = wage. This means that all firms are of size n_target
        #self.F_target=self.N/self.n_target   # "NOT COMPATABLE" WITH  ALLOCATE LABOUR TO FIRMS (below)  CHECK 
        
        # IDENTIFY AGGREGATE INDUSTRY DEMAND FOR LABOUR 
        #self.worker_demand = self.n_target * self.F # self.n_target * self.F_target

        # DEFINE THE EXCESS DEMAND RATIO
        #edr = (self.worker_demand - self.worker_supply) / max(abs(self.worker_demand), abs(self.worker_supply)) #positive or negative
        
        # APPLY SHORT-SIDE RULE  (to find out how many CAN be employed) ___
        # self.N =   # selfmin(self.worker_demand, self.worker_supply)

        # ALLOCATE LABOUR TO FIRMS (All firms get equal labour, have equal MPL)
        # N = self.n * self.F     # "NOT COMPATABLE" WITH  F2  CHECK 
        #self.F = self.worker_supply / self.n     
        
        # #TODO INCREMENT STATE VARIABLES TOWARDS TARGETS     NOT USED  REMOVE?
        # self.n        = (1 - self.adjn) * self.n + self.adjn * self.n_target
        # self.k        = (1 - self.adjk) * self.k + self.adjk * self.k_target 
        # self.F        = (1 - self.adjF) * self.F + self.adjF * self.F_target
        # self.wage     = (1 - self.adjw) * self.wage + self.adjw * self.wage_target
        # #self.y        = (1 - self.adjy) * self.y + self.adjy * self.y*F_target 


        #     WAGE OFFER ________ 2 versions______________moving this up makes demand respond instantly
        ##     w1) --- BASED ON EXCESS DEMAND
        # self.wage = (1 + self.adjw *edr)*self.wage
        # ##     w2) --- BASED ON MPL  (respond directly to flaw in behavour??)
        # self.wage_target =  VMPL / ov
        # self.wage        = (1 - self.adjw) * self.wage_target + self.adjw * self.wage_target



    def get_worker_supply(self, city_extent = None):
        # Fast model calculates worker supply based on city_extent
        if city_extent:
            # agent_count = math.pi * (city_extent ** 2)  #  Euclidian radius of the circular city
            agent_count   = 2 * (city_extent ** 2)        #  Block metric radius of the circular city
            worker_supply = self.density * agent_count + self.seed_population
        # Main model counts the workers who choose to work
        else:
            agent_count = self.model.workforce.get_agent_count(self.model.workforce.workers)
            # If the city is in the bottom corner center_city is false, and effective population must be multiplied by 4
            if self.model.center_city:
                worker_supply = self.density * agent_count + self.seed_population
            else:
                worker_supply = 4 * self.density * agent_count + self.seed_population
        # Avoid divide by zero errors
        if worker_supply == 0:
            worker_supply = 1
        return worker_supply

    def get_agglomeration_population(self, worker_supply):  # INCLUDE NON WORKER POOPULATION
        return self.mult * (worker_supply)

    def get_p_dot(self):
        try:
            p_dot = ((self.model.firm.wage_premium / self.model.firm.old_wage_premium)**self.model.mortgage_period - 1)/self.r
        except ZeroDivisionError:
            # Handle division by zero
            p_dot = None
            self.model.logger.error(f"ZeroDivisionError at time_step {self.model.schedule.time} for Land ID {self.unique_id}, old_wage_premium {self.model.firm.old_wage_premium}")
        except Exception as e:
            # Handle other exceptions
            self.model.logger.error(f"An error occurred: {str(e)}")
            p_dot = None
        return p_dot

class Bank(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos

    def get_reservation_price(self, R_N, r, r_target, m, p_dot, capital_gains_tax, transport_cost, expectations):
        # The reservation price follows the same equation as max_bid
        return self.get_max_desired_bid(R_N, r, r_target, m, p_dot, capital_gains_tax, transport_cost, expectations)

    def get_max_desired_bid(self, R_N, r, r_target, m, p_dot, capital_gains_tax, transport_cost, expectations):
        T      = self.model.mortgage_period
        delta  = self.model.delta
        # capital_gains_tax = self.model.capital_gains_tax # person and investor send.

        if R_N is not None and r is not None and r_target is not None and m is not None and p_dot is not None:
            R_NT   = (((1 + r)**T - 1) / r) * R_N
            # return R_NT / ((1 - m) * r_target/(delta**T) - p_dot) 
            return (1 - capital_gains_tax) * R_NT / ((1 - m) * r_target/(delta**T) - expectations * p_dot +(1+r)**T*m) # Revised denominator from eqn 6:20

        else:
            self.model.logger.error(f'Get_max_desired_bid None error Rn {R_N}, r {r}, r_target {r_target}, m {m}, p_dot {p_dot}')
            return 0.

    def get_average_wealth(self):
        rural_home_value     = self.get_rural_home_value()
        avg_locational_value = self.model.firm.wage_premium / (3 * self.model.r_prime)
        if not self.model.center_city:
            avg_locational_value = avg_locational_value/4
        return rural_home_value + avg_locational_value

        # The value of average_wealth is the value of a home + savings half way through a lifespan.
        # Value of house on average in the city, since we know the area and volume of a cone.
        # avg_wealth = rural_home_value + avg_locational_value + modifier_for_other_cities_or_capital_derived_wealth
        # TODO consider adding modifier_for_other_cities_or_capital_derived_wealth


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
    
    def __init__(self, unique_id, model, pos, r_investor, capital_gains_tax, investor_expectations, investor_turnover, properties_owned = []):
        super().__init__(unique_id, model)
        self.pos = pos
        self.borrowing_rate    = r_investor # self.model.r_target
        self.investor_turnover = investor_turnover

        # Properties for bank as an asset holder
        # self.property_management_costs = property_management_costs # TODO 
        self.properties_owned      = properties_owned
        if not capital_gains_tax:
            self.model.logger.warning(f'No capital gains tax for investor {self.unique_id}.')
        self.capital_gains_tax     = capital_gains_tax
        self.expectations          = investor_expectations

    def list_properties(self):
        no_props = len(self.properties_owned)
        # print(f'Time {self.model.schedule.time}, no_props {no_props}')
        for prop in self.properties_owned:
            if self.model.random.random() < self.investor_turnover: # 5% chance
                # print(f'Time {self.model.schedule.time}, List investor property {prop.unique_id}, no_props {no_props}')
                # print(f'List investor property {prop.unique_id}')
                reservation_price = self.model.bank.get_reservation_price(
                    R_N = prop.net_rent, 
                    r = self.model.r_prime, 
                    r_target = self.model.r_target, 
                    m =  self.model.max_mortgage_share,
                    p_dot =  prop.p_dot, 
                    capital_gains_tax = self.capital_gains_tax,
                    transport_cost = prop.transport_cost,
                    expectations   = self.expectations)

                # List the property for sale
                self.model.realtor.list_property_for_sale(self, prop, reservation_price)

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
        P_bid    = self.model.bank.get_max_desired_bid(R_N, r, r_target, m, p_dot, self.capital_gains_tax, transport_cost, self.expectations)
        bid_type = 'investor'
        self.model.logger.debug(f'get_max_bid returns: {bid_type} {P_bid}\n')
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
                    if listing.seller != highest_bid.bidder: 
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
                if isinstance(listing.seller, Person):
                    # self.model.logger.debug('No allocation')
                    self.model.logger.debug(f'Property {listing.sale_property.unique_id}, {listing.sale_property.pos} NOT sold by seller {listing.seller}')
                    # List property  to rent it to a newcomer
                    self.rental_listings.append(listing.sale_property)
                    # Track ownership with retired_agents
                    self.model.retired_agents.add_property(listing.seller.unique_id, listing.sale_property)
                    listing.sale_property.owner = self.model.retired_agents
                    # Remove retiring agent from the model
                    listing.seller.remove()
                #  TODO fix this
                # if isinstance(listing.seller, Investor):
                #     # TODO check handling of investor sale when no purchase is made
                #     self.model.logger.debug(f'Investor seller lists homes that do not sell. Investor {listing.seller.unique_id} keeps property {listing.property.unique_id}.')
                else:
                    print('Error. Seller is not an investor or a person')


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

            # # Record data for forecasting
            # new_row = {
            # 'land_id':        allocation.sale_property.unique_id,
            # 'realized_price': allocation.final_price,
            # 'time_step':      self.model.schedule.time,
            # 'transport_cost': allocation.sale_property.transport_cost,
            # 'wage':           self.model.firm.wage,
            # }

            # if isinstance(allocation.seller, Person):
            #     pass
            #     # self.handle_seller_departure(allocation)
            # elif isinstance(allocation.seller, Investor):
            #     self.model.logger.debug(f'In complete_transaction, before purchase, seller is Investor, id {allocation.seller.unique_id}.')
            # else:
            #     self.model.logger.debug(f'In complete_transaction, before purchase, seller {allocation.seller.unique_id} was not a person or investor. Seller {allocation.seller}.')

            # self.model.logger.debug(f'Time {self.model.schedule.time}, Property {allocation.property.unique_id}, Price {allocation.property.realized_price}')
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
        # self.model.logger.debug(f'Time {self.model.schedule.time} New worker {buyer.unique_id} Loc {sale_property}') # TEMP

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
        
        self.bid_type  = None
        self.bid_value = 0
        self.R_N       = 0
        self.density   = self.model.firm.density
         
    def step(self):
        self.model.property.change_dist(self.distance_from_center)
        self.R_N             = self.model.property.net_rent
        self.p_dot           = self.model.property.p_dot
        self.transport_cost  = self.model.property.transport_cost
        m                    = self.model.max_mortgage_share

        if self.bidder_name == 'Investor':
            self.model.logger.debug(f'Investor: dist {self.distance_from_center}, savings {self.bidder_savings}, R_N {self.R_N}') # transport_cost {self.transport_cost}')
            self.bid_value,  self.bid_type = self.model.investor.get_max_bid(m = m,
                            R_N            = self.R_N,
                            p_dot          = self.p_dot,
                            transport_cost = self.transport_cost)

        elif self.bidder_name.split()[0]  == 'Savings': # TODO CHECK THIS IS actually person type and catch errors
            self.model.logger.debug(f'Savings: dist {self.distance_from_center}, savings {self.bidder_savings}, R_N {self.R_N}') #  transport_cost {self.transport_cost}')
            M     = self.model.person.get_max_mortgage(self.bidder_savings)
            self.bid_value,  self.bid_type = self.model.person.get_max_bid(
                                         m = m, 
                                         M = M, 
                                         R_N = self.R_N, 
                                         p_dot = self.p_dot, 
                                         transport_cost = self.transport_cost, 
                                         savings = self.bidder_savings)

        else:
            self.model.logger.warning("Unexpected 'bidder_name': dist {self.distance_from_center}, savings {self.bidder_savings}")

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
