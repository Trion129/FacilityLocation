import math
from typing import List
from data_models import Point, Facility, Customer
from mip import Model, xsum, minimize, BINARY, Var, CBC


def euclideam_length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class NaiveMIPModel:
    m: Model = None
    customer_facility_map: List[List[Var]] = None
    facility_customer_map: List[List[Var]] = None

    def __init__(self, facilities: List[Facility], customers: List[Customer]):
        self.m = Model('NaiveFacilityMIP', solver_name=CBC)
        self.customer_facility_map = [[self.m.add_var(var_type=BINARY)
                                       for facility in range(len(facilities))] for customer in range(len(customers))]
        self.facility_customer_map = list(zip(*self.customer_facility_map))

        for customer in range(len(customers)):
            self.m.add_constr(xsum(self.customer_facility_map[customer]) == 1)

        for facility in range(len(facilities)):
            self.m.add_constr(xsum([customers[index].demand * variable
                                    for index, variable in enumerate(self.facility_customer_map[facility])]) <=
                              facilities[facility].capacity)

        facility_enabled = [self.m.add_var(var_type=BINARY) for facility in range(len(facilities))]

        for facility in range(len(facilities)):
            for customer in range(len(customers)):
                self.m.add_constr(self.facility_customer_map[facility][customer] <= facility_enabled[facility])

        self.m.objective = xsum([facilities[f].setup_cost * facility_enabled[f] for f in range(len(facilities))]) + \
                           xsum([euclideam_length(facilities[facility].location, customers[customer].location) *
                                 self.facility_customer_map[facility][customer]
                                 for facility in range(len(facilities)) for customer in range(len(customers))])
        self.m.verbose = True
        self.m.max_seconds = 60 * 20

    def optimize(self):
        self.m.optimize()

    def get_best_score(self):
        return self.m.objective_value

    def get_solution(self):
        solution = []
        for customer in self.customer_facility_map:
            for index, facility in enumerate(customer):
                if facility.x >= 0.99:
                    solution.append(index)
                    break
        return solution
