import math
from bisect import bisect_left
from collections import defaultdict
from typing import List, Dict, Tuple
from data_models import Point, Facility, Customer
from mip import Model, xsum, BINARY, Var, CBC
from sklearn.neighbors import NearestNeighbors


def euclidean_length(point1: Point, point2: Point):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class LeaderModel:
    m: Model = None
    customer_facility: List[List[int]] = None
    facility_customer: List[List[int]] = None
    facilities: List[Facility] = None
    leaders: List[Facility] = None
    customers: List[Customer] = None
    var_store: Dict[Tuple[int, int], Var] = defaultdict()
    facilities_enabled: List[Var] = []
    NEIGHBOURS = 5
    RADIUS = 1
    CAPACITY_LEADERS = 2
    SETUP_LEADERS = 5

    def __init__(self, facilities: List[Facility], customers: List[Customer]):
        self.facilities = facilities
        self.customers = customers
        self.leaders = self.get_capacity_leaders()
        self.leaders += self.get_setup_leaders()
        self.dedupe_leaders()
        customer_distance_leaders: List[Dict[int, int]] = self.get_customer_distance_leaders()

        self.m = Model('LeaderModel', solver_name=CBC)
        self.initialize_customer_facility(customer_distance_leaders)
        self.initialize_facility_customer(customer_distance_leaders)

        self.model_adjustments()
        self.create_model_variables()
        self.add_necessary_constraints()
        self.construct_objective_fn()

    def model_adjustments(self):
        self.m.verbose = True
        self.m.max_seconds = 60 * 10

    def initialize_facility_customer(self, customer_distance_leaders):
        self.facility_customer = [[]] * len(self.facilities)
        for facility in self.facilities:
            facility_vars = []
            found = False
            for leader in self.leaders:
                if leader.index == facility.index:
                    for customer in self.customers:
                        facility_vars.append(customer.index)
                    self.facility_customer[facility.index] = facility_vars
                    found = True
            if found:
                continue
            for customer in self.customers:
                if facility.index in customer_distance_leaders[customer.index]:
                    facility_vars.append(customer.index)
            self.facility_customer[facility.index] = facility_vars

    def construct_objective_fn(self):
        setup_expression = []
        for facility in self.facilities:
            setup_expression.append(0.5 * facility.setup_cost * self.facilities_enabled[facility.index])

        distance_expressions = []
        for facility in self.facilities:
            for customer_selected in self.facility_customer[facility.index]:
                distance = euclidean_length(facility.location,
                                            self.customers[customer_selected].location)
                distance_expressions.append(distance * self.var_store[(customer_selected, facility.index)])

        self.m.objective = xsum(setup_expression) + xsum(distance_expressions)

    def add_necessary_constraints(self):
        for customer in self.customers:
            self.m.add_constr(xsum(map(lambda x: self.var_store[(customer.index, x)],
                                       self.customer_facility[customer.index])) == 1)
        for facility in self.facilities:
            expressions = []
            for customer_selected in self.facility_customer[facility.index]:
                expressions.append(self.customers[customer_selected].demand *
                                   self.var_store[(customer_selected, facility.index)])

                self.m.add_constr(self.var_store[(customer_selected, facility.index)] <=
                                  self.facilities_enabled[facility.index])
            self.m.add_constr(xsum(expressions) <= facility.capacity)

    def initialize_customer_facility(self, customer_distance_leaders):
        self.customer_facility = [0] * len(self.customers)
        for customer in self.customers:
            customer_vars = []
            for facility in self.leaders:
                customer_vars.append(facility.index)
            for facility in customer_distance_leaders[customer.index]:
                customer_vars.append(facility)
            self.customer_facility[customer.index] = customer_vars

    def optimize(self):
        self.m.optimize()

    def get_best_score(self):
        return self.m.objective_value

    def get_solution(self):
        solution = []
        for customer in self.customers:
            for facility_selected in self.customer_facility[customer.index]:
                variable = self.var_store[(customer.index, facility_selected)]
                if variable.x >= 0.99:
                    solution.append(facility_selected)
                    break
        return solution

    def get_capacity_leaders(self):
        return list(sorted(self.facilities, key=lambda facility: -facility.capacity))[:self.CAPACITY_LEADERS]

    def get_setup_leaders(self):
        return list(sorted(self.facilities, key=lambda facility: facility.setup_cost))[:self.SETUP_LEADERS]

    def get_customer_distance_leaders(self) -> List[Dict[int, int]]:
        facilities_already_selected = set()
        for facility in self.leaders:
            facilities_already_selected.add(facility.index)

        facilities: List[Facility] = list(filter(lambda x: x.index not in facilities_already_selected, self.facilities))
        facility_locations = map(lambda x: [x.location.x, x.location.y], facilities)
        customer_locations = map(lambda x: [x.location.x, x.location.y], self.customers)

        neigh = NearestNeighbors(n_neighbors=self.NEIGHBOURS, algorithm="ball_tree")
        neigh.fit(list(facility_locations))
        neighbour_indices = neigh.kneighbors(list(customer_locations), return_distance=False)

        sorted_facilities_x = list(sorted(map(lambda x: (x.location.x, x.index), facilities)))
        sorted_facilities_y = list(sorted(map(lambda x: (x.location.y, x.index), facilities)))

        customer_facility_map = [0] * len(self.customers)
        for neighbours, customer in zip(neighbour_indices, self.customers):
            selections = set()
            for neighbour in neighbours:
                selections.add(facilities[neighbour].index)

            customer_facility_map[customer.index] = selections
        return customer_facility_map

    def dedupe_leaders(self):
        seen = set()
        leader_temp = []
        for leader in self.leaders:
            if leader.index in seen:
                continue
            else:
                leader_temp.append(leader)
                seen.add(leader.index)
        self.leaders = leader_temp

    def create_model_variables(self):
        for customer in self.customers:
            facilities_selected = self.customer_facility[customer.index]
            for facility in facilities_selected:
                self.var_store[(customer.index, facility)] = self.m.add_var(var_type=BINARY)

        self.facilities_enabled = [0] * len(self.facilities)
        for facility in self.facilities:
            self.facilities_enabled[facility.index] = self.m.add_var(var_type=BINARY)
