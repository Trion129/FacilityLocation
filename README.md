# Facility Location problem solver
This is a MIP solver for Facility location problem made as part of Discrete Optimization course.

There are 2 solvers:
1. Naive Formulation: pretty much a bruteforce implementation.
2. Leader formulation: I trim the edges by limiting neighbours to nearest neighbours.
Then I take some cheapest and biggest facilities and allow those as neighbours too.
This is to avoid infeasibility when neighbours do not suffice.

My patience was that my solver should be able to solve instances within 15 mins.
So it is similarly designed to be a fast solution.

I used naive solver for small instances and leader one for larger ones.