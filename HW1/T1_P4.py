#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import math

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
# plt.figure(1)
# plt.plot(years, republican_counts, 'o')
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.figure(2)
# plt.plot(years, sunspot_counts, 'o')
# plt.xlabel("Year")
# plt.ylabel("Number of Sunspots")
# plt.figure(3)
# plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
# plt.xlabel("Number of Sunspots")
# plt.ylabel("Number of Republicans in Congress")
# plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part,is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
    
    # TO DO
    output = []
    if part == 'a':
        j = 5
        for x in xx:
            basis = [1]
            for i in range(1, j+1):
                basis.append(x**i)
            output.append(basis)
        return np.array(output)

    y = np.arange(1960, 2010, 5)
    if part == 'b':
        for x in xx:
            basis = [1]
            for u in y:
                basis.append(math.pow(math.e, -(x-u)**2/25))
            output.append(basis)
        return np.array(output)

    if part == 'c':
        j = 5
        for x in xx:
            basis = [1]
            for i in range(1, j+1):
                basis.append(math.cos(x/i))
            output.append(basis)
        return np.array(output)

    if part == 'd':
        j = 25
        for x in xx:
            basis = [1]
            for i in range(1, j+1):
                basis.append(math.cos(x/i))
            output.append(basis)
        return np.array(output)

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
# grid_X = np.vstack((np.ones(grid_years.shape), grid_years))

# grid_Yhat  = np.dot(grid_X.T, w)

grid_sunspots = np.linspace(10.2, 154.6, 200)

# TODO: plot and report sum of squared error for each basis

# 4.1
# for part in ['a', 'b', 'c', 'd']:
#     basis = make_basis(years, part)
#     W = find_weights(basis, Y)
#     grid = make_basis(grid_years, part)
#     basis_yhat = np.dot(basis, W)
#     grid_yhat = np.dot(grid, W)

#     loss = 0
#     for index, y in enumerate(Y):
#         loss += (basis_yhat[index] - Y[index])**2

#     plt.plot(years, republican_counts, 'o', grid_years, grid_yhat, '-')
#     plt.xlabel("Year")
#     plt.ylabel("Number of Republicans in Congress")
#     plt.title('Part ' + part + ' regression. Loss = ' + str(loss))
#     plt.savefig('Part' + part + 'regression.png')
#     plt.show()

# 4.2 
for part in ['a', 'c', 'd']:
    basis = make_basis(sunspot_counts[years<last_year], part, False)
    W = find_weights(basis, Y[years<last_year])
    grid = make_basis(grid_sunspots, part, False)
    basis_yhat = np.dot(basis, W)
    grid_yhat = np.dot(grid, W)

    loss = 0
    for index, y in enumerate(basis_yhat):
        loss += (y - Y[index])**2


    plt.plot(sunspot_counts, republican_counts, 'o', grid_sunspots, grid_yhat, "-")
    plt.xlabel("Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.title('Part ' + part + ' Regression Sunspots. Loss = ' + str(loss))
    plt.savefig('Part' + part + 'sunspot.png')
    plt.show()


# make_basis(X, 'a', is_years=True)
# make_basis(X, 'b', is_years=True)
# make_basis(X, 'c', is_years=True)
# make_basis(X, 'd', is_years=True)

# # Plot the data and the regression line.
# plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.show()