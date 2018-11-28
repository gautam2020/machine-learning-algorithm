from numpy import *
# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def stepgradient(points,b,theta,learning_rate):
    b_gradient = 0
    theta_gradient = 0
    M = len(points)
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (2/M)*((b+x*theta)-y)
        theta_gradient += (2/M)*x*((b+x*theta)-y)
    new_b = b - (learning_rate*b_gradient)
    new_theta = theta - (learning_rate*theta_gradient)
    return new_b, new_theta

def start_gradient_descent(points, initial_b,initial_theta,iterations,learning_rate):
    b = initial_b
    theta = initial_theta
    for  i in range(iterations):
        b,theta = stepgradient(points,b,theta,learning_rate)
        print(b, theta)
    return b,theta

points = genfromtxt("C:\\Users\\sikka\\Desktop\\wine_data.csv",delimiter=",")
learning_rate = 0.0001
initial_b = 0
initial_theta = 0
number_of_iterations = 100
b,theta = start_gradient_descent(points, initial_b,initial_theta,number_of_iterations,learning_rate)
