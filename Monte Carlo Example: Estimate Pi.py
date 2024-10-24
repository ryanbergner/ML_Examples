import random

# Define the number of iterations
num_iterations = 1000000

# Initialize the count of points inside the circle
count_inside_circle = 0

# Run the Monte Carlo simulation
for _ in range(num_iterations):
    # Generate random x and y coordinates
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    # Calculate the distance from the origin
    distance = x**2 + y**2

    # Check if the point is inside the circle
    if distance <= 1:
        count_inside_circle += 1

# Calculate the ratio of points inside the circle to the total number of points
ratio = count_inside_circle / num_iterations

# Estimate the value of pi
estimated_pi = ratio * 4

print(f"Estimated value of pi: {estimated_pi}")