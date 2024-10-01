import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.animation import FuncAnimation
from math import sqrt
import csv
import random

# [ GLOBAL OBJECTS ]

colors = ['red','blue','yellow','green']
maxDistance = 0

# [ NODE GENERATION ]

def GenerateRandomPoints(lambdaValue , limitValue) :
    value = int(np.random.poisson(lambdaValue,1)[0])
    while not (0 <= value <= limitValue) :
        value = int(np.random.poisson(lambdaValue,1)[0])
    return value

def GeneratePoints(n,dimention,minDistance) :
    print(n,dimention)

# Initializing RB's

    points = [[-500,-500,'black'],[-500,-500,'black'],[-500,-500,'black']]
    while(len(points) < n) :
        while 1 :
            lambdaValue1 = random.randint(0,dimention[0])
            lambdaValue2 = random.randint(0,dimention[1])
            x = GenerateRandomPoints(lambdaValue1,dimention[0])
            y = GenerateRandomPoints(lambdaValue2,dimention[1])
            z = colors[random.randint(0,3)]
            if all(np.sqrt((x-p[0])**2 + (y-p[1])**2) >= minDistance for p in points) : 
                points.append((x,y,z))
                break

    return points

# [ POSITION AND DIRECTION ]

def AssignDirection(points):
    directions = []
    for _ in range(len(points)):
        # Generate random angle in radians
        angle = np.random.uniform(0, 2*np.pi)
        # Convert angle to directional vector
        direction = (np.cos(angle), np.sin(angle))
        directions.append(direction)
    return directions

def save_positions(points):
    with open('positions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X-coordinate', 'Y-coordinate'])
        for point in points:
            writer.writerow([point[0], point[1]])

def save_cluster(clusters):
    with open('clusters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Cluster Number', 'Points'])
        for ind in range(len(clusters)):
            writer.writerow([ind,clusters[ind]])

def save_rbs(rbs):
    with open('rbs.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Coordinates','cluster'])
        for item in rbs:
            coords = item
            writer.writerow([coords])

# [ UPDATE FRAMES ] 

def update(frame):
    global points, scatter, arrows
    
    # Update positions of points
    for i in range(len(points)):
        x, y, color = points[i]
        x += random.randint(-10, 10)  # Update x-coordinate randomly
        y += random.randint(-10, 10)  # Update y-coordinate randomly
        points[i] = (x, y, color)
    
    # Change color of randomly selected points
    num_points_to_change_color = random.randint(1, len(points))
    points_to_change_color_indices = random.sample(range(len(points)), num_points_to_change_color)
    for i in points_to_change_color_indices:
        if points[i][2] == 'white' or points[i][2] == 'black' :
            continue
        points[i] = (points[i][0], points[i][1], random.choice(['red', 'green', 'blue', 'yellow']))
    
    # Update scatter plot with new positions and colors
    scatter.set_offsets([(x, y) for x, y, _ in points])
    scatter.set_color([color for _, _, color in points])

    # Update point sizes
    sizes = [100 if i < 3 else 50 for i in range(len(points))]  # Larger size for first three points
    scatter.set_sizes(sizes)
    
    # Update arrows for direction vectors
    for i, (x, y, _) in enumerate(points):
        if(i<3) :
            continue
        dx = x - old_points[i][0]  # Change in x-coordinate
        dy = y - old_points[i][1]  # Change in y-coordinate
        arrows[i].remove()  # Remove old arrow
        arrows[i] = ax.arrow(old_points[i][0], old_points[i][1], dx, dy, head_width=2, head_length=2, color='black')  # Add new arrow
    
    old_points[:] = [(x, y) for x, y, _ in points]  # Update old points
    custom_cluster(points, maxDistance) 
    
    return scatter, arrows

    
# [ CREATING CLUSTERS ] 

def find_new_rb_coord(cluster, prev_rbs, max_distance,clust_no):
    x_sum = sum(coord[0] for coord in cluster)
    y_sum = sum(coord[1] for coord in cluster)
    centroid_x = x_sum / len(cluster)
    centroid_y = y_sum / len(cluster)

    for rb_coord in prev_rbs:
        dist = sqrt((centroid_x - rb_coord[0]) ** 2 + (centroid_y - rb_coord[1]) ** 2)
        if dist <= max_distance:
            return None  # Not suitable, retry

    return [centroid_x, centroid_y, 'black',clust_no]

def custom_cluster(points, max_distance):
    clusters = []
    vis = [0] * len(points)
    rb_cord = []
    
    for i, (x1, y1, color1) in enumerate(points):
        # Initialize a new cluster with the current point
        if color1 == 'black' or color1 == 'white' or vis[i]:
            if color1 == 'black' or color1 == 'white':
                rb_cord.append([x1, y1])
            continue

        vis[i] = 1

        current_cluster = [(x1, y1, color1)]

        for j, (x2, y2, color2) in enumerate(points):
            if vis[j] == 0 and i != j and color1 == color2:
                distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance <= max_distance:
                    vis[j] = 1
                    current_cluster.append((x2, y2, color2))

        if current_cluster:
            clusters.append(current_cluster)

    # Sort clusters based on their size
    clusters.sort(key=len, reverse=True)

    # Assign RB's

    dis = 2 * max_distance
    rbs = []

    for i, cluster in enumerate(clusters):
        if len(rbs) >= 3:
            break
        new_rb_coord = find_new_rb_coord(cluster, rbs, dis,i)
        if new_rb_coord:
            rbs.append(new_rb_coord)

    # If not all black points are assigned, set remaining black points at (-500, -500)
    for _ in range(len(rbs), min(3, len(clusters))):
        rbs.append([-500, -500, "white"])

    save_cluster(clusters)
    save_rbs(rbs)


# [ MAIN FUNCTION ]

if __name__ == "__main__":

    n = int(input("[ TOTAL NODES ] : "))
    dimention = list(map(int, input("[ DIMENTIONS ] : ").split()))
    minDistance = int(input("[ MINIMUM DISTANCE ] : "))
    maxDistance = int(input("[ MAXIMUM CLUSTER DISTANCE ] "))

    points = GeneratePoints(n,dimention,minDistance)
    old_points = [(x, y) for x, y, _ in points] 
    directions = AssignDirection(points)

    # Graph values

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_title('Points with Coordinates and Colors')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')

    # Plot points
    scatter = ax.scatter([x for x, _, _ in points], [y for _, y, _ in points], c=[color for _, _, color in points], s=50)

    # Initialize arrows for direction vectors
    arrows = [ax.arrow(x, y, 0, 0, head_width=2, head_length=2, color='black') for x, y, _ in points]

    ani = FuncAnimation(fig, update, frames=100, interval=1000, repeat=False)
    plt.show()