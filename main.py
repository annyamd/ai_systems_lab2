import re
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from queue import PriorityQueue
import csv


def print_line():
    print("--------------------------------")


def print_help():
    print_line()
    print("Command list:\n"
          "sg -- see graph;\n"
          "-----------------\n"
          "1 -- execute BFS;\n"
          "2 -- execute DFS;\n"
          "3 [limit] -- execute DLS;\n"
          "4 -- execute iterative-deepening depth-first search\n"
          "5 -- execute bidirectional search\n"
          "-----------------\n"
          "i1 -- execute informed BFS;\n"
          "i2 -- execute A*")
    print_line()


def print_err():
    print("Incorrect command. To see the list of commands, type \"h\".")


def plot_graph():
    nx.draw(g, with_labels=True)
    plt.show()


# search algorithms

def bfs(a, b):
    dist = {node: [] for node in nodes}
    dist[a] = [a]
    queue = deque([a])

    while len(queue) > 0:
        top = queue.popleft()
        if top == b:
            print(dist[b])
            break
        for i in g[top]:
            # print(i)
            if not dist[i]:
                queue.append(i)
                dist[i].extend(dist[top])
                dist[i].append(i)


def dfs(a, b):
    dls(a, b, -1)


def dls(a, b, limit):
    dist = {node: [] for node in nodes}
    dist[a] = [a]
    queue = deque([a])

    def f(depth):
        if depth >= limit >= 0:
            return False
        else:
            top = queue.pop()
            print("cur is " + top)

            if top == b:
                print(dist[b])
                return True

            for i in g[top]:
                if not dist[i]:
                    queue.append(i)
                    dist[i].extend(dist[top])
                    dist[i].append(i)
                    if f(depth + 1):
                        return True

    if not f(0):
        print("Unsuccessful search with limit = " + str(limit))
        return False
    return True


def iddfs(a, b):
    limit = 0

    while not dls(a, b, limit):
        limit += 1

    print("Limit = " + str(limit))


def bidir_search(a, b):
    dist1 = {node: [] for node in nodes}
    dist2 = {node: [] for node in nodes}
    dist1[a] = [a]
    dist2[b] = [b]

    queue1 = deque([a])
    queue2 = deque([b])

    while len(queue1) > 0 and len(queue2) > 0:
        print("1")
        bfs_bidir(queue1, dist1)
        print("2")
        bfs_bidir(queue2, dist2)

        intersection = is_intersecting(dist1, dist2)
        if intersection != -1:
            dist2[intersection].remove(intersection)
            dist2[intersection].reverse()
            path = dist1[intersection] + dist2[intersection]
            print("Found intersection: " + intersection)
            print(path)
            return


def bfs_bidir(queue, dist):
    top = queue.pop()
    print(top)

    for i in g[top]:
        if not dist[i]:
            queue.append(i)
            dist[i].extend(dist[top])
            dist[i].append(i)
    # print(dist)


def is_intersecting(dist1, dist2):
    for i in list(nodes):
        if dist1[i] and dist2[i]:
            return i

    return -1


def informed_bsf(a, b):
    pq = PriorityQueue()
    pq.put((0, a))
    visited = {node: False for node in nodes}
    visited[a] = True

    while not pq.empty():
        top = pq.get()[1]
        # Displaying the path having the lowest cost
        print(top, end=" ")
        if top == b:
            break

        for i in g[top]:
            if not visited[i]:
                visited[i] = True
                pq.put((costs[i], i))
    print()


def a_star(a, b):
    opened = {a}
    closed = set([])

    # poo has present distances from start to all other nodes
    # the default value is +infinity
    dist = {a: 0}  #g

    # par contains an adjac mapping of all nodes
    parents = {a: a}

    while len(opened) > 0:
        top = None

        print("Доступные переходы: ")
        for v in opened:
            print(v + " - " + str(dist[v] + costs[v]) + ", ", end="")
            if top is None or dist[v] + costs[v] < dist[top] + costs[top]:
                top = v
        print("\n" + "Переход к: " + top + "\n")
        if top is None:
            print('Path does not exist!')
            return None

        if top == b:
            path = []

            while parents[top] != top:
                path.append(top)
                top = parents[top]

            path.append(a)
            path.reverse()

            print(path)
            return path

        for i in g[top]:
            if i not in opened and i not in closed:
                opened.add(i)
                parents[i] = top
                dist[i] = dist[top] + g[i][top]['weight']
            else:
                if dist[i] > (dist[top] + g[i][top]['weight']):
                    dist[i] = dist[top] + g[i][top]['weight']
                    parents[i] = top

                    if i in closed:
                        closed.remove(i)
                        opened.add(i)

        opened.remove(top)
        closed.add(top)

    print('Path does not exist!')
    return None


def get_costs():
    with open('costs.csv', mode='r', encoding='utf8') as infile:
        reader = csv.reader(infile)
        return {rows[0]: int(rows[1]) for rows in reader}


def prompt():
    a = "Мурманск"
    b = "Одесса"

    while 1:
        try:
            inp = input(">")
        except EOFError:
            print("")
            break

        if inp == "sg":
            plot_graph()
        elif inp == "1":
            bfs(a, b)
        elif inp == "2":
            dfs(a, b)
        elif re.match(r"^3 \d+$", inp):
            limit = float(inp.split()[1])
            dls(a, b, limit)
        elif inp == "4":
            iddfs(a, b)
        elif inp == "5":
            bidir_search(a, b)
        elif inp == "i1":
            informed_bsf(a, b)
        elif inp == "i2":
            a_star(a, b)
        elif inp == "h":
            print_help()
        elif inp == "q":
            break
        else:
            print_err()


def main():
    print_help()
    prompt()


g = nx.read_weighted_edgelist("сии2.csv", delimiter=",")
nodes = g.nodes
edges = g.edges
costs = get_costs()
main()
