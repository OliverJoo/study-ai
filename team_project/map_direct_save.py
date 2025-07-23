# Searching Path and Visualization
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from collections import deque
import heapq
import itertools


# --- Utility to convert grid to graph representation ---
def grid_to_graph(grid):
    try:
        rows, cols = len(grid), len(grid[0])
        nodes = []
        edges = {}  # Using dict for adjacency list
        node_map = {}
        rev_node_map = []

        node_idx = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:  # If not an obstacle
                    nodes.append(node_idx)
                    node_map[(r, c)] = node_idx
                    rev_node_map.append((r, c))
                    edges[node_idx] = []
                    node_idx += 1

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    current_node_idx = node_map[(r, c)]
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                            neighbor_idx = node_map[(nr, nc)]
                            edges[current_node_idx].append((neighbor_idx, 1))  # weight is 1
        return nodes, edges, node_map, rev_node_map
    except (IndexError, KeyError) as e:
        print(f"Error creating graph from grid. Details: {e}")
        return None, None, None, None


# --- Pathfinding Algorithms ---
def bfs(grid, start, end):
    """Breadth-First Search (Single Source Shortest Path)."""
    try:
        rows, cols = len(grid), len(grid[0])
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (y, x), path = queue.popleft()
            if (y, x) == end:
                return path
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < rows
                    and 0 <= nx < cols
                    and grid[ny][nx] == 0
                    and (ny, nx) not in visited
                ):
                    visited.add((ny, nx))
                    queue.append(((ny, nx), path + [(ny, nx)]))
        return None
    except (IndexError, TypeError) as e:
        print(f"Error during BFS execution. Details: {e}")
        return None


def dijkstra(grid, start, end):
    """Dijkstra's Algorithm (Single Source Shortest Path)."""
    try:
        rows, cols = len(grid), len(grid[0])
        pq = [(0, start, [start])]
        visited = set()

        while pq:
            cost, (y, x), path = heapq.heappop(pq)
            if (y, x) in visited:
                continue
            visited.add((y, x))
            if (y, x) == end:
                return path

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and grid[ny][nx] == 0:
                    heapq.heappush(pq, (cost + 1, (ny, nx), path + [(ny, nx)]))
        return None
    except (IndexError, TypeError) as e:
        print(f"Error during Dijkstra execution. Details: {e}")
        return None


def a_star(grid, start, end):
    """A* Search Algorithm (Single Source Shortest Path)."""
    try:
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        rows, cols = len(grid), len(grid[0])
        pq = [(heuristic(start, end), 0, start, [start])]
        visited = set()

        while pq:
            _, g_cost, (y, x), path = heapq.heappop(pq)
            if (y, x) in visited:
                continue
            visited.add((y, x))
            if (y, x) == end:
                return path

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and grid[ny][nx] == 0:
                    new_g_cost = g_cost + 1
                    heapq.heappush(
                        pq,
                        (
                            new_g_cost + heuristic((ny, nx), end),
                            new_g_cost,
                            (ny, nx),
                            path + [(ny, nx)],
                        ),
                    )
        return None
    except (IndexError, TypeError) as e:
        print(f"Error during A* execution. Details: {e}")
        return None


def bellman_ford(grid, start, end):
    """Bellman-Ford Algorithm (Single Source Shortest Path)"""
    try:
        nodes, edges, node_map, rev_node_map = grid_to_graph(grid)
        if not all([nodes, edges, node_map, rev_node_map]):
            return None
        
        start_idx = node_map.get(start)
        end_idx = node_map.get(end)

        if start_idx is None or end_idx is None:
            print("Error: Start or end node not found in graph.")
            return None

        dist = {node: float("inf") for node in nodes}
        pred = {node: None for node in nodes}
        dist[start_idx] = 0

        for _ in range(len(nodes) - 1):
            for u in nodes:
                if u in edges:
                    for v, weight in edges[u]:
                        if dist[u] != float("inf") and dist[u] + weight < dist[v]:
                            dist[v] = dist[u] + weight
                            pred[v] = u

        # Path reconstruction
        path_idx = []
        curr = end_idx
        while curr is not None:
            path_idx.append(curr)
            curr = pred.get(curr) # Use .get for safety
        path_idx.reverse()

        return (
            [rev_node_map[i] for i in path_idx]
            if path_idx and path_idx[0] == start_idx
            else None
        )
    except Exception as e:
        print(f"An unexpected error occurred in Bellman-Ford. Details: {e}")
        return None


def floyd_warshall(grid):
    """Floyd-Warshall Algorithm (All-Pairs Shortest Path)."""
    try:
        nodes, _, node_map, rev_node_map = grid_to_graph(grid)
        if not all([nodes, node_map, rev_node_map]):
            return None, None

        num_nodes = len(nodes)
        dist = [[float("inf")] * num_nodes for _ in range(num_nodes)]
        next_node = [[None] * num_nodes for _ in range(num_nodes)]

        for i in range(num_nodes):
            dist[i][i] = 0
            next_node[i][i] = i

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 0:
                    u = node_map[(r, c)]
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < len(grid)
                            and 0 <= nc < len(grid[0])
                            and grid[nr][nc] == 0
                        ):
                            v = node_map[(nr, nc)]
                            dist[u][v] = 1
                            next_node[u][v] = v

        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]

        def get_path(start_pos, end_pos):
            u, v = node_map.get(start_pos), node_map.get(end_pos)
            if u is None or v is None or dist[u][v] == float("inf"):
                return None
            path_idx = [u]
            while u != v:
                u = next_node[u][v]
                path_idx.append(u)
            return [rev_node_map[i] for i in path_idx]

        def get_dist(start_pos, end_pos):
            u, v = node_map.get(start_pos), node_map.get(end_pos)
            if u is None or v is None:
                return float("inf")
            return dist[u][v]

        return get_path, get_dist
    except Exception as e:
        print(f"An unexpected error occurred in Floyd-Warshall. Details: {e}")
        return None, None


def johnson(grid):
    """Johnson's Algorithm (All-Pairs Shortest Path)."""
    try:
        nodes, _, node_map, rev_node_map = grid_to_graph(grid)
        if not all([nodes, node_map, rev_node_map]):
            return None, None
            
        all_paths = {}
        for i in range(len(rev_node_map)):
            start_node_pos = rev_node_map[i]
            for j in range(len(rev_node_map)):
                if i != j:
                    end_node_pos = rev_node_map[j]
                    path = dijkstra(grid, start_node_pos, end_node_pos)
                    all_paths[(start_node_pos, end_node_pos)] = path

        def get_path(start_pos, end_pos):
            return all_paths.get((start_pos, end_pos))

        def get_dist(start_pos, end_pos):
            path = all_paths.get((start_pos, end_pos))
            return len(path) - 1 if path else float("inf")

        return get_path, get_dist
    except Exception as e:
        print(f"An unexpected error occurred in Johnson's algorithm. Details: {e}")
        return None, None


# --- Main Drawing and Pathfinding Logic ---
def draw_map_with_path(df, path, output_filename):
    try:
        fig, ax = plt.subplots(figsize=(12, 12))
        max_x = df["x"].max() + 1
        max_y = df["y"].max() + 1
        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)
        ax.set_aspect("equal")
        ax.set_xticks(range(max_x + 1))
        ax.set_yticks(range(max_y + 1))
        ax.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
        
        markers = {
            "apartment": ("o", "brown"),
            "building": ("o", "brown"),
            "bandalgomcoffee": ("s", "green"),
            "myhome": ("^", "green"),
        }
        
        df_sorted = df.sort_values(by="ConstructionSite", ascending=True)
        
        for _, row in df_sorted.iterrows():
            x, y = row["x"], row["y"]
            if row["ConstructionSite"] == 1:
                ax.add_patch(
                    patches.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1, facecolor="grey", edgecolor="black"
                    )
                )
            else:
                struct_type = str(row.get("struct", "")).strip().replace("_", "").lower()
                if struct_type in markers:
                    marker, color = markers[struct_type]
                    ax.plot(x, y, marker=marker, color=color, markersize=10)
        
        if path:
            path_y, path_x = zip(*path)
            ax.plot(path_x, path_y, color="red", linewidth=2, marker="o", markersize=4)
            
        legend_elements = [
            patches.Patch(facecolor="grey", edgecolor="black", label="Construction Site"),
            plt.Line2D([0], [0], marker="o", color="w", label="Apartment/Building", markerfacecolor="brown", markersize=10),
            plt.Line2D([0], [0], marker="s", color="w", label="Bandal Gom Coffee", markerfacecolor="green", markersize=10),
            plt.Line2D([0], [0], marker="^", color="w", label="My House", markerfacecolor="green", markersize=10),
            plt.Line2D([0], [0], color="red", lw=2, label="Shortest Path"),
        ]
        
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.3, 1.02))
        plt.title("Map with Shortest Path")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close()
        print(f"Map saved to {output_filename}")

    except (KeyError, ValueError) as e:
        print(f"Error processing data for drawing. Details: {e}")
    except IOError as e:
        print(f"Error saving map image. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during map drawing. Details: {e}")


def main():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data_source")
        df = pd.read_csv(os.path.join(data_dir, "mas_map.csv"))

        max_x = df["x"].max() + 1
        max_y = df["y"].max() + 1
        grid = [[0] * max_x for _ in range(max_y)]
        
        for _, row in df[df["ConstructionSite"] == 1].iterrows():
            grid[row["y"]][row["x"]] = 1

        my_home_df = df[df["struct"].str.strip().replace("_", "").str.lower() == "myhome"]
        cafes_df = df[df["struct"].str.strip().replace("_", "").str.lower() == "bandalgomcoffee"]

        if my_home_df.empty:
            print("Error: 'My Home' not found in the data.")
            return

        start_pos = (my_home_df.iloc[0]["y"], my_home_df.iloc[0]["x"])

        shortest_path = None
        if not cafes_df.empty:
            for _, cafe in cafes_df.iterrows():
                end_pos = (cafe["y"], cafe["x"])
                
                # Select pathfinding algorithm
                # path = dijkstra(grid, start_pos, end_pos)
                path = a_star(grid, start_pos, end_pos)
                # path = bfs(grid, start_pos, end_pos)
                # path = bellman_ford(grid, start_pos, end_pos)
                # path = floyd_warshall(grid, start_pos, end_pos)
                # path = johnson(grid, start_pos, end_pos)

                if path and (shortest_path is None or len(path) < len(shortest_path)):
                    shortest_path = path

            if shortest_path:
                path_df = pd.DataFrame(shortest_path, columns=["y", "x"])
                path_output_path = os.path.join(base_dir, "home_to_cafe.csv")
                path_df.to_csv(path_output_path, index=False)
                print(f"Shortest path saved to {path_output_path}")
                
                map_output_filename = os.path.join(base_dir, "result_img", "map_final.png")
                draw_map_with_path(df, shortest_path, map_output_filename)
            else:
                print("No path found to any cafe.")
        else:
            print("No cafes found in the data.")

    except FileNotFoundError as e:
        print(f"Error loading data: File not found. Details: {e}")
    except (KeyError, IndexError) as e:
        print(f"Error processing data: Missing or invalid data column. Details: {e}")
    except IOError as e:
        print(f"Error saving path file. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main execution. Details: {e}")


if __name__ == "__main__":
    main()