# map_direct_save.py
# Shortest Path Navigation and Complete TSP Optimization System
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from collections import deque
import heapq
import itertools
import time
import math
from functools import lru_cache
import numpy as np
import random


class SimpleKMeans:
    """Custom K-means clustering implementation"""

    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            centroid = X[random.randint(0, n_samples - 1)]
            centroids[k] = centroid

        return centroids

    def _assign_clusters(self, X, centroids):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples)

        for i, point in enumerate(X):
            distances = []
            for centroid in centroids:
                distance = np.sqrt(np.sum((point - centroid) ** 2))
                distances.append(distance)

            labels[i] = np.argmin(distances)

        return labels

    def _update_centroids(self, X, labels):
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                centroids[k] = X[random.randint(0, len(X) - 1)]

        return centroids

    def fit_predict(self, X):
        centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iters):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)

            if np.allclose(centroids, new_centroids, rtol=1e-4):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels
        return labels.astype(int)


def bfs(grid, start, end):
    """BFS shortest path algorithm"""
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
    """Dijkstra shortest path algorithm"""
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
    """A* shortest path algorithm"""
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


class OptimizedPathfinder:
    """Optimized pathfinding system with caching"""

    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    @lru_cache(maxsize=2000)
    def get_distance(self, start, end):
        try:
            if start == end:
                return 0

            queue = deque([(start, 0)])
            visited = {start}

            while queue:
                (y, x), dist = queue.popleft()

                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    neighbor = (ny, nx)

                    if (
                        0 <= ny < self.rows
                        and 0 <= nx < self.cols
                        and self.grid[ny][nx] == 0
                        and neighbor not in visited
                    ):
                        if neighbor == end:
                            return dist + 1

                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

            return float("inf")
        except (IndexError, TypeError) as e:
            print(f"Error calculating distance. Details: {e}")
            return float("inf")

    @lru_cache(maxsize=1000)
    def get_path(self, start, end):
        try:
            if start == end:
                return tuple([start])

            queue = deque([(start, [start])])
            visited = {start}

            while queue:
                (y, x), path = queue.popleft()

                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    neighbor = (ny, nx)

                    if (
                        0 <= ny < self.rows
                        and 0 <= nx < self.cols
                        and self.grid[ny][nx] == 0
                        and neighbor not in visited
                    ):
                        new_path = path + [neighbor]

                        if neighbor == end:
                            return tuple(new_path)

                        visited.add(neighbor)
                        queue.append((neighbor, new_path))

            return None
        except (IndexError, TypeError) as e:
            print(f"Error finding path. Details: {e}")
            return None

    def get_cache_stats(self):
        dist_stats = self.get_distance.cache_info()
        path_stats = self.get_path.cache_info()
        return {"distance_cache": dist_stats, "path_cache": path_stats}


class ProgressMonitor:
    """Progress monitoring system"""

    def __init__(self, total_tasks, task_name="Task"):
        self.total_tasks = total_tasks
        self.current_task = 0
        self.start_time = time.time()
        self.task_name = task_name
        self.last_update_time = 0
        self.update_interval = 1.0

    def update(self, increment=1):
        self.current_task += increment
        current_time = time.time()

        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time

        progress_percent = (self.current_task / self.total_tasks) * 100
        elapsed_time = current_time - self.start_time

        if self.current_task > 0 and self.current_task < self.total_tasks:
            avg_time_per_task = elapsed_time / self.current_task
            remaining_tasks = self.total_tasks - self.current_task
            estimated_remaining_time = avg_time_per_task * remaining_tasks

            print(
                f"\r{self.task_name} Progress: {progress_percent:.1f}% "
                f"({self.current_task:,}/{self.total_tasks:,}) | "
                f"Elapsed: {elapsed_time:.1f}s | "
                f"ETA: {estimated_remaining_time:.1f}s",
                end="",
            )
        else:
            print(
                f"\r{self.task_name} Progress: {progress_percent:.1f}% "
                f"({self.current_task:,}/{self.total_tasks:,})",
                end="",
            )

    def finish(self):
        total_time = time.time() - self.start_time
        print(f"\n{self.task_name} Complete! Total time: {total_time:.2f}s")


class CompleteTSPSolver:
    """Complete TSP solver for visiting ALL structures (bonus requirement)"""

    def __init__(self, pathfinder):
        self.pathfinder = pathfinder

    def solve_complete_tsp(self, waypoints_dict, start_key, end_key):
        """
        Bonus: Complete TSP visiting ALL structures exactly once
        Requirement: Visit all apartments, buildings, cafes, then return home
        """
        try:
            print("Bonus TSP: Complete structure visitation optimization")

            if start_key not in waypoints_dict or end_key not in waypoints_dict:
                print(
                    f"Error: Start key ({start_key}) or end key ({end_key}) not found"
                )
                return [], [], float("inf")

            intermediate_keys = [
                key
                for key in waypoints_dict.keys()
                if key != start_key and key != end_key and key is not None
            ]

            print(f"TSP Configuration:")
            print(f"  Start: {start_key}")
            print(f"  End: {end_key}")
            print(f"  Intermediate waypoints: {len(intermediate_keys)}")
            print(f"  Total structures to visit: {len(intermediate_keys) + 2}")

            total_permutations = math.factorial(len(intermediate_keys))
            print(f"  Total permutations to evaluate: {total_permutations:,}")

            if total_permutations > 100000:
                print(
                    "  Warning: High computational complexity. Using optimized approach..."
                )
                return self._optimized_large_tsp(
                    waypoints_dict, start_key, end_key, intermediate_keys
                )

            return self._complete_small_tsp(
                waypoints_dict, start_key, end_key, intermediate_keys
            )
        except Exception as e:
            print(f"An unexpected error occurred in TSP solver. Details: {e}")
            return [], [], float("inf")

    def _complete_small_tsp(
        self, waypoints_dict, start_key, end_key, intermediate_keys
    ):
        """Complete TSP solution for small to medium problems"""
        best_cost = float("inf")
        best_sequence = None
        best_full_path = None

        total_permutations = math.factorial(len(intermediate_keys))
        progress_monitor = ProgressMonitor(total_permutations, "Complete TSP")

        print("Computing optimal TSP solution...")

        for permutation in itertools.permutations(intermediate_keys):
            current_sequence = [start_key] + list(permutation) + [end_key]
            current_cost = self._calculate_sequence_cost(
                current_sequence, waypoints_dict
            )

            if current_cost < best_cost:
                best_cost = current_cost
                best_sequence = current_sequence

                full_path = self._construct_full_path(best_sequence, waypoints_dict)
                if full_path:
                    best_full_path = full_path

                elapsed = time.time() - progress_monitor.start_time
                print(
                    f"\nNew best solution found! Cost: {best_cost}, Time: {elapsed:.1f}s"
                )

            progress_monitor.update()

        progress_monitor.finish()

        return best_sequence, best_full_path, best_cost

    def _optimized_large_tsp(
        self, waypoints_dict, start_key, end_key, intermediate_keys
    ):
        """Optimized approach for large TSP problems using clustering"""
        clusters = self._cluster_waypoints(waypoints_dict, intermediate_keys)
        print(f"Clustering result: {len(clusters)} clusters")

        cluster_order = self._optimize_cluster_order(
            clusters, waypoints_dict, start_key
        )

        final_sequence = [start_key]
        total_cost = 0
        current_pos = waypoints_dict[start_key]

        cluster_monitor = ProgressMonitor(len(cluster_order), "Cluster Processing")

        for cluster_idx in cluster_order:
            cluster_keys = clusters[cluster_idx]

            if cluster_keys:
                entry_key = min(
                    cluster_keys,
                    key=lambda k: self.pathfinder.get_distance(
                        current_pos, waypoints_dict[k]
                    ),
                )

                cluster_sequence, cluster_cost = self._solve_cluster_tsp(
                    cluster_keys, entry_key, waypoints_dict
                )

                if cluster_sequence:
                    final_sequence.extend(cluster_sequence)
                    total_cost += cluster_cost
                    current_pos = waypoints_dict[cluster_sequence[-1]]

            cluster_monitor.update()

        cluster_monitor.finish()

        final_cost = self.pathfinder.get_distance(current_pos, waypoints_dict[end_key])
        final_sequence.append(end_key)
        total_cost += final_cost

        full_path = self._construct_full_path(final_sequence, waypoints_dict)

        return final_sequence, full_path, total_cost

    def _cluster_waypoints(self, waypoints_dict, intermediate_keys, max_cluster_size=8):
        """Cluster waypoints geographically"""
        if len(intermediate_keys) <= max_cluster_size:
            return [intermediate_keys]

        coordinates = []
        for key in intermediate_keys:
            pos = waypoints_dict[key]
            coordinates.append([pos[0], pos[1]])

        coordinates = np.array(coordinates)

        n_clusters = max(2, len(intermediate_keys) // max_cluster_size)
        kmeans = SimpleKMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)

        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(intermediate_keys[i])

        return list(clusters.values())

    def _optimize_cluster_order(self, clusters, waypoints_dict, start_key):
        """Optimize cluster visiting order"""
        start_pos = waypoints_dict[start_key]

        def cluster_distance(cluster_idx):
            cluster = clusters[cluster_idx]
            if not cluster:
                return float("inf")

            min_dist = float("inf")
            for key in cluster:
                if key in waypoints_dict:
                    dist = self.pathfinder.get_distance(start_pos, waypoints_dict[key])
                    min_dist = min(min_dist, dist)
            return min_dist

        cluster_indices = list(range(len(clusters)))
        sorted_indices = sorted(cluster_indices, key=cluster_distance)
        return sorted_indices

    def _solve_cluster_tsp(self, cluster_keys, entry_key, waypoints_dict):
        """Solve TSP within a single cluster"""
        if len(cluster_keys) <= 1:
            return cluster_keys, 0

        if entry_key not in cluster_keys:
            entry_key = cluster_keys[0]

        other_keys = [k for k in cluster_keys if k != entry_key]

        if not other_keys:
            return [entry_key], 0

        best_cost = float("inf")
        best_path = None

        for permutation in itertools.permutations(other_keys):
            current_path = [entry_key] + list(permutation)
            current_cost = self._calculate_sequence_cost(current_path, waypoints_dict)

            if current_cost < best_cost:
                best_cost = current_cost
                best_path = current_path

        return best_path if best_path else cluster_keys, best_cost

    def _calculate_sequence_cost(self, sequence, waypoints_dict):
        """Calculate total cost for a waypoint sequence"""
        if len(sequence) < 2:
            return 0

        total_cost = 0
        for i in range(len(sequence) - 1):
            current_key = sequence[i]
            next_key = sequence[i + 1]

            if current_key is None or next_key is None:
                return float("inf")

            if current_key not in waypoints_dict or next_key not in waypoints_dict:
                return float("inf")

            start_pos = waypoints_dict[current_key]
            end_pos = waypoints_dict[next_key]
            cost = self.pathfinder.get_distance(start_pos, end_pos)

            if cost == float("inf"):
                return float("inf")
            total_cost += cost

        return total_cost

    def _construct_full_path(self, sequence, waypoints_dict):
        """Construct complete coordinate path from waypoint sequence"""
        if len(sequence) < 2:
            return []

        full_path = []

        for i in range(len(sequence) - 1):
            start_key = sequence[i]
            end_key = sequence[i + 1]

            if start_key in waypoints_dict and end_key in waypoints_dict:
                segment_path = self.pathfinder.get_path(
                    waypoints_dict[start_key], waypoints_dict[end_key]
                )
                if segment_path:
                    segment_list = list(segment_path)
                    if i == 0:
                        full_path.extend(segment_list)
                    else:
                        full_path.extend(segment_list[1:])

        return full_path


def draw_map_with_path(df, path, output_filename, title="Shortest Path Map"):
    """Visualize map with red path line"""
    try:
        fig, ax = plt.subplots(figsize=(15, 12))
        max_x = df["x"].max() + 1
        max_y = df["y"].max() + 1
        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)
        ax.set_aspect("equal")
        ax.set_xticks(range(max_x + 1))
        ax.set_yticks(range(max_y + 1))
        ax.grid(True, which="both", color="lightgray", linestyle="-", linewidth=0.5)

        markers = {
            "apartment": ("o", "brown", 12),
            "building": ("s", "darkblue", 12),
            "bandalgomcoffee": ("^", "green", 15),
            "myhome": ("*", "red", 18),
        }

        df_sorted = df.sort_values(by="ConstructionSite", ascending=True)
        for _, row in df_sorted.iterrows():
            x, y = row["x"], row["y"]
            if row["ConstructionSite"] == 1:
                ax.add_patch(
                    patches.Rectangle(
                        (x - 0.4, y - 0.4),
                        0.8,
                        0.8,
                        facecolor="gray",
                        edgecolor="black",
                        linewidth=1,
                    )
                )
            else:
                struct_type = str(row["struct"]).strip().replace("_", "").lower()
                if struct_type in markers:
                    marker, color, size = markers[struct_type]
                    ax.plot(
                        x,
                        y,
                        marker=marker,
                        color=color,
                        markersize=size,
                        markeredgecolor="black",
                        markeredgewidth=1,
                    )

        if path and len(path) > 1:
            path_y, path_x = zip(*path)
            ax.plot(
                path_x,
                path_y,
                color="red",
                linewidth=3,
                alpha=0.8,
                marker="o",
                markersize=6,
                markerfacecolor="red",
                label="Shortest Path",
            )

            ax.plot(
                path_x[0],
                path_y[0],
                marker="o",
                color="lime",
                markersize=15,
                markeredgecolor="darkgreen",
                markeredgewidth=2,
            )
            ax.plot(
                path_x[-1],
                path_y[-1],
                marker="o",
                color="gold",
                markersize=15,
                markeredgecolor="darkorange",
                markeredgewidth=2,
            )

        legend_elements = [
            patches.Patch(
                facecolor="gray", edgecolor="black", label="Construction Site (Blocked)"
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Apartment",
                markerfacecolor="brown",
                markersize=12,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Building",
                markerfacecolor="darkblue",
                markersize=12,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label="Bandal Gom Coffee",
                markerfacecolor="green",
                markersize=15,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                label="My Home",
                markerfacecolor="red",
                markersize=18,
            ),
            plt.Line2D([0], [0], color="red", lw=3, label="Shortest Path"),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(1.25, 1.02)
        )

        plt.title(
            f"{title}\nTotal Distance: {len(path) - 1 if path else 0} units",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Map saved successfully: {output_filename}")

    except (KeyError, ValueError) as e:
        print(f"Error processing data for drawing. Details: {e}")
    except IOError as e:
        print(f"Error saving map image. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during map drawing. Details: {e}")


def main():
    """Main execution function with CSV saving functionality"""
    try:
        print("=" * 70)
        print("Shortest Path Navigation and Complete TSP Optimization System")
        print("=" * 70)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data_source")
        csv_path = os.path.join(data_dir, "mas_map.csv")

        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} file not found.")
            return

        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully: {len(df)} location records")

        max_x = df["x"].max() + 1
        max_y = df["y"].max() + 1
        grid = [[0] * max_x for _ in range(max_y)]

        construction_sites = df[df["ConstructionSite"] == 1]
        for _, row in construction_sites.iterrows():
            grid[row["y"]][row["x"]] = 1
        print(
            f"Construction sites set as obstacles: {len(construction_sites)} locations"
        )

        my_home = df[
            df["struct"].str.strip().str.replace("_", "").str.lower() == "myhome"
        ]
        cafes = df[
            df["struct"].str.strip().str.replace("_", "").str.lower()
            == "bandalgomcoffee"
        ]

        if my_home.empty or cafes.empty:
            print("Error: 'MyHome' or 'BandalgomCoffee' not found.")
            return

        my_home_pos = (my_home.iloc[0]["y"], my_home.iloc[0]["x"])
        print(f"My home position: {my_home_pos}")
        print(f"Number of cafes: {len(cafes)}")

        result_img_dir = os.path.join(base_dir, "result_img")
        os.makedirs(result_img_dir, exist_ok=True)

        print("\n=== Main Task: Home to Nearest Cafe Shortest Path ===")

        shortest_path = None
        nearest_cafe_pos = None
        min_distance = float("inf")

        for _, cafe in cafes.iterrows():
            cafe_pos = (cafe["y"], cafe["x"])
            path = a_star(grid, my_home_pos, cafe_pos)

            if path and len(path) < min_distance:
                shortest_path = path
                nearest_cafe_pos = cafe_pos
                min_distance = len(path)

        if shortest_path:
            print(f"Shortest path found: {my_home_pos} -> {nearest_cafe_pos}")
            print(f"Path length: {len(shortest_path) - 1} units")

            path_df = pd.DataFrame(shortest_path, columns=["y", "x"])
            home_to_cafe_path = os.path.join(base_dir, "home_to_cafe.csv")
            path_df.to_csv(home_to_cafe_path, index=False)
            print(f"Path saved to CSV: {home_to_cafe_path}")

            map_final_path = os.path.join(result_img_dir, "map_final.png")
            draw_map_with_path(
                df, shortest_path, map_final_path, "Home to Nearest Cafe Shortest Path"
            )
        else:
            print("No path found to any cafe.")

        print("\n=== Bonus Task: Complete Structure Visitation TSP ===")

        pathfinder = OptimizedPathfinder(grid)

        all_structures = df[df["ConstructionSite"] == 0].copy()
        global waypoints
        waypoints = {}

        structure_counts = {
            "apartment": 0,
            "building": 0,
            "bandalgomcoffee": 0,
            "myhome": 0,
        }

        for _, row in all_structures.iterrows():
            struct_name = str(row["struct"]).strip().replace("_", "").lower()
            if struct_name and struct_name != "nan" and len(struct_name) > 0:
                waypoint_key = f"{struct_name}_{row.name}"
                waypoints[waypoint_key] = (row["y"], row["x"])

                if struct_name in structure_counts:
                    structure_counts[struct_name] += 1

        print(f"Structure analysis:")
        for struct_type, count in structure_counts.items():
            print(f"  - {struct_type.title()}: {count}")
        print(f"  - Total structures: {sum(structure_counts.values())}")

        bandalgom_start_key = None
        my_home_end_key = None

        for key in waypoints:
            if "bandalgomcoffee" in key:
                bandalgom_start_key = key
                break
        for key in waypoints:
            if "myhome" in key:
                my_home_end_key = key
                break

        if not bandalgom_start_key or not my_home_end_key:
            print("Error: Cannot find TSP start or end points.")
        else:
            print(f"TSP start point: {bandalgom_start_key}")
            print(f"TSP end point: {my_home_end_key}")

            complete_solver = CompleteTSPSolver(pathfinder)

            tsp_start_time = time.time()
            optimal_sequence, optimal_full_path, optimal_cost = (
                complete_solver.solve_complete_tsp(
                    waypoints, bandalgom_start_key, my_home_end_key
                )
            )
            tsp_total_time = time.time() - tsp_start_time

            print("\n" + "=" * 60)
            print("Complete TSP Optimization Results")
            print("=" * 60)
            print(f"Computation time: {tsp_total_time:.2f} seconds")
            print(f"Optimal total distance: {optimal_cost}")

            if optimal_sequence and optimal_cost != float("inf"):
                readable_path = " -> ".join(
                    [key.split("_")[0].title() for key in optimal_sequence]
                )
                print(f"Visit sequence: {readable_path}")
                print(f"Total waypoints: {len(optimal_sequence)}")

                visited_structures = set()
                for key in optimal_sequence:
                    struct_type = key.split("_")[0]
                    visited_structures.add(struct_type)

                print(f"Structure types visited: {sorted(visited_structures)}")
                print(
                    f"Complete visitation: {'YES' if len(visited_structures) >= 4 else 'NO'}"
                )

                if optimal_full_path:
                    map_final_bonus_path = os.path.join(
                        result_img_dir, "map_final_bonus.png"
                    )
                    draw_map_with_path(
                        df,
                        optimal_full_path,
                        map_final_bonus_path,
                        f"Bonus: Complete Structure Visitation TSP (Distance: {optimal_cost})",
                    )
            else:
                print("No optimal path found for complete structure visitation.")

        if "pathfinder" in locals():
            cache_stats = pathfinder.get_cache_stats()
            dist_cache = cache_stats["distance_cache"]
            path_cache = cache_stats["path_cache"]

            print(f"\nPerformance Statistics:")
            total_dist_calls = dist_cache.hits + dist_cache.misses
            total_path_calls = path_cache.hits + path_cache.misses

            if total_dist_calls > 0:
                print(
                    f"  Distance calculation cache efficiency: {dist_cache.hits / total_dist_calls * 100:.1f}%"
                )
            if total_path_calls > 0:
                print(
                    f"  Path calculation cache efficiency: {path_cache.hits / total_path_calls * 100:.1f}%"
                )
            print(f"  Total computation calls: {total_dist_calls + total_path_calls:,}")

        print("\n" + "=" * 70)
        print("All Tasks Completed Successfully!")
        print(f"Output files:")
        print(f"  - Main task CSV: home_to_cafe.csv")
        print(f"  - Main task PNG: {result_img_dir}/map_final.png")
        print(f"  - Bonus task PNG: {result_img_dir}/map_final_bonus.png")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"Error loading data: File not found. Details: {e}")
    except (KeyError, IndexError) as e:
        print(f"Error processing data: Missing or invalid data column. Details: {e}")
    except IOError as e:
        print(f"Error saving file. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main execution. Details: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
