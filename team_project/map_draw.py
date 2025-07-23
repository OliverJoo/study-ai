# Map Visualization
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def draw_main():
    try:
        # Get the absolute path to the directory where the script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data_source")

        # 1. Load the analyzed data
        df = pd.read_csv(os.path.join(data_dir, "mas_map.csv"))

    except FileNotFoundError as e:
        print(f"File not found. Details: {e}")
        return
    except pd.errors.ParserError as e:
        print(f"Could not parse CSV file. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return

    try:
        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(12, 12))

        # 2. Set the coordinate system (top-left origin)
        max_x = df["x"].max() + 1
        max_y = df["y"].max() + 1
        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)
        ax.set_aspect("equal")

        # 3. Draw grid lines
        ax.set_xticks(range(max_x + 1))
        ax.set_yticks(range(max_y + 1))
        ax.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)

        # 5. Prioritize construction sites (draw them last to appear on top)
        df = df.sort_values(by="ConstructionSite", ascending=True)

        # Define markers and colors for other structures
        markers = {
            "apartment": ("o", "brown"),
            "building": ("o", "brown"),
            "bandalgomcoffee": ("s", "green"),
            "myhome": ("^", "green"),
        }

        # 4. & 5. Plot each point based on its type
        for _, row in df.iterrows():
            x, y = row["x"], row["y"]

            if row["ConstructionSite"] == 1:
                ax.add_patch(
                    patches.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1, facecolor="grey", edgecolor="black"
                    )
                )
            else:
                if pd.isna(row["struct"]) or row["struct"] == "":
                    continue

                struct_type = row["struct"].strip().replace("_", "").lower()

                if struct_type in markers:
                    marker, color = markers[struct_type]
                    ax.plot(x, y, marker=marker, color=color, markersize=10)

        # 8. (Bonus) Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Apartment/Building",
                markerfacecolor="brown",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Bandal Gom Coffee",
                markerfacecolor="green",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label="My House",
                markerfacecolor="green",
                markersize=10,
            ),
            patches.Patch(facecolor="grey", edgecolor="black", label="Construction Site"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.3, 1.02))

        plt.title("Area Map")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # 6. Save the map as a PNG file
        output_path = os.path.join(base_dir, "result_img", "map.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Map saved to {output_path}")

    except KeyError as e:
        print(f"A required column or key was not found. Details: {e}")
    except ValueError as e:
        print(f"A value is invalid or out of range. Details: {e}")
    except IOError as e:
        print(f"Could not write to file. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during map drawing: {e}")


if __name__ == "__main__":
    draw_main()
