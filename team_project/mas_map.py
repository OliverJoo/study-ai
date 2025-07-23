# Data Analysis
import pandas as pd
import os

# Set pandas display options to show all data
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def analysis_main():
    try:
        # Get the absolute path to the directory where the script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data_source")

        # List of CSV files to load
        file_names = ["area_category.csv", "area_map.csv", "area_struct.csv"]

        # Load all CSV files into a dictionary of DataFrames
        data_frames = {}
        for file in file_names:
            file_path = os.path.join(data_dir, file)
            data_frames[file.replace(".csv", "")] = pd.read_csv(file_path)

    except FileNotFoundError as e:
        print(f"Error loading data: File not found. Details: {e}")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing data: Could not parse CSV file. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return

    try:
        # Rename the column ' struct' to 'struct' in area_category dataframe
        data_frames["area_category"].rename(columns={" struct": "struct"}, inplace=True)

        # 1. Merge area_struct with area_map on ['x', 'y']
        merged_df = pd.merge(
            data_frames["area_struct"], data_frames["area_map"], on=["x", "y"]
        )

        # 2. Merge with area_category to get structure names
        merged_df = pd.merge(
            merged_df, data_frames["area_category"], on="category", how="left"
        )

        # 3. Sort the final DataFrame by 'area'
        merged_df.sort_values(by="area", inplace=True)
        print("\nStruct:")
        print(f"{merged_df['struct'].value_counts()}\n")

        # Save the filtered DataFrame to a CSV file
        output_path = os.path.join(data_dir, "mas_map.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"Data successfully saved to {output_path}")

        # 4. Filtering by 'area = 1'
        filtered_df = merged_df[merged_df["area"] == 1]

        # Generate and print the summary statistics report for the 'struct' column
        print("\n--- Summary Statistics Report for 'struct in area-1' column ---")
        struct_summary = filtered_df["struct"].describe()
        struct_counts = filtered_df["struct"].value_counts()

        print("\nValue Counts:")
        print(struct_counts)
        print("\nDescription:")
        print(struct_summary)
        print()
        print(filtered_df)
        print()

    except KeyError as e:
        print(f"Error processing data: A required column or key was not found. Details: {e}")
    except IOError as e:
        print(f"Error saving data: Could not write to file. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")


if __name__ == "__main__":
    analysis_main()
