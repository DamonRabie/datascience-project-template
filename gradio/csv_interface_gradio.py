import os
from io import StringIO

import gradio as gr
import pandas as pd


# Function to read CSV file and convert it to a DataFrame
def read_csv_file(temp_file):
    if isinstance(temp_file, str):
        df = pd.read_csv(StringIO(temp_file))
    else:
        df = pd.read_csv(temp_file.name)
    return df  # Return the DataFrame


# Function to convert DataFrame to a list
def df_to_list(temp_df):
    row_list = []
    col_names = ['a', 'b']

    for index, rows in temp_df.iterrows():
        # Create list for the current row
        my_list = [rows[col] for col in col_names]

        # Append the list to the final list
        row_list.append(my_list)

    return row_list


# Function to perform a find operation (TODO: Implement find values logic)
def find_func(fields):
    # TODO: Implement the logic to find values based on input fields
    return "test"


# Function to save items to a CSV file
def save_func(*items_list):
    to_be_saved = list(items_list)
    tmp_df = pd.DataFrame.from_records([to_be_saved],
                                       columns=["a", "b"])

    tmp_df.to_csv(path_or_buf=output_path, mode='a', index=False, header=not os.path.exists(output_path))


output_path = '../data/sample_output.csv'

# Read the initial CSV data
df = pd.read_csv('../data/sample_input.csv')

if os.path.exists(output_path):
    pre_results = pd.read_csv(output_path)
    pre_results_ids = pre_results["id"].unique()
    df = df[~df['id'].isin(pre_results_ids)]

# Get a list of unique IDs from the DataFrame
df_list = df.sort_values('date')['id'].to_list()

# Create a Gradio interface
with gr.Blocks() as demo:
    # Define a Dropdown for field_valid_1
    with gr.Row():
        field_valid_1_drop_down = gr.Dropdown(label='field_valid_1_drop_down', choices=['0', '1'], value='0')

    # Add a separator line
    gr.Markdown("______________________________________________________________________________________")

    # Define a Textbox for field_1
    with gr.Row():
        field_1_text_box = gr.Textbox(label='field_1_text_box', interactive=True)

    # Define a "Find" button
    find_button = gr.Button("Find")

    # Link the "Find" button to the find_func function
    find_button.click(find_func, inputs=field_1_text_box, outputs=[field_valid_1_drop_down])

    # Provide example data for the Textbox
    gr.Examples(
        df_list,
        field_1_text_box
    )

    # Define an "Upload CSV" button (if needed)
    # upload_button = gr.UploadButton(label="Upload CSV", file_types=['.csv'], live=True, file_count="single")
    # upload_button.upload(fn=process_csv_file, inputs=upload_button, outputs=, api_name="upload_csv")

# Launch the Gradio interface
demo.launch()
