import os
import gradio as gr
import random
import time

import pandas as pd
from openai import OpenAI

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))


def get_radii(atom_id, radius_type="vdw"):
    
    
    xl = pd.ExcelFile("Elemental_Radii.xlsx")
    df = xl.parse(sheet_name=0, header = 2, index_col=1)
    
    if radius_type == "cov":
        key = 6
    elif radius_type == "vdw":
        key = 7
    else:
        raise ValueError("radius_type not supported")
    if atom_id in df.index:
        return df.loc[atom_id][key]
    else:
        raise ValueError("Elemental symbol not found")


def get_pore_data(material_id: str):
    """
    material_id: IZA framework code
    """
    pore_data_df = pd.read_csv("iza_bulk.csv", header = 0, sep = ",", on_bad_lines='skip', index_col="cif")

    if material_id in pore_data_df.index:
        row = pore_data_df.loc[material_id]
        return row.to_dict()
    else:
        raise ValueError("Material ID not found")

SYSTEM_INSTRUCTIONS = """\You are an AI assistant, designed to help researchers discover novel materials by querying material databases. You can provide information about the properties of elements, compounds, and materials, as well as answer general scientific questions. You can also execute Python code to perform calculations and simulations. To get started, simply ask a question or provide a command.
Keep your answers concise and to the point. 
If you need help with math expressions, for e.g. for solving an arithmatic question, use the following format to convert the expression into code:\n\n```python\n<code>\nprint(<result_variable_1>)\nprint(<result_variable_2>)\n```


For example, if you want to calculate the sum of 2 and 3, you can write the following code block:
```python\na = 2\nb = 3\nc = a + b\nprint(c)\n```

For example, the radii of any atom can be found using the following python function in the current execution environment.

def get_radii(atom_id, radius_type="vdw"):
    import pandas as pd
    
    xl = pd.ExcelFile("Elemental_Radii.xlsx") # Load the excel file containing the elemental radii
    df = xl.parse(sheet_name=0, header = 2, index_col=1)
    
    if radius_type is "cov":
        key = 6
    elif radius_type is "vdw":
        key = 7
    else:
        raise ValueError("radius_type not supported")
    if atom_id in df.index:
        return df.loc[atom_id][key]
    else:
        raise ValueError("Elemental symbol not found")

Where, atom_id is the atomic symbol of the element and radius_type = "vdw" for Van der Waals or "cov" for Covalent.

If a user asks for the radius of an element, first ask the user for the atomic symbol, next ask the user for the radius type, and then finally call the function with the atomic symbol and the radius type, using the format specified above.
For example, if the user enquires about the radius of Carbon, and responds with "C" for symbol and "vdw" for radius type, the function call would be:

```python\nr = get_radii("C", radius_type="vdw")\nprint(r)\n```

For example, the statistics associated of zeolite materials in the IZA database can be found usig the following python function in the current execution environment:

When material_id: IZA framework code

def get_pore_data(material_id: str):

    import pandas as pd
    pore_data_df = pd.read_csv("iza_bulk.csv", header = 0, sep = ",", on_bad_lines='skip', index_col="cif")

    if material_id in pore_data_df.index:
        row = pore_data_df.loc[material_id]
        return row.to_dict()
    else:
        raise ValueError("Material ID not found")

If the user provides the material ID, generate the following function fall to get the data of the material:
    ```python\nmaterial_id = "ABW"\npore_data = get_pore_data(material_id)\nprint(f"PLD: {pore_data['pld']}, LCD: {pore_data['lcd']}, ASA: {pore_data['asa']}, AV: {pore_data['av']}, PSD: {pore_data['psd mean']}")\n```


The iza_bulk.csv contains for information for a number of zeolite materials. The cif column contains the IZA framework code, which is used as the index for the DataFrame. The other columns contain various properties of the zeolite materials, such as the pore limiting diameter (PLD), largest cavity diameter (LCD), accessible surface area (ASA), void volume (AV), and pore size distribution (PSD mean).

	Unnamed: 0	pld	lcd	asa	av	psd mean	psd std	n_paths	paths mean	paths std	xdim	ydim	zdim	n_channels
cif														
ABW	0	2.476562	4.242640	2824.94	1546.396	0.994927	0.524161	589.0	1.690595	0.268248	18.5	9.5	16.8	1.0
ACO	1	3.601562	4.604346	5859.62	3399.968	1.090194	0.590122	550.0	1.732020	0.351302	18.6	18.6	18.4	1.0
AEI	2	3.601562	7.337575	19212.96	15170.100	1.418122	0.816693	2376.0	1.444696	0.115261	26.2	25.6	36.9	1.0


So, as another example of usage, if the user wants to idenfity the list of materials with PLD smaller than 3, the following code block can be used:

```python\npore_data_df = pd.read_csv("iza_bulk.csv", header = 0, sep = ",", on_bad_lines='skip', index_col="cif")\nfiltered_df = pore_data_df[pore_data_df['pld'] < 3]\nprint(filtered_df)\n```

If you cannot answer a question or need help, please ask the user for clarification or provide an error message.
"""

def get_system_instructions():
    return SYSTEM_INSTRUCTIONS


# # Simulate your custom Python code execution engine
# def execute_python_code(code):
#     try:
#         exec_globals = {}
#         exec(code, exec_globals)
#         return exec_globals
#     except Exception as e:
#         return str(e)


def execute_code_block(code_block):
    """
    Executes the given code block and captures the standard output (stdout) generated 
    during the execution. The code block is executed in the current environment, allowing
    access to all the functions, libraries, and variables already loaded.
    """

    import io
    import sys

    # Capture the output of the executed code
    old_stdout = sys.stdout  # Save the current stdout
    new_stdout = io.StringIO()  # Create a new StringIO object to capture output
    sys.stdout = new_stdout  # Redirect stdout to the StringIO object
    
    # Define a local context to capture variables during execution
    exec_globals = globals()
    exec_locals = locals()

    try:
        # Execute the code block
        exec(code_block, exec_globals, exec_locals)
        
        # Get the output from the StringIO object
        output = new_stdout.getvalue()
    except Exception as e:
        # In case of exception, capture the error message
        output = str(e)
    finally:
        # Restore the original stdout
        sys.stdout = old_stdout
    
    return output.strip()

# Detect and execute code blocks within the chatbot response
def detect_and_execute_code(chat_history, code_execution_engine):
    last_message = chat_history[-1]['content']
    if "```python" in last_message:
        # Extract the code between the markdown tags
        code_start = last_message.find("```python") + len("```python")
        code_end = last_message.find("```", code_start)
        code_block = last_message[code_start:code_end].strip()
        
        # Execute the extracted code
        code_result = code_execution_engine(code_block)
        
        # Return the code result to inform the next turn
        return code_result
    return None



with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        # bot_message = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])

        print("Chat history:", chat_history)

        # Add user's message to the conversation history
        if len(chat_history) == 0:
            # If it's the first message, start with the notebook context
            chat_history.append({"role": "system", "content": get_system_instructions()})

        chat_history.append({"role": "user", "content": message})
        response = client.chat.completions.create(
            messages=[{"role": m["role"], "content": m["content"]} for m in chat_history], 
            temperature=0, 
            max_tokens=512, 
            model="gpt-4o-mini"
            )

        bot_message = response.choices[0].message.content

        chat_history.append({"role": "assistant", "content": bot_message})

        # Check for any Python code and execute it
        code_result = detect_and_execute_code(chat_history, execute_code_block)

        if code_result:
            # If code was executed, provide the result as feedback to the conversation
            chat_history.append({"role": "assistant", "content": f"Result from code execution: {code_result}"})

        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()