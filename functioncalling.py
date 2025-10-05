from google import genai
from google.genai import types
import json
import matplotlib.pyplot as plt
from lightkurve import search_targetpixelfile

# --- 1. Define the Function and Tool Schema ---

# Define the actual Python function that the model can call
def plot_pixelfile(starname):
    """
    Searches for and plots the Target Pixel File (TPF) for a given star name 
    from the TESS or Kepler/K2 mission data using lightkurve, specifically for Quarter 16.
    """
    print(f"Executing plot_pixelfile for: {starname}...")
    try:
        # NOTE: This line requires lightkurve to be installed and working.
        # It will also download data, which takes time and requires an internet connection.
        pixelfile = search_targetpixelfile(starname).download()
        
        # Plot the first frame of the TPF
        # In a real environment, this will open a plot window.
        pixelfile.plot()
        plt.show()
        
        # Display the plot
        # plt.show() # Uncomment this line to display the plot when running locally.
        
        return f"Plotting process initiated for star '{starname}'. The plot window should appear momentarily."
    except Exception as e:
        # Return an error message if the search or download fails
        return f"Error plotting pixelfile for '{starname}': {e}"

# Define the Tool Schema for the Gemini model
plot_pixelfile_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="plot_pixelfile",
            description="Searches for and plots the Target Pixel File (TPF) for a given star name (e.g., 'KIC 8462852', 'Kepler-10') from the TESS or Kepler/K2 mission data, specifically for Quarter 16.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "starname": types.Schema(
                        type=types.Type.STRING,
                        description="The name or target ID of the astronomical target (star, exoplanet system) to plot.",
                    )
                },
                required=["starname"],
            ),
        )
    ]
)

# --- 2. Define the Function Handler Map ---

# A dictionary to map the model's function call name (string) to the actual Python function object
available_functions = {
    "plot_pixelfile": plot_pixelfile,
}

# --- 3. Implement the Two-Turn Conversation Loop ---

# Replace 'YOUR_API_KEY' with your actual Google AI API key
# client = genai.Client(api_key="AIzaSyA9Xmklq0FvdkIohDdECcAVIjhGpYqw6xw")
# NOTE: It's best practice to initialize the client without an argument if the API key
# is set in the environment variable (GEMINI_API_KEY).
client = genai.Client(api_key="AIzaSyA9Xmklq0FvdkIohDdECcAVIjhGpYqw6xw")

# The user's prompt that triggers the function call
user_prompt = "Can you explain me about exoplanets"

# --- Turn 1: Model decides to call the function ---

print("--- Turn 1: Sending prompt and tools to the model ---")

# Pass the tools (function schema) to the generate_content call
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_prompt,
    config=types.GenerateContentConfig(tools=[plot_pixelfile_tool])
)

# Check if the model decided to call a function
if response.function_calls:
    print("\n✅ Model has called a function.")
    function_call = response.function_calls[0]
    
    function_name = function_call.name
    function_args = dict(function_call.args)

    print(f"Function Name: {function_name}")
    print(f"Arguments: {function_args}")
    
    # --- Execute the Function ---
    if function_name in available_functions:
        function_to_call = available_functions[function_name]
        
        # Call the Python function with the arguments provided by the model
        function_output = function_to_call(**function_args)
        
        print(f"\nFunction Execution Result: {function_output}")

        # --- Turn 2: Send the function output back to the model ---
        print("\n--- Turn 2: Sending function result back to the model ---")

        # Build the conversation turns in the required order:
        # 1) user turn (the original user prompt)
        # 2) assistant turn with the function call
        # 3) tool turn with the function response
        contents = [
            # User turn
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            ),

            # Assistant turn containing the function call the model made
            types.Content(
                role="assistant",
                parts=[
                    types.Part.from_function_call(
                        name=function_name,
                        args=function_args,
                    )
                ],
            ),

            # Tool turn with the function's result
            types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=function_name,
                        response={"result": function_output},
                    )
                ],
            ),
        ]

        # Call the model again with the history and function result
        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=[plot_pixelfile_tool]) # Include tools again for context
        )

        print("\n✅ Final Model Response:")
        print(final_response.text)

    else:
        print(f"Error: Function '{function_name}' not found in available_functions.")

else:
    print("\nModel did not call a function. Text response:")
    print(response.text)