import subprocess



# Replace 'path/to/your/asp/program.lp' with the actual path to your ASP program file
asp_program_path = 'dec_logic.sp'

# Define the command to run Clingo with the ASP program
clingo_command = ['clingo', asp_program_path]

try:
    # Run Clingo and capture the output
    output = subprocess.check_output(clingo_command, universal_newlines=True)

    # Print the output
    print(output)

except subprocess.CalledProcessError as e:
    # If an error occurs, print the error message
    print("Error:", e)
