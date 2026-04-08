import nbformat
import os

folder_path = "."  # current directory
final_output = []

for file in os.listdir(folder_path):
    if file.endswith(".ipynb"):
        notebook_path = os.path.join(folder_path, file)
        
        # Add notebook header
        final_output.append(f"\n\n################ {file} ################\n")
        
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                exec_count = cell.get("execution_count", "N/A")
                
                final_output.append(f"\n===== Cell {i} (Execution {exec_count}) =====\n")
                
                for j, output in enumerate(cell.get("outputs", [])):
                    final_output.append(f"--- Output {j} ---\n")
                    
                    if "text" in output:
                        final_output.append(output["text"])
                    
                    elif "data" in output:
                        if "text/plain" in output["data"]:
                            final_output.append(output["data"]["text/plain"])
                    
                    elif output.output_type == "error":
                        error_msg = "\n".join(output.get("traceback", []))
                        final_output.append(error_msg)

# Save everything into ONE file
with open("all_notebooks_outputs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(final_output))

print("✅ All outputs saved to all_notebooks_outputs.txt")