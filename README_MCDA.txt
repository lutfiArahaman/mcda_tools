THIS STEP CAN BE USE FOR BOTH AHP AND FAHP

1. Prepare the Script File
Open a text editor or an Integrated Development Environment (IDE) like Notepad++, VS Code, or PyCharm.

Copy the entire Python script from above and paste it into a new file.

Save the file with a .py extension (e.g., ahp_tool.py).

2. Open ArcGIS Pro
Launch ArcGIS Pro.

3. Create a New Toolbox
In the Catalog pane, right-click on the folder where you want to save the toolbox.

Select New > Toolbox.

Name the toolbox (e.g., AHPToolbox).

4. Add a Script Tool
Right-click the toolbox you just created and select Add > Script.

In the Script Tool Wizard:

Name: Enter a name for the tool (e.g., AHP_Weight_Calculator).

Label: Provide a descriptive label (e.g., AHP Weight Calculator).

Description: Add a description of what the tool does (e.g., Calculates AHP weights and saves them as individual text files).

Click Next.

Script File:

Browse to the .py file you saved earlier (ahp_tool.py) and select it.

Click Next.

Parameters: Add two parameters:

Input Files:

Name: InputFiles

Data Type: File

Direction: Input

MultiValue: Check the box (to allow multiple files).

File Filter: Set to Text Files (*.txt).

Output Directory:

Name: OutputDirectory

Data Type: Folder

Direction: Input.

Click Finish to complete the script tool setup.

