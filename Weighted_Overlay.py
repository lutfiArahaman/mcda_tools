import arcpy
from arcpy.sa import WeightedOverlay

class Toolbox(object):
    """Python toolbox for Weighted Overlay Analysis"""

    def __init__(self):
        """Define the toolbox properties"""
        self.label = "Weighted Overlay Toolbox"
        self.alias = "WeightedOverlay"
        self.tools = [WeightedOverlayTool]


class WeightedOverlayTool(object):
    """Tool for Weighted Overlay Analysis"""

    def __init__(self):
        """Define the tool properties"""
        self.label = "Weighted Overlay Analysis"
        self.description = "Performs Weighted Overlay Analysis using 7 input rasters and weights"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []

        # Input Rasters and Weight Files (7 each)
        for i in range(1, 8):
            params.append(arcpy.Parameter(
                displayName=f"Input Raster {i}",
                name=f"input_raster{i}",
                datatype="GPRasterLayer",
                parameterType="Required",
                direction="Input"
            ))

            params.append(arcpy.Parameter(
                displayName=f"Weight File for Raster {i} (.txt)",
                name=f"weight_file{i}",
                datatype="DEFile",
                parameterType="Required",
                direction="Input"
            ))

        # Output Raster
        param_output = arcpy.Parameter(
            displayName="Output Raster",
            name="output_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        params.append(param_output)

        return params

    def isLicensed(self):
        """Check if the tool is licensed to execute"""
        return arcpy.CheckExtension("Spatial") == "Available"

    def updateParameters(self, parameters):
        """Modify parameter properties before validation"""
        return

    def updateMessages(self, parameters):
        """Modify messages for parameters"""
        return

    def execute(self, parameters, messages):
        """Tool execution"""
        arcpy.CheckOutExtension("Spatial")

        # Get parameters
        input_rasters = [parameters[i].valueAsText for i in range(0, 14, 2)]  # Extract raster inputs
...         weight_files = [parameters[i].valueAsText for i in range(1, 14, 2)]  # Extract weight files
...         output_raster = parameters[14].valueAsText  # Output raster
... 
...         # Read weights from text files
...         weights = []
...         try:
...             for weight_file in weight_files:
...                 with open(weight_file, 'r') as f:
...                     weights.append(int(f.read().strip()))
...         except Exception as e:
...             arcpy.AddError(f"Error reading weight files: {e}")
...             raise arcpy.ExecuteError(f"Error reading weight files: {e}")
... 
...         # Validate total weight sum
...         total_weight = sum(weights)
...         if total_weight != 100:
...             raise arcpy.ExecuteError("The sum of all weights must equal 100.")
... 
...         # Create the weighted overlay table string
...         overlay_table_parts = []
...         for raster, weight in zip(input_rasters, weights):
...             overlay_table_parts.append(
...                 f"'{raster}' {weight} 'Value' (1 1; 2 2; 3 3; 4 4; 5 5; NODATA NODATA)"
...             )
...         overlay_table = f"({'; '.join(overlay_table_parts)});1 9 1"
... 
...         # Perform Weighted Overlay
...         result_raster = WeightedOverlay(in_weighted_overlay_table=overlay_table)
... 
...         # Save the result
...         result_raster.save(output_raster)
... 
...         # Release the Spatial Analyst extension
...         arcpy.CheckInExtension("Spatial")
... 
