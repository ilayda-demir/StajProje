# 3D Model Sectioning Tool

This application is designed for working with 3D models (STL/OBJ) and
extracting section data through user interaction.

## Features

-   Load 3D models (STL/OBJ) into the OpenGL viewer.
-   Pick points on the model with **CTRL + Left Click** or manually enter point coordinates using the UI fields.
-   Take sections along the **X, Y, or Z axis** at the selected point.
-   Export section coordinates to a `.txt` file.

## Coordinate Systems

-   **Normalized Space**: Coordinates are scaled and centered for
    consistent visualization.
-   **Original Space**: Coordinates preserve the model's original
    values.

Depending on the selected option, the exported `.csv` file will contain
either normalized or original coordinates.

## How to Use

1.  Run the application (`main.py`).
2.  Load a 3D model using the **Load Model** button.
3.  Select a point on the model:
    -   Use **CTRL + Left Click** to pick directly on the model.
    -   Or manually enter coordinates in the input fields.
4.  Choose the desired axis (X, Y, or Z) and click **Compute Section**.
5.  Export the results using the **Export Section** button. The output
    `.txt` file will be saved with the chosen coordinate system.

## Requirements

Install dependencies via:

``` bash
pip install -r requirements.txt
```

## Notes

-   Works with large STL models efficiently.
-   User can freely switch between **normalized** and **original**
    coordinate systems before exporting.

------------------------------------------------------------------------

Developed with **PyQt5**, **OpenGL**, and **Trimesh**.
