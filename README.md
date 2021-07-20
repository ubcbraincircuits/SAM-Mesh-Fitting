# SAM-Mesh-Fitting
Repository on files needed to perform 3D mesh fitting using UVI dense-mappings through PyTorch3D. Currently focusing on facial expressions of mice.

Our main goal is to ble able to map real behavioural videos onto a 3D surface-based mesh model of the c57bl6 mouse (Synthetic Animated Mouse - SAM) so that, hopefully, analysis of skin deformations (such as those during facial expressions) is more robust. This mapping onto a 3D surface may facilitate amalgamation of data across experiments where current methods fail (ie. Histogram of Gradients HoG from different perspectives, or non-headfixed mice). We hope that downstream analyses will benefit from this normalization of data where 2D videos fail (ie. reaching with a paw can look vastly different from varying perspectives but mean the same thing).

The Google Colab Notebook SAM-mesh-fitting currently fits the mesh onto a single video but can be easily extended to optimize through various camera angles to produce a real 3D mapping with little to no estimation.

# Workflow
### Synthetic Data Generation
### U-Net training for UVI classification & regression (UV coordinates, Index of body parcel)
### Mesh Fitting
We use PyTorch3D to perform differentiable renderings of the mesh model with the UVI maps as textures to render an image that can be compared to the behavioural videos that underwent UVI mappings. This allows us to optimize the per-vertex deformations to minimize the loss (loss is summation of silhouette error, UVI rendering error, as well as mesh regularization). 
##### Camera angle optimization
We first optimize the initial camera position to make the vertex deformation much easier and consistent.
![Camera Fitting through differentiable rendering of silhouettes](assets/camera_fitting.gif)
