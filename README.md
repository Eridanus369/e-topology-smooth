# E Topology Smooth / E 拓扑平滑

[![Blender Version](https://img.shields.io/badge/Blender-4.5%2B-orange)](https://blender.org)
[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

**English** | [**中文**](#中文说明)

Advanced mesh topology smoothing tool with G0-G4 continuity analysis and multi-algorithm curve fitting for Blender 4.5+.

---

##  Features

###   Continuity Analysis (G0-G4)
- **G0**: Position continuity - measures vertex displacement from neighbors
- **G1**: Tangent continuity - analyzes edge direction consistency
- **G2**: Curvature continuity - evaluates surface smoothness
- **G3**: Curvature rate of change
- **G4**: Curvature acceleration

###   Multi-Algorithm Curve Fitting
- **Auto Detect**: Automatically chooses best fitting method
- **Circle**: Perfect circular fitting (Kasa method)
- **Ellipse**: Elliptical fitting (direct least squares)
- **B-Spline**: Flexible curve fitting (requires scipy)
- **Power Function**: y = a*x^b + c fitting
- **Polynomial**: Polynomial curve fitting

###   Core Features
- **Corner Defect Detection**: Automatically identifies and repairs sharp corners
- **Outlier Detection**: Statistical analysis to identify problematic vertices
- **Uniform Spacing**: Arc-length parameterization for evenly distributed points
- **Flatten to Plane**: Project curves to optimal plane before fitting
- **Multi-threading Support**: Performance optimization for large meshes

###   User Interface
- **Sidebar Panel**: Easy access in 3D View > Edit Mode
- **Real-time Analysis**: Immediate feedback on mesh quality
- **Adjustable Parameters**: Fine-tune detection thresholds
- **Bilingual Support**: English and Chinese UI

---

##   Requirements

- **Blender**: 4.5.0 or newer
- **Dependencies** (included in package):
  - NumPy 2.4.2
  - SciPy 1.17.0 (optional, advanced features)

---

##   Installation

### Method 1: Direct Download (Recommended)
1. Download the latest release from [Releases](https://github.com/Eridanus369/e-topology-smooth/releases)
2. In Blender, go to **Edit > Preferences > Add-ons**
3. Click **Install...** and select the downloaded ZIP file
4. Enable **"E Topology Smooth"**

### Method 2: Git Clone
```bash
git clone https://github.com/Eridanus369/e-topology-smooth.git
cd e-topology-smooth
# Copy folder to Blender add-ons directory