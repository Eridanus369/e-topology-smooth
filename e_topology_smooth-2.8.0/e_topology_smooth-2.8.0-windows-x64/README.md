# E Topology Smooth / E 拓扑平滑

[![Blender Version](https://img.shields.io/badge/Blender-5.2%2B-orange)](https://blender.org)
[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://python.org)

**English** | [**中文**](#中文说明)

Advanced mesh topology smoothing tool with G0-G4 continuity analysis and multi-algorithm curve fitting for Blender 5.2+.

---

## Features

### Continuity Analysis (G0-G4)
- **G0**: Position continuity - measures vertex displacement from neighbors
- **G1**: Tangent continuity - analyzes edge direction consistency
- **G2**: Curvature continuity - evaluates surface smoothness
- **G3**: Curvature rate of change
- **G4**: Curvature acceleration

### Multi-Algorithm Curve Fitting
- **Auto Detect**: Automatically chooses best fitting method
- **Circle**: Perfect circular fitting (Kasa method)
- **Ellipse**: Elliptical fitting (direct least squares)
- **B-Spline**: Flexible curve fitting (requires scipy)
- **Power Function**: y = a*x^b + c fitting
- **Polynomial**: Polynomial curve fitting

### Core Features
- **Corner Defect Detection**: Automatically identifies and repairs sharp corners
- **Outlier Detection**: Statistical analysis to identify problematic vertices
- **Uniform Spacing**: Arc-length parameterization for evenly distributed points
- **Flatten to Plane**: Project curves to optimal plane before fitting
- **Multi-threading Support**: Performance optimization for large meshes
- **Integrated Space Tool**: Uniform vertex spacing along loops (from LoopTools)

### User Interface
- **Sidebar Panel**: Easy access in 3D View > Edit Mode
- **Real-time Analysis**: Immediate feedback on mesh quality
- **Adjustable Parameters**: Fine-tune detection thresholds
- **Bilingual Support**: English and Chinese UI

---

## Requirements

- **Blender**: 5.2.0 or newer
- **Python**: 3.14
- **Dependencies** (included in package):
  - NumPy 2.4.2
  - SciPy 1.18.1 (optional, advanced features) — built for CPython 3.14 (`cp314`)

> ⚠️ **Note**: Blender 5.2+ embeds Python 3.14. If you previously used a SciPy wheel built for Python 3.13 (`cp313`), the extension system will report an incompatibility error. Version 2.8.0 fixes this by bundling the `cp314` SciPy wheel.

---

## Installation

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
```

### Method 3: Extension Installation
If you install via the Blender extension system and the SciPy wheel is missing, you can obtain the correct wheel from PyPI:

**https://pypi.org/project/scipy/#files**

Search for `cp314` and pick the file matching your platform (e.g. `scipy-1.18.1-cp314-cp314-win_amd64.whl` for Windows x64), then place it in the extension's `wheels/` directory.

---

## Changelog

### v2.8.0 (2026-09-10)
- **Python 3.14 Compatibility Fix**: Resolved the "This Python version (3.14) isn't compatible with (3.13)" error by updating the bundled SciPy wheel from `cp313` to `cp314`
- **Blender 5.2+ Support**: Raised the minimum Blender version requirement to 5.2.0
- **Robust SciPy Import**: Broadened the exception handling for SciPy import, so the add-on still loads with reduced functionality when SciPy is unavailable (e.g. ABI mismatch, missing module)
- **Version Update**: Bumped add-on version to 2.8.0

### v2.7.8 (2026-03-31)
- **BOM Character Fix**: Removed BOM character from __init__.py file to resolve startup errors
- **Version Update**: Updated add-on version to 2.7.8
- **Performance Optimizations**: Further improvements to startup performance

### v2.7.7 (2026-03-28)
- **Blender 5.1 Support**: Updated compatibility to Blender 5.1+ (Python 3.13)
- **Integrated Space Tool**: Added LoopTools Space algorithm integration for uniform vertex spacing
- **Enhanced Curve Fitting**: Improved curve fitting repair with automatic space distribution
- **Performance Optimizations**: Better memory management and faster processing

### v2.7.1
- Initial release with G0-G4 continuity analysis
- Multi-algorithm curve fitting support

---

## 中文说明

### 功能特点

#### 连续性分析 (G0-G4)
- **G0**: 位置连续性 - 测量顶点与邻居的位移
- **G1**: 切线连续性 - 分析边方向一致性
- **G2**: 曲率连续性 - 评估曲面平滑度
- **G3**: 曲率变化率
- **G4**: 曲率加速度

#### 多算法曲线拟合
- **自动检测**: 自动选择最佳拟合方法
- **圆形**: 完美圆形拟合 (Kasa方法)
- **椭圆**: 椭圆拟合 (直接最小二乘法)
- **B样条**: 灵活曲线拟合 (需要scipy)
- **幂函数**: y = a*x^b + c 拟合
- **多项式**: 多项式曲线拟合

#### 核心功能
- **尖角缺陷检测**: 自动识别和修复尖锐角点
- **异常值检测**: 统计分析识别问题顶点
- **均匀间距**: 弧长参数化实现均匀分布点
- **压平到平面**: 拟合前将曲线投影到最佳平面
- **多线程支持**: 大型网格性能优化
- **集成Space工具**: 沿循环均匀分布顶点 (来自LoopTools)

### 系统要求
- **Blender**: 5.2.0 或更新版本
- **Python**: 3.14
- **依赖项** (包含在包中):
  - NumPy 2.4.2
  - SciPy 1.18.1 (可选，高级功能) — 为 CPython 3.14 编译 (`cp314`)

> ⚠️ **注意**: Blender 5.2+ 内嵌 Python 3.14。如果之前使用的是为 Python 3.13 编译的 SciPy wheel (`cp313`)，扩展系统会报 Python 版本不兼容错误。v2.8.0 通过捆绑 `cp314` 的 SciPy wheel 修复了该问题。

### 安装说明

#### 方式一：直接下载（推荐）
1. 从 [Releases](https://github.com/Eridanus369/e-topology-smooth/releases) 下载最新版本
2. 在 Blender 中打开 **编辑 > 偏好设置 > 插件**
3. 点击 **安装...** 并选择下载的 ZIP 文件
4. 启用 **"E Topology Smooth"**

#### 方式二：Git 克隆
```bash
git clone https://github.com/Eridanus369/e-topology-smooth.git
cd e-topology-smooth
# 将文件夹复制到 Blender 插件目录
```

#### 方式三：扩展安装
如果通过 Blender 扩展系统安装后提示缺少 SciPy wheel，可以从 PyPI 获取对应的 wheel：

**https://pypi.org/project/scipy/#files**

搜索 `cp314`，选择与你平台匹配的文件（例如 Windows x64 使用 `scipy-1.18.1-cp314-cp314-win_amd64.whl`），然后放入扩展的 `wheels/` 目录。

### 更新日志

#### v2.8.0 (2026-09-10)
- **Python 3.14 兼容性修复**: 通过将捆绑的 SciPy wheel 从 `cp313` 更新为 `cp314`，解决了 "This Python version (3.14) isn't compatible with (3.13)" 报错
- **Blender 5.2+ 支持**: 将最低 Blender 版本要求提升至 5.2.0
- **健壮的 SciPy 导入**: 扩大了 SciPy 导入的异常捕获范围，即使 SciPy 不可用（ABI 不匹配、模块缺失等），插件仍能以降级模式加载
- **版本更新**: 将插件版本更新至 2.8.0

#### v2.7.8 (2026-03-31)
- **BOM字符修复**: 移除 __init__.py 文件中的 BOM 字符，解决启动错误
- **版本更新**: 将插件版本更新至 2.7.8
- **性能优化**: 进一步改进启动性能

#### v2.7.7 (2026-03-28)
- **Blender 5.1 支持**: 更新兼容性至 Blender 5.1+ (Python 3.13)
- **集成Space工具**: 添加 LoopTools Space 算法集成，实现顶点均匀分布
- **增强曲线拟合**: 改进曲线拟合修复功能，自动空间分布
- **性能优化**: 更好的内存管理和更快的处理速度

---

## License

GPL-3.0-or-later © 2024-2026 Eridanus