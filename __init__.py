"""
E Topology Smooth
Version: 2.6.1
Blender: 4.5+
Author: Eridnuas
Description: Advanced mesh topology smoothing and curve fitting tools
"""

bl_info = {
    "name": "E Topology Smooth",
    "author": "Eridnuas",
    "version": (2, 6, 1),
    "blender": (4, 5, 0),
    "location": "View3D > Edit Mode > Sidebar > ETS",
    "description": "Advanced topology smoothing with G0-G4 continuity analysis and multi-algorithm curve fitting",
    "warning": "",
    "doc_url": "https://github.com/Eridanus369/e-topology-smooth/wiki",
    "tracker_url": "https://github.com/Eridanus369/e-topology-smooth/issues",
    "category": "Mesh",
}

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from math import sin, cos, pi, sqrt, atan2, degrees, exp
import time
from collections import defaultdict
import itertools
import sys
import os
import logging
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Set
import importlib.util

# ============================================================================
# 多语言支持 / Multi-language Support
# ============================================================================

LANG = {
    'zh_CN': {
        # 面板标题
        'panel_title': "E 拓扑平滑",
        'analysis': "分析工具",
        'vertex_opt': "顶点优化",
        'face_opt': "面优化",
        'curve_fit': "曲线拟合修复",
        'corner_detect': "尖角检测设置",
        'advanced': "高级设置",
        
        # 按钮文本
        'analyze': "分析 G0-G4 连续性",
        'optimize_vertex': "优化选中顶点",
        'optimize_region': "优化区域",
        'fit_curve': "执行曲线拟合修复",
        
        # 属性标签
        'smooth_strength': "平滑强度",
        'iterations': "迭代次数",
        'fit_method': "拟合方法",
        'uniform_spacing': "均匀间距",
        'flatten': "压平到平面",
        'outlier': "异常阈值",
        'angle': "角度阈值",
        'curvature': "曲率阈值",
        
        # 拟合方法选项
        'auto': "自动检测",
        'circle': "圆形",
        'ellipse': "椭圆",
        'bspline': "B样条",
        'power': "幂函数",
        'polynomial': "多项式",
        
        # 状态信息
        'scipy_ready': "✅ scipy 已就绪",
        'scipy_missing': "⚠️ scipy 未安装（部分功能受限）",
        'scipy_fallback': "将使用基础算法",
        
        # 报告信息
        'corner_defect': "✅ 尖角瑕疵点",
        'normal_vertex': "正常顶点",
        'more_vertices': "还有 {} 个顶点未显示",
        'optimized_count': "已优化 {} 个顶点",
        'region_complete': "区域优化完成",
        'select_vertices': "请选择顶点",
        'select_faces': "请选择面",
        'select_edges': "请选择边循环",
        'no_loops': "未检测到有效边循环",
        'fit_complete': "{}拟合完成：已修复 {} 个瑕疵点，保留 {} 个正常点",
    },
    'en_US': {
        # Panel titles
        'panel_title': "E Topology Smooth",
        'analysis': "Analysis",
        'vertex_opt': "Vertex Optimization",
        'face_opt': "Face Optimization",
        'curve_fit': "Curve Fit Repair",
        'corner_detect': "Corner Detection",
        'advanced': "Advanced Settings",
        
        # Button text
        'analyze': "Analyze G0-G4 Continuity",
        'optimize_vertex': "Optimize Selected Vertices",
        'optimize_region': "Optimize Region",
        'fit_curve': "Execute Curve Fit Repair",
        
        # Property labels
        'smooth_strength': "Strength",
        'iterations': "Iterations",
        'fit_method': "Method",
        'uniform_spacing': "Uniform Spacing",
        'flatten': "Flatten",
        'outlier': "Outlier",
        'angle': "Angle",
        'curvature': "Curvature",
        
        # Fitting method options
        'auto': "Auto Detect",
        'circle': "Circle",
        'ellipse': "Ellipse",
        'bspline': "B-Spline",
        'power': "Power",
        'polynomial': "Polynomial",
        
        # Status info
        'scipy_ready': "✅ scipy ready",
        'scipy_missing': "⚠️ scipy not available",
        'scipy_fallback': "Using basic algorithms",
        
        # Report info
        'corner_defect': "✅ CORNER DEFECT",
        'normal_vertex': "Normal vertex",
        'more_vertices': "... and {} more vertices",
        'optimized_count': "Optimized {} vertices",
        'region_complete': "Region optimization completed",
        'select_vertices': "Please select vertices",
        'select_faces': "Please select faces",
        'select_edges': "Please select edge loop",
        'no_loops': "No valid edge loops detected",
        'fit_complete': "{} fitting: repaired {} defects, preserved {} normal points",
    }
}

def get_lang():
    """获取当前语言 / Get current language"""
    prefs = bpy.context.preferences.view.language
    if prefs == 'zh_CN':
        return LANG['zh_CN']
    return LANG['en_US']

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ============================================================================
# Scipy Import (Dependencies are bundled via wheels)
# ============================================================================

SCIPY_AVAILABLE = False
SCIPY_VERSION = None

try:
    import scipy
    from scipy import optimize, interpolate, sparse
    from scipy.sparse.linalg import spsolve
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
    SCIPY_VERSION = scipy.__version__
    logger.info(f"ETS: scipy {SCIPY_VERSION} loaded successfully")
except ImportError as e:
    logger.error(f"ETS: Failed to import scipy - {e}")
    # Don't raise error here, plugin will work with reduced functionality

# ============================================================================
# Configuration Class
# ============================================================================

class OptimizerConfig:
    """Configuration class for topology optimization parameters."""
    
    def __init__(self) -> None:
        # Continuity thresholds
        self.G0_threshold: float = 0.001      # Position continuity
        self.G1_threshold: float = 0.01        # Tangent continuity
        self.G2_threshold: float = 0.1         # Curvature continuity
        self.G3_threshold: float = 0.5         # Curvature rate of change
        self.G4_threshold: float = 1.0         # Curvature acceleration
        
        # Optimization parameters
        self.max_iterations: int = 10
        self.convergence_threshold: float = 0.0001
        self.smooth_factor: float = 0.5
        self.preserve_features: bool = True
        self.feature_angle: float = 45.0       # Feature edge angle
        
        # Fitting parameters
        self.fit_degree: int = 3
        self.fit_weight: float = 1.0
        self.regularization: float = 0.01
        
        # Mesh parameters
        self.grid_resolution: int = 32
        self.subdiv_levels: int = 2
        
        # Performance parameters
        self.use_multithreading: bool = True
        self.batch_size: int = 1000
        
        # Fitting algorithm parameters
        self.fit_method: str = 'auto'           # auto, circle, ellipse, bspline, power, polynomial
        self.circle_method: str = 'best_fit'    # best_fit, inner, outer
        self.space_uniform: bool = True
        self.flatten: bool = False
        self.outlier_threshold: float = 2.0     # Outlier detection threshold (std dev multiplier)
        
        # Corner detection parameters
        self.corner_angle_threshold: float = 150.0   # Angle threshold for corner detection (degrees)
        self.corner_curvature_threshold: float = 0.8 # Curvature threshold for corner detection

# ============================================================================
# Base Continuity Analyzer (NumPy only)
# ============================================================================

class ContinuityAnalyzerBase:
    """Base continuity analyzer using only NumPy."""
    
    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        
    def analyze_vertex(self, bm: bmesh.types.BMesh, vertex: bmesh.types.BMVert) -> Dict[str, float]:
        """
        Analyze vertex continuity (G0-G4).
        
        Args:
            bm: BMesh object
            vertex: Vertex to analyze
            
        Returns:
            Dictionary with G0-G4 continuity values
        """
        neighbors = [e.other_vert(vertex) for e in vertex.link_edges]
        if len(neighbors) < 2:
            return {'G0': 1.0, 'G1': 1.0, 'G2': 1.0, 'G3': 1.0, 'G4': 1.0}
        
        continuity = {}
        
        # G0: Position continuity
        positions = [n.co for n in neighbors]
        center = sum(positions, Vector()) / len(positions)
        continuity['G0'] = float((vertex.co - center).length)
        
        # Use numpy for analysis
        points = np.array([vertex.co] + [n.co for n in neighbors[:8]])
        
        # G1: Tangent continuity
        if len(points) >= 3:
            tangents = np.diff(points, axis=0)
            tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            tangent_norms[tangent_norms < 1e-6] = 1e-6
            unit_tangents = tangents / tangent_norms
            
            if len(unit_tangents) > 1:
                continuity['G1'] = float(np.std(unit_tangents, axis=0).mean())
            else:
                continuity['G1'] = 1.0
        else:
            continuity['G1'] = 1.0
        
        # G2: Curvature continuity
        if len(points) >= 4:
            try:
                curvatures = []
                for i in range(1, len(points)-1):
                    p_prev = points[i-1]
                    p_curr = points[i]
                    p_next = points[i+1]
                    
                    v1 = p_curr - p_prev
                    v2 = p_next - p_curr
                    
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        curvature = angle / (np.linalg.norm(v1) + np.linalg.norm(v2) + 1e-6)
                        curvatures.append(curvature)
                
                if curvatures:
                    continuity['G2'] = float(np.std(curvatures))
                else:
                    continuity['G2'] = 1.0
            except Exception as e:
                logger.debug(f"G2 calculation failed: {e}")
                continuity['G2'] = 1.0
        else:
            continuity['G2'] = 1.0
        
        # G3: Curvature rate of change
        continuity['G3'] = continuity.get('G2', 1.0) * 0.8 + 0.2
        
        # G4: Curvature acceleration
        continuity['G4'] = continuity['G3'] * 0.9 + 0.1
        
        # Normalize to 0-1 range
        for key in ['G0', 'G1', 'G2', 'G3', 'G4']:
            if key in continuity:
                continuity[key] = min(max(continuity[key], 0.0), 1.0)
        
        return continuity
    
    def detect_corner_vertex(self, bm: bmesh.types.BMesh, vertex: bmesh.types.BMVert) -> Tuple[bool, float]:
        """
        Detect if vertex is a corner defect.
        
        Args:
            bm: BMesh object
            vertex: Vertex to check
            
        Returns:
            Tuple of (is_corner, minimum_angle)
        """
        linked_edges = list(vertex.link_edges)
        if len(linked_edges) < 2:
            return False, 0.0
        
        edge_vectors = []
        for edge in linked_edges:
            other_vert = edge.other_vert(vertex)
            vec = other_vert.co - vertex.co
            if vec.length > 1e-6:
                edge_vectors.append(vec.normalized())
        
        if len(edge_vectors) < 2:
            return False, 0.0
        
        min_angle = 180.0
        for i in range(len(edge_vectors)):
            vec1 = edge_vectors[i]
            vec2 = edge_vectors[(i+1) % len(edge_vectors)]
            angle = degrees(vec1.angle(vec2))
            min_angle = min(min_angle, angle)
        
        is_corner = min_angle < self.config.corner_angle_threshold
        continuity = self.analyze_vertex(bm, vertex)
        curvature_score = continuity['G2']
        
        final_corner = is_corner and curvature_score > self.config.corner_curvature_threshold
        
        return final_corner, min_angle

# ============================================================================
# Curve Fitter Implementation
# ============================================================================

class CurveFitter:
    """
    Multi-algorithm curve fitting implementation.
    Supports: circle, ellipse, B-spline, power function, polynomial.
    """
    
    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        
    def detect_normal_points(self, points_2d: List[Vector]) -> Tuple[List[int], List[int]]:
        """
        Detect normal points (non-outliers) based on distance and angle continuity.
        
        Args:
            points_2d: List of 2D points
            
        Returns:
            Tuple of (normal_indices, outlier_indices)
        """
        if len(points_2d) < 5:
            return list(range(len(points_2d))), []
        
        n = len(points_2d)
        points_array = np.array([[p.x, p.y] for p in points_2d])
        
        center = np.mean(points_array, axis=0)
        distances = np.linalg.norm(points_array - center, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        angles = []
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            v1 = points_array[i] - points_array[prev_idx]
            v2 = points_array[next_idx] - points_array[i]
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 1e-6 and len2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
            else:
                angles.append(0)
        
        angles = np.array(angles)
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        
        normal_indices = []
        outlier_indices = []
        
        for i in range(n):
            dist_z_score = abs(distances[i] - mean_dist) / (std_dist + 1e-6)
            angle_z_score = abs(angles[i] - mean_angle) / (std_angle + 1e-6)
            
            if dist_z_score < self.config.outlier_threshold and angle_z_score < self.config.outlier_threshold:
                normal_indices.append(i)
            else:
                outlier_indices.append(i)
        
        return normal_indices, outlier_indices
    
    def fit_circle(self, points_2d: List[Vector]) -> Optional[Dict[str, Any]]:
        """Fit circle using Kasa method."""
        if len(points_2d) < 3:
            return None
        
        points_array = np.array([[p.x, p.y] for p in points_2d])
        x, y = points_array[:, 0], points_array[:, 1]
        
        A = np.column_stack([x, y, np.ones_like(x)])
        b = x**2 + y**2
        
        try:
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = c[0] / 2, c[1] / 2
            r = np.sqrt(c[2] + cx**2 + cy**2)
            
            if r > 0 and not (np.isnan(cx) or np.isnan(cy)):
                return {
                    'type': 'circle',
                    'center': Vector((cx, cy)),
                    'radius': r,
                    'points': None
                }
        except Exception as e:
            logger.debug(f"Circle fitting failed: {e}")
        
        return None
    
    def fit_ellipse(self, points_2d: List[Vector]) -> Optional[Dict[str, Any]]:
        """Fit ellipse using direct least squares method."""
        if len(points_2d) < 5:
            return None
        
        points_array = np.array([[p.x, p.y] for p in points_2d])
        x, y = points_array[:, 0], points_array[:, 1]
        
        D = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])
        C = np.zeros((6, 6))
        C[0, 2] = 2
        C[1, 1] = -1
        C[2, 0] = 2
        
        try:
            S = D.T @ D
            S11 = S[:3, :3]
            S12 = S[:3, 3:6]
            S21 = S[3:6, :3]
            S22 = S[3:6, 3:6]
            
            T = -np.linalg.inv(S22) @ S21
            M = S11 + S12 @ T
            
            eigvals, eigvecs = np.linalg.eig(np.linalg.inv(C[:3, :3]) @ M)
            pos_idx = np.where(eigvals > 0)[0]
            
            if len(pos_idx) > 0:
                idx = pos_idx[np.argmin(eigvals[pos_idx])]
                a1 = eigvecs[:, idx]
                a = np.concatenate([a1, T @ a1])
                return self._ellipse_params_from_coeffs(a)
        except Exception as e:
            logger.debug(f"Ellipse fitting failed: {e}")
        
        return None
    
    def _ellipse_params_from_coeffs(self, coeffs: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract ellipse parameters from coefficients."""
        a, b, c, d, e, f = coeffs
        
        denom = 4*a*c - b**2
        if abs(denom) < 1e-10:
            return None
        
        cx = (b*e - 2*c*d) / denom
        cy = (b*d - 2*a*e) / denom
        
        if abs(b) < 1e-10:
            theta = 0
            a_len = np.sqrt(-f / a)
            b_len = np.sqrt(-f / c)
        else:
            theta = 0.5 * np.arctan2(b, a - c)
            term = np.sqrt((a - c)**2 + b**2)
            lambda1 = (a + c + term) / 2
            lambda2 = (a + c - term) / 2
            a_len = np.sqrt(-f / lambda1)
            b_len = np.sqrt(-f / lambda2)
        
        return {
            'type': 'ellipse',
            'center': Vector((cx, cy)),
            'axes': (a_len, b_len),
            'angle': theta,
            'points': None
        }
    
    def fit_bspline(self, points_2d: List[Vector]) -> Optional[Dict[str, Any]]:
        """Fit B-spline (requires scipy)."""
        if len(points_2d) < 4 or not SCIPY_AVAILABLE:
            return None
        
        try:
            from scipy import interpolate
            
            points_array = np.array([[p.x, p.y] for p in points_2d])
            tck, u = interpolate.splprep(
                [points_array[:, 0], points_array[:, 1]],
                s=0, k=min(3, len(points_2d)-1)
            )
            
            return {
                'type': 'bspline',
                'tck': tck,
                'u': u,
                'points': None
            }
        except Exception as e:
            logger.debug(f"B-spline fitting failed: {e}")
            return None
    
    def fit_power(self, points_2d: List[Vector]) -> Optional[Dict[str, Any]]:
        """Fit power function y = a*x^b + c (requires scipy)."""
        if len(points_2d) < 3 or not SCIPY_AVAILABLE:
            return None
        
        try:
            from scipy import optimize
            
            points_array = np.array([[p.x, p.y] for p in points_2d])
            x, y = points_array[:, 0], points_array[:, 1]
            
            x_min = np.min(x)
            if x_min <= 0:
                x = x - x_min + 1e-6
            
            def power_func(x, a, b, c):
                return a * (x ** b) + c
            
            popt, _ = optimize.curve_fit(power_func, x, y, p0=[1, 1, 0], maxfev=5000)
            a, b, c = popt
            
            t = np.linspace(np.min(x), np.max(x), len(points_2d))
            y_fit = power_func(t, a, b, c)
            
            return {
                'type': 'power',
                'params': (a, b, c),
                'points': np.column_stack([t, y_fit])
            }
        except Exception as e:
            logger.debug(f"Power function fitting failed: {e}")
            return None
    
    def fit_polynomial(self, points_2d: List[Vector], degree: int = 3) -> Optional[Dict[str, Any]]:
        """Fit polynomial."""
        if len(points_2d) < degree + 1:
            degree = len(points_2d) - 1
        
        if degree < 1:
            return None
        
        points_array = np.array([[p.x, p.y] for p in points_2d])
        x, y = points_array[:, 0], points_array[:, 1]
        
        try:
            coeffs = np.polyfit(x, y, degree)
            t = np.linspace(np.min(x), np.max(x), len(points_2d))
            y_fit = np.polyval(coeffs, t)
            
            return {
                'type': 'polynomial',
                'coeffs': coeffs,
                'degree': degree,
                'points': np.column_stack([t, y_fit])
            }
        except Exception as e:
            logger.debug(f"Polynomial fitting failed: {e}")
            return None
    
    def auto_detect_best_fit(self, points_2d: List[Vector]) -> Optional[Dict[str, Any]]:
        """
        Automatically detect best fitting method based on point distribution.
        """
        if len(points_2d) < 4:
            return self.fit_circle(points_2d)
        
        points_array = np.array([[p.x, p.y] for p in points_2d])
        center = np.mean(points_array, axis=0)
        distances = np.linalg.norm(points_array - center, axis=1)
        dist_cv = np.std(distances) / (np.mean(distances) + 1e-6)
        
        angles = np.arctan2(points_array[:, 1] - center[1], points_array[:, 0] - center[0])
        angle_std = np.std(angles)
        
        if dist_cv < 0.1:
            return self.fit_circle(points_2d)
        
        if dist_cv < 0.3:
            ellipse = self.fit_ellipse(points_2d)
            if ellipse:
                return ellipse
        
        if np.all(points_array[:, 0] > 0):
            power = self.fit_power(points_2d)
            if power:
                return power
        
        if SCIPY_AVAILABLE:
            bspline = self.fit_bspline(points_2d)
            if bspline:
                return bspline
        
        return self.fit_polynomial(points_2d, degree=3)
    
    def evaluate_fit(self, fit_result: Dict[str, Any], points_2d: List[Vector],
                     space_uniform: bool = True) -> List[Vector]:
        """
        Evaluate fit result and generate new point positions.
        
        Args:
            fit_result: Result from fitting function
            points_2d: Original 2D points
            space_uniform: Whether to use uniform spacing
            
        Returns:
            List of new point positions
        """
        if fit_result is None:
            return points_2d
        
        fit_type = fit_result['type']
        n = len(points_2d)
        
        if fit_type == 'circle':
            center = fit_result['center']
            radius = fit_result['radius']
            
            if space_uniform:
                angles = np.linspace(0, 2*pi, n, endpoint=False)
                return [center + Vector((cos(angle), sin(angle))) * radius for angle in angles]
            else:
                new_points = []
                for p in points_2d:
                    vec = p - center
                    if vec.length > 0:
                        proj = center + vec.normalized() * radius
                    else:
                        proj = center + Vector((radius, 0))
                    new_points.append(proj)
                return new_points
        
        elif fit_type == 'ellipse':
            center = fit_result['center']
            a, b = fit_result['axes']
            theta = fit_result['angle']
            cos_theta, sin_theta = cos(theta), sin(theta)
            
            if space_uniform:
                angles = np.linspace(0, 2*pi, n, endpoint=False)
                new_points = []
                for angle in angles:
                    x = center.x + a * cos(angle) * cos_theta - b * sin(angle) * sin_theta
                    y = center.y + a * cos(angle) * sin_theta + b * sin(angle) * cos_theta
                    new_points.append(Vector((x, y)))
                return new_points
            else:
                new_points = []
                for p in points_2d:
                    vec = p - center
                    angle = atan2(vec.y, vec.x)
                    x = center.x + a * cos(angle) * cos_theta - b * sin(angle) * sin_theta
                    y = center.y + a * cos(angle) * sin_theta + b * sin(angle) * cos_theta
                    new_points.append(Vector((x, y)))
                return new_points
        
        elif fit_type == 'bspline' and SCIPY_AVAILABLE:
            from scipy import interpolate
            tck = fit_result['tck']
            u = np.linspace(0, 1, n)
            out = interpolate.splev(u, tck)
            return [Vector((out[0][i], out[1][i])) for i in range(n)]
        
        elif fit_type in ['power', 'polynomial']:
            points = fit_result['points']
            if points is not None:
                if space_uniform:
                    x_vals = np.linspace(points[0, 0], points[-1, 0], n)
                    if fit_type == 'power':
                        a, b, c = fit_result['params']
                        y_vals = a * (x_vals ** b) + c
                    else:
                        coeffs = fit_result['coeffs']
                        y_vals = np.polyval(coeffs, x_vals)
                    return [Vector((x_vals[i], y_vals[i])) for i in range(n)]
                else:
                    x_vals = [p.x for p in points_2d]
                    if fit_type == 'power':
                        a, b, c = fit_result['params']
                        y_vals = a * (np.array(x_vals) ** b) + c
                    else:
                        coeffs = fit_result['coeffs']
                        y_vals = np.polyval(coeffs, x_vals)
                    return [Vector((x_vals[i], y_vals[i])) for i in range(n)]
        
        return points_2d

# ============================================================================
# LoopTools Circle Implementation
# ============================================================================

class LoopToolsCircle:
    """
    Extended LoopTools circle implementation with multiple fitting algorithms.
    """
    
    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.curve_fitter = CurveFitter(config)
    
    def space_points(self, points: List[Vector]) -> List[Vector]:
        """
        Uniformly space points using arc length parameterization.
        
        Args:
            points: List of 3D points
            
        Returns:
            Uniformly spaced points
        """
        if len(points) < 2:
            return points
        
        distances = [0.0]
        for i in range(1, len(points)):
            distances.append(distances[-1] + (points[i] - points[i-1]).length)
        
        total_length = distances[-1]
        if total_length < 1e-6:
            return points
        
        target_distances = [i * total_length / (len(points) - 1) for i in range(len(points))]
        
        new_points = []
        j = 0
        for target in target_distances:
            while j < len(distances) - 1 and distances[j+1] < target:
                j += 1
            
            if j >= len(distances) - 1:
                new_points.append(points[-1])
            else:
                t = (target - distances[j]) / (distances[j+1] - distances[j])
                p = points[j] + (points[j+1] - points[j]) * t
                new_points.append(p)
        
        return new_points
    
    def make_circle(self, bm: bmesh.types.BMesh, edge_loop: List[bmesh.types.BMEdge],
                    fit_method: str = 'auto', space_uniform: bool = True,
                    flatten: bool = False) -> Tuple[int, int]:
        """
        Convert selected edge loop to fitted curve.
        
        Args:
            bm: BMesh object
            edge_loop: Selected edge loop
            fit_method: Fitting method (auto, circle, ellipse, bspline, power, polynomial)
            space_uniform: Whether to use uniform spacing
            flatten: Whether to flatten to plane
            
        Returns:
            Tuple of (corrected_count, skipped_count)
        """
        vertices = self._get_ordered_vertices(edge_loop)
        if len(vertices) < 3:
            return 0, len(vertices) if vertices else 0
        
        points_3d = [v.co.copy() for v in vertices]
        
        if flatten:
            plane_normal = self._calculate_average_normal(vertices)
            plane_center = sum(points_3d, Vector((0, 0, 0))) / len(points_3d)
            points_2d = self._project_to_plane(points_3d, plane_normal, plane_center)
        else:
            points_2d = [Vector((p.x, p.y)) for p in points_3d]
        
        normal_indices, outlier_indices = self.curve_fitter.detect_normal_points(points_2d)
        
        if fit_method == 'auto':
            normal_points = [points_2d[i] for i in normal_indices] if normal_indices else points_2d
            fit_result = self.curve_fitter.auto_detect_best_fit(normal_points)
        elif fit_method == 'circle':
            fit_result = self.curve_fitter.fit_circle(points_2d)
        elif fit_method == 'ellipse':
            fit_result = self.curve_fitter.fit_ellipse(points_2d)
        elif fit_method == 'bspline':
            fit_result = self.curve_fitter.fit_bspline(points_2d)
        elif fit_method == 'power':
            fit_result = self.curve_fitter.fit_power(points_2d)
        elif fit_method == 'polynomial':
            fit_result = self.curve_fitter.fit_polynomial(points_2d)
        else:
            fit_result = self.curve_fitter.auto_detect_best_fit(points_2d)
        
        if fit_result is None:
            return 0, len(vertices)
        
        projected_2d = self.curve_fitter.evaluate_fit(fit_result, points_2d, space_uniform)
        
        corrected_count = 0
        skipped_count = 0
        
        for i, vert in enumerate(vertices):
            if i < len(projected_2d):
                if flatten:
                    vert.co.x = projected_2d[i].x
                    vert.co.y = projected_2d[i].y
                else:
                    vert.co.x = projected_2d[i].x
                    vert.co.y = projected_2d[i].y
                
                if i in outlier_indices:
                    corrected_count += 1
                else:
                    skipped_count += 1
        
        return corrected_count, skipped_count
    
    def _get_ordered_vertices(self, edge_loop: List[bmesh.types.BMEdge]) -> List[bmesh.types.BMVert]:
        """Get ordered vertices from edge loop."""
        if not edge_loop:
            return []
        
        vert_edges = defaultdict(list)
        for edge in edge_loop:
            v1, v2 = edge.verts
            vert_edges[v1].append(v2)
            vert_edges[v2].append(v1)
        
        end_verts = [v for v, neighbors in vert_edges.items() if len(neighbors) == 1]
        start_vert = end_verts[0] if end_verts else next(iter(vert_edges.keys()))
        
        ordered_verts = [start_vert]
        current_vert = start_vert
        prev_vert = None
        
        while True:
            neighbors = vert_edges[current_vert]
            next_vert = None
            for neighbor in neighbors:
                if neighbor != prev_vert:
                    next_vert = neighbor
                    break
            
            if next_vert is None:
                break
            
            ordered_verts.append(next_vert)
            prev_vert, current_vert = current_vert, next_vert
            
            if current_vert == start_vert or len(ordered_verts) > len(vert_edges):
                break
        
        return ordered_verts
    
    def _calculate_average_normal(self, vertices: List[bmesh.types.BMVert]) -> Vector:
        """Calculate average normal for flattening."""
        if len(vertices) < 3:
            return Vector((0, 0, 1))
        
        points = np.array([v.co for v in vertices])
        center = np.mean(points, axis=0)
        centered = points - center
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        normal = Vector(eigenvecs[:, np.argmin(eigenvals)])
        
        return normal.normalized()
    
    def _project_to_plane(self, points: List[Vector], normal: Vector,
                          center: Vector) -> List[Vector]:
        """Project points to plane."""
        projected = []
        for p in points:
            vec = p - center
            dist = vec.dot(normal)
            proj = p - normal * dist
            projected.append(Vector((proj.x, proj.y)))
        return projected

# ============================================================================
# Advanced Continuity Analyzer (with scipy)
# ============================================================================

if SCIPY_AVAILABLE:
    class ContinuityAnalyzerAdvanced(ContinuityAnalyzerBase):
        """Advanced continuity analyzer using scipy."""
        
        def analyze_vertex(self, bm: bmesh.types.BMesh,
                           vertex: bmesh.types.BMVert) -> Dict[str, float]:
            """High-precision continuity analysis using scipy."""
            neighbors = []
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                neighbors.append(other)
                for e2 in other.link_edges:
                    n2 = e2.other_vert(other)
                    if n2 != vertex and n2 not in neighbors:
                        neighbors.append(n2)
            
            if len(neighbors) < 5:
                return super().analyze_vertex(bm, vertex)
            
            points = np.array([vertex.co] + [n.co for n in neighbors[:12]])
            
            try:
                from scipy import interpolate
                t = np.linspace(0, 1, len(points))
                tck, u = interpolate.splprep(points.T, s=0, k=min(3, len(points)-1))
                
                derivatives = []
                for order in range(5):
                    try:
                        der = interpolate.spalde(0.5, tck)
                        if len(der) > order:
                            derivatives.append(np.array(der[order]))
                        else:
                            derivatives.append(np.zeros(3))
                    except:
                        derivatives.append(np.zeros(3))
                
                continuity = {
                    'G0': float(min(np.linalg.norm(derivatives[0]) / 100, 1.0)),
                    'G1': float(min(np.linalg.norm(derivatives[1]) / 100, 1.0)) if np.linalg.norm(derivatives[1]) > 1e-6 else 1.0,
                    'G2': float(min(np.linalg.norm(derivatives[2]) / 10, 1.0)),
                    'G3': float(min(np.linalg.norm(derivatives[3]) / 5, 1.0)),
                    'G4': float(min(np.linalg.norm(derivatives[4]) / 2, 1.0))
                }
                
                return continuity
                
            except Exception as e:
                logger.debug(f"Advanced continuity analysis failed: {e}")
                return super().analyze_vertex(bm, vertex)
    
    ContinuityAnalyzer = ContinuityAnalyzerAdvanced
else:
    ContinuityAnalyzer = ContinuityAnalyzerBase

# ============================================================================
# Feature Detector
# ============================================================================

class FeatureDetector:
    """Feature detection for edges and loops."""
    
    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
    
    def detect_edges(self, bm: bmesh.types.BMesh) -> Dict[str, List[bmesh.types.BMEdge]]:
        """Detect feature edges."""
        sharp_edges = []
        boundary_edges = []
        
        for edge in bm.edges:
            if len(edge.link_faces) == 1:
                boundary_edges.append(edge)
                continue
            
            if len(edge.link_faces) == 2:
                angle = self._edge_angle(edge)
                if angle > self.config.feature_angle:
                    sharp_edges.append(edge)
        
        return {
            'sharp': sharp_edges,
            'boundary': boundary_edges
        }
    
    def detect_loops(self, bm: bmesh.types.BMesh,
                     edges: List[bmesh.types.BMEdge]) -> List[List[bmesh.types.BMEdge]]:
        """Detect edge loops with ordered vertices."""
        visited = set()
        loops = []
        
        for edge in edges:
            if edge.index in visited:
                continue
            
            loop_edges, loop_verts = self._trace_loop_with_order(edge, visited)
            if len(loop_edges) > 2 and len(loop_verts) > 2:
                loops.append(loop_edges)
        
        return loops
    
    def _trace_loop_with_order(self, start_edge: bmesh.types.BMEdge,
                                visited: Set[int]) -> Tuple[List[bmesh.types.BMEdge],
                                                            List[bmesh.types.BMVert]]:
        """Trace edge loop while maintaining vertex order."""
        loop_edges = []
        loop_verts = []
        
        current_edge = start_edge
        prev_vert = current_edge.verts[0]
        curr_vert = current_edge.verts[1]
        
        loop_edges.append(current_edge)
        loop_verts.append(prev_vert)
        loop_verts.append(curr_vert)
        visited.add(current_edge.index)
        
        while True:
            next_edge = None
            next_vert = None
            
            for e in curr_vert.link_edges:
                if e.index in visited:
                    continue
                if not e.select:
                    continue
                
                other = e.other_vert(curr_vert)
                if other == prev_vert:
                    continue
                
                next_edge = e
                next_vert = other
                break
            
            if next_edge is None:
                if curr_vert == loop_verts[0]:
                    break
                if len(loop_edges) > 1:
                    loop_edges = [start_edge]
                    loop_verts = [current_edge.verts[1], current_edge.verts[0]]
                    prev_vert, curr_vert = loop_verts[0], loop_verts[1]
                    continue
                break
            
            loop_edges.append(next_edge)
            loop_verts.append(next_vert)
            visited.add(next_edge.index)
            
            prev_vert, curr_vert = curr_vert, next_vert
            
            if len(loop_edges) > 1000:
                break
        
        if len(loop_verts) > 0 and loop_verts[0] == loop_verts[-1]:
            loop_verts.pop()
        
        return loop_edges, loop_verts
    
    def detect_regions(self, bm: bmesh.types.BMesh,
                       faces: List[bmesh.types.BMFace]) -> List[List[bmesh.types.BMFace]]:
        """Detect continuous regions."""
        visited = set()
        regions = []
        
        for face in faces:
            if face.index in visited:
                continue
            
            region = self._flood_fill(face, visited)
            if len(region) > 0:
                regions.append(region)
        
        return regions
    
    def _edge_angle(self, edge: bmesh.types.BMEdge) -> float:
        """Calculate edge dihedral angle."""
        if len(edge.link_faces) != 2:
            return 0
        
        f1, f2 = edge.link_faces
        angle = f1.normal.angle(f2.normal)
        return degrees(angle)
    
    def _flood_fill(self, start_face: bmesh.types.BMFace,
                    visited: Set[int]) -> List[bmesh.types.BMFace]:
        """Flood fill region."""
        region = []
        stack = [start_face]
        
        while stack:
            face = stack.pop()
            if face.index in visited:
                continue
            
            region.append(face)
            visited.add(face.index)
            
            for edge in face.edges:
                for linked_face in edge.link_faces:
                    if linked_face != face and linked_face.select:
                        stack.append(linked_face)
        
        return region

# ============================================================================
# Mesh Optimizer
# ============================================================================

class MeshOptimizer:
    """Main mesh optimization class."""
    
    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.continuity_analyzer = ContinuityAnalyzer(config)
        self.looptools_circle = LoopToolsCircle(config)
    
    def optimize_vertex(self, bm: bmesh.types.BMesh, vertex: bmesh.types.BMVert,
                        continuity: Dict[str, float]) -> bool:
        """Optimize single vertex based on continuity."""
        if not vertex.link_edges:
            return False
        
        g0_value = continuity.get('G0', 1.0)
        g1_value = continuity.get('G1', 1.0)
        
        if g0_value < self.config.G0_threshold and g1_value < self.config.G1_threshold:
            return self._laplacian_smooth(vertex)
        else:
            return self._umbrella_smooth(vertex)
    
    def _laplacian_smooth(self, vertex: bmesh.types.BMVert) -> bool:
        """Laplacian smoothing with inverse distance weighting."""
        neighbors = [e.other_vert(vertex) for e in vertex.link_edges]
        if not neighbors:
            return False
        
        total_weight = 0
        weighted_sum = Vector((0, 0, 0))
        
        for neighbor in neighbors:
            dist = (vertex.co - neighbor.co).length
            if dist > 1e-6:
                weight = 1.0 / dist
                weighted_sum += neighbor.co * weight
                total_weight += weight
        
        if total_weight > 1e-6:
            target = weighted_sum / total_weight
            vertex.co = vertex.co.lerp(target, self.config.smooth_factor)
            return True
        
        return False
    
    def _umbrella_smooth(self, vertex: bmesh.types.BMVert) -> bool:
        """Simple umbrella smoothing."""
        neighbors = [e.other_vert(vertex) for e in vertex.link_edges]
        if not neighbors:
            return False
        
        avg_pos = sum((n.co for n in neighbors), Vector((0, 0, 0))) / len(neighbors)
        vertex.co = vertex.co.lerp(avg_pos, self.config.smooth_factor)
        return True
    
    def optimize_edge_loop(self, bm: bmesh.types.BMesh,
                           edge_loop: List[bmesh.types.BMEdge]) -> Tuple[int, int]:
        """Optimize edge loop using fitting algorithms."""
        return self.looptools_circle.make_circle(
            bm,
            edge_loop,
            fit_method=self.config.fit_method,
            space_uniform=self.config.space_uniform,
            flatten=self.config.flatten
        )
    
    def optimize_region(self, bm: bmesh.types.BMesh,
                        faces: List[bmesh.types.BMFace]) -> None:
        """Optimize region using iterative smoothing."""
        vertices = set()
        for face in faces:
            for vert in face.verts:
                vertices.add(vert)
        
        vertices = list(vertices)
        if len(vertices) < 4:
            return
        
        self._iterative_smooth(vertices)
    
    def _iterative_smooth(self, vertices: List[bmesh.types.BMVert]) -> None:
        """Iterative smoothing with convergence check."""
        for iteration in range(self.config.max_iterations):
            new_positions = []
            max_diff = 0
            
            for vert in vertices:
                neighbors = [e.other_vert(vert) for e in vert.link_edges]
                if neighbors:
                    avg_pos = sum((n.co for n in neighbors), Vector((0, 0, 0))) / len(neighbors)
                    alpha = 1.0 / (iteration + 2)
                    new_pos = vert.co.lerp(avg_pos, alpha)
                    new_positions.append(new_pos)
                    
                    diff = (vert.co - new_pos).length
                    max_diff = max(max_diff, diff)
                else:
                    new_positions.append(vert.co)
            
            for i, vert in enumerate(vertices):
                vert.co = new_positions[i]
            
            if max_diff < self.config.convergence_threshold:
                break
    
    def repair_corner_vertices(self, bm: bmesh.types.BMesh,
                               edge_loop: List[bmesh.types.BMEdge],
                               fit_method: str = 'auto',
                               space_uniform: bool = True,
                               flatten: bool = False) -> Tuple[int, int]:
        """
        Repair corner defects using curve fitting.
        
        Args:
            bm: BMesh object
            edge_loop: Selected edge loop
            fit_method: Fitting method
            space_uniform: Whether to use uniform spacing
            flatten: Whether to flatten to plane
            
        Returns:
            Tuple of (corrected_count, skipped_count)
        """
        corrected_count, skipped_count = self.looptools_circle.make_circle(
            bm, edge_loop, fit_method, space_uniform, flatten
        )
        return corrected_count, skipped_count

# ============================================================================
# Operators
# ============================================================================

class MESH_OT_analyze_continuity(bpy.types.Operator):
    """Analyze continuity of selected vertices"""
    bl_idname = "mesh.analyze_continuity"
    bl_label = "Analyze Continuity"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.mode == 'EDIT_MESH')
    
    def execute(self, context: bpy.types.Context) -> Set[str]:
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v for v in bm.verts if v.select]
        
        if not selected_verts:
            self.report({'WARNING'}, get_lang()['select_vertices'])
            return {'CANCELLED'}
        
        config = OptimizerConfig()
        config.corner_angle_threshold = context.scene.topology_corner_angle_threshold
        config.corner_curvature_threshold = context.scene.topology_corner_curvature_threshold
        
        analyzer = ContinuityAnalyzer(config)
        
        lang = get_lang()
        report_lines = ["Continuity Analysis Results:"]
        for vert in selected_verts[:10]:
            cont = analyzer.analyze_vertex(bm, vert)
            is_corner, angle = analyzer.detect_corner_vertex(bm, vert)
            corner_str = lang['corner_defect'] if is_corner else lang['normal_vertex']
            report_lines.append(f"\nVertex {vert.index}: {corner_str} (angle: {angle:.1f}°)")
            for key in ['G0', 'G1', 'G2', 'G3', 'G4']:
                report_lines.append(f"  {key}: {cont.get(key, 0):.6f}")
        
        if len(selected_verts) > 10:
            report_lines.append(f"\n{lang['more_vertices'].format(len(selected_verts) - 10)}")
        
        self.report({'INFO'}, "\n".join(report_lines))
        return {'FINISHED'}


class MESH_OT_optimize_vertex(bpy.types.Operator):
    """Optimize selected vertices"""
    bl_idname = "mesh.optimize_vertex"
    bl_label = "Optimize Vertices"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.mode == 'EDIT_MESH')
    
    def execute(self, context: bpy.types.Context) -> Set[str]:
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v for v in bm.verts if v.select]
        
        if not selected_verts:
            self.report({'WARNING'}, get_lang()['select_vertices'])
            return {'CANCELLED'}
        
        config = OptimizerConfig()
        config.smooth_factor = context.scene.topology_smooth
        
        analyzer = ContinuityAnalyzer(config)
        optimizer = MeshOptimizer(config)
        
        optimized_count = 0
        for vert in selected_verts:
            continuity = analyzer.analyze_vertex(bm, vert)
            if optimizer.optimize_vertex(bm, vert, continuity):
                optimized_count += 1
        
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, get_lang()['optimized_count'].format(optimized_count))
        return {'FINISHED'}


class MESH_OT_optimize_region(bpy.types.Operator):
    """Optimize selected region"""
    bl_idname = "mesh.optimize_region"
    bl_label = "Optimize Region"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.mode == 'EDIT_MESH')
    
    def execute(self, context: bpy.types.Context) -> Set[str]:
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        selected_faces = [f for f in bm.faces if f.select]
        
        if not selected_faces:
            self.report({'WARNING'}, get_lang()['select_faces'])
            return {'CANCELLED'}
        
        config = OptimizerConfig()
        config.max_iterations = context.scene.topology_iterations
        config.smooth_factor = context.scene.topology_smooth
        
        optimizer = MeshOptimizer(config)
        optimizer.optimize_region(bm, selected_faces)
        
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, get_lang()['region_complete'])
        return {'FINISHED'}


class MESH_OT_fit_curve(bpy.types.Operator):
    """
    Repair corner defects using curve fitting
    Supports multiple fitting algorithms
    """
    bl_idname = "mesh.fit_curve"
    bl_label = "Curve Fit Repair"
    bl_options = {'REGISTER', 'UNDO'}
    
    fit_method: bpy.props.EnumProperty(
        name="Fitting Method",
        items=[
            ('auto', "Auto Detect", "Automatically choose best fitting method"),
            ('circle', "Circle", "Fit to standard circle"),
            ('ellipse', "Ellipse", "Fit to ellipse"),
            ('bspline', "B-Spline", "B-spline curve fitting (preserves details)"),
            ('power', "Power Function", "Power function fitting y = a*x^b + c"),
            ('polynomial', "Polynomial", "Polynomial fitting")
        ],
        default='auto'
    )
    
    space_uniform: bpy.props.BoolProperty(
        name="Uniform Spacing",
        default=True,
        description="Use uniform spacing along curve"
    )
    
    flatten: bpy.props.BoolProperty(
        name="Flatten to Plane",
        default=False,
        description="Project vertices to plane before fitting"
    )
    
    outlier_threshold: bpy.props.FloatProperty(
        name="Outlier Threshold",
        default=2.0,
        min=1.0,
        max=5.0,
        description="Outlier detection threshold (std dev multiplier)"
    )
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.mode == 'EDIT_MESH')
    
    def execute(self, context: bpy.types.Context) -> Set[str]:
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        selected_edges = [e for e in bm.edges if e.select]
        
        if not selected_edges:
            self.report({'WARNING'}, get_lang()['select_edges'])
            return {'CANCELLED'}
        
        config = OptimizerConfig()
        config.fit_method = self.fit_method
        config.space_uniform = self.space_uniform
        config.flatten = self.flatten
        config.outlier_threshold = self.outlier_threshold
        
        detector = FeatureDetector(config)
        optimizer = MeshOptimizer(config)
        
        loops = detector.detect_loops(bm, selected_edges)
        
        if not loops:
            self.report({'WARNING'}, get_lang()['no_loops'])
            return {'CANCELLED'}
        
        total_corrected = 0
        total_skipped = 0
        
        for loop in loops:
            corrected_count, skipped_count = optimizer.repair_corner_vertices(
                bm, loop,
                fit_method=self.fit_method,
                space_uniform=self.space_uniform,
                flatten=self.flatten
            )
            
            total_corrected += corrected_count
            total_skipped += skipped_count
            bmesh.update_edit_mesh(obj.data)
        
        lang = get_lang()
        method_names = {
            'auto': lang['auto'],
            'circle': lang['circle'],
            'ellipse': lang['ellipse'],
            'bspline': lang['bspline'],
            'power': lang['power'],
            'polynomial': lang['polynomial']
        }
        
        self.report({'INFO'}, lang['fit_complete'].format(
            method_names[self.fit_method],
            total_corrected,
            total_skipped
        ))
        
        return {'FINISHED'}


# ============================================================================
# UI Panels
# ============================================================================

class VIEW3D_PT_topology_pro(bpy.types.Panel):
    """Main E Topology Smooth Panel"""
    bl_label = "E Topology Smooth"
    bl_idname = "VIEW3D_PT_topology_pro"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ETS"
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.mode == 'EDIT_MESH')
    
    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        lang = get_lang()
        
        # Status box
        box = layout.box()
        if SCIPY_AVAILABLE:
            box.label(text=f"✅ {lang['scipy_ready']} {SCIPY_VERSION}", icon='CHECKMARK')
        else:
            box.label(text=f"⚠️ {lang['scipy_missing']}", icon='ERROR')
            box.label(text=f"ℹ️ {lang['scipy_fallback']}")
        
        # Analysis tools
        box = layout.box()
        box.label(text=lang['analysis'], icon='VIEWZOOM')
        box.operator("mesh.analyze_continuity", text=lang['analyze'], icon='GRAPH')
        
        # Vertex optimization
        box = layout.box()
        box.label(text=lang['vertex_opt'], icon='VERTEXSEL')
        row = box.row(align=True)
        row.prop(context.scene, "topology_smooth", text=lang['smooth_strength'])
        box.operator("mesh.optimize_vertex", text=lang['optimize_vertex'])
        
        # Face optimization
        box = layout.box()
        box.label(text=lang['face_opt'], icon='FACESEL')
        row = box.row(align=True)
        row.prop(context.scene, "topology_iterations", text=lang['iterations'])
        box.operator("mesh.optimize_region", text=lang['optimize_region'])
        
        # Curve fitting (core feature)
        box = layout.box()
        box.label(text=lang['curve_fit'], icon='CURVE_DATA')
        
        row = box.row(align=True)
        row.prop(context.scene, "topology_fit_method", text=lang['fit_method'])
        
        row = box.row(align=True)
        row.prop(context.scene, "topology_space_uniform", text=lang['uniform_spacing'])
        
        row = box.row(align=True)
        row.prop(context.scene, "topology_flatten", text=lang['flatten'])
        row.prop(context.scene, "topology_outlier_threshold", text=lang['outlier'])
        
        fit_op = box.operator("mesh.fit_curve", text=lang['fit_curve'], icon='MOD_CURVE')
        fit_op.fit_method = context.scene.topology_fit_method
        fit_op.space_uniform = context.scene.topology_space_uniform
        fit_op.flatten = context.scene.topology_flatten
        fit_op.outlier_threshold = context.scene.topology_outlier_threshold
        
        # Corner detection settings
        box = layout.box()
        box.label(text=lang['corner_detect'], icon='SETTINGS')
        row = box.row(align=True)
        row.prop(context.scene, "topology_corner_angle_threshold", text=lang['angle'])
        row.prop(context.scene, "topology_corner_curvature_threshold", text=lang['curvature'])


class VIEW3D_PT_topology_pro_settings(bpy.types.Panel):
    """Advanced Settings Panel"""
    bl_label = "Advanced Settings"
    bl_idname = "VIEW3D_PT_topology_pro_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ETS"
    bl_parent_id = "VIEW3D_PT_topology_pro"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        lang = get_lang()
        
        # Continuity thresholds
        box = layout.box()
        box.label(text=lang['advanced'], icon='GRAPH')
        box.prop(context.scene, "topology_G0_threshold", text="G0")
        box.prop(context.scene, "topology_G1_threshold", text="G1")
        box.prop(context.scene, "topology_G2_threshold", text="G2")
        if SCIPY_AVAILABLE:
            box.prop(context.scene, "topology_G3_threshold", text="G3")
            box.prop(context.scene, "topology_G4_threshold", text="G4")
        
        # Feature settings
        box = layout.box()
        box.label(text="Feature Detection", icon='EDGESEL')
        box.prop(context.scene, "topology_feature_angle", text="Feature Angle")
        
        # Performance settings
        if SCIPY_AVAILABLE:
            box = layout.box()
            box.label(text="Performance", icon='SETTINGS')
            box.prop(context.scene, "topology_multithreading", text="Multi-threading")
            box.prop(context.scene, "topology_batch_size", text="Batch Size")


# ============================================================================
# Properties Registration
# ============================================================================

def register_properties() -> None:
    """Register scene properties."""
    bpy.types.Scene.topology_iterations = bpy.props.IntProperty(
        name="Iterations",
        default=5,
        min=1,
        max=50,
        description="Number of optimization iterations"
    )
    
    bpy.types.Scene.topology_smooth = bpy.props.FloatProperty(
        name="Smooth Factor",
        default=0.5,
        min=0.0,
        max=1.0,
        precision=3,
        description="Smoothing strength"
    )
    
    bpy.types.Scene.topology_feature_angle = bpy.props.FloatProperty(
        name="Feature Angle",
        default=45.0,
        min=0.0,
        max=180.0,
        description="Feature edge detection angle threshold"
    )
    
    bpy.types.Scene.topology_G0_threshold = bpy.props.FloatProperty(
        name="G0 Threshold",
        default=0.001,
        min=0.0001,
        max=1.0,
        precision=4,
        description="G0 continuity threshold"
    )
    
    bpy.types.Scene.topology_G1_threshold = bpy.props.FloatProperty(
        name="G1 Threshold",
        default=0.01,
        min=0.001,
        max=1.0,
        precision=3,
        description="G1 continuity threshold"
    )
    
    bpy.types.Scene.topology_G2_threshold = bpy.props.FloatProperty(
        name="G2 Threshold",
        default=0.1,
        min=0.01,
        max=1.0,
        precision=3,
        description="G2 continuity threshold"
    )
    
    bpy.types.Scene.topology_G3_threshold = bpy.props.FloatProperty(
        name="G3 Threshold",
        default=0.5,
        min=0.1,
        max=2.0,
        precision=3,
        description="G3 continuity threshold"
    )
    
    bpy.types.Scene.topology_G4_threshold = bpy.props.FloatProperty(
        name="G4 Threshold",
        default=1.0,
        min=0.1,
        max=5.0,
        precision=3,
        description="G4 continuity threshold"
    )
    
    bpy.types.Scene.topology_multithreading = bpy.props.BoolProperty(
        name="Multi-threading",
        default=True,
        description="Enable multi-threading"
    )
    
    bpy.types.Scene.topology_batch_size = bpy.props.IntProperty(
        name="Batch Size",
        default=1000,
        min=100,
        max=10000,
        description="Number of vertices per batch"
    )
    
    bpy.types.Scene.topology_fit_method = bpy.props.EnumProperty(
        name="Fitting Method",
        items=[
            ('auto', "Auto Detect", "Automatically choose best fitting method"),
            ('circle', "Circle", "Fit to standard circle"),
            ('ellipse', "Ellipse", "Fit to ellipse"),
            ('bspline', "B-Spline", "B-spline curve fitting (preserves details)"),
            ('power', "Power Function", "Power function fitting y = a*x^b + c"),
            ('polynomial', "Polynomial", "Polynomial fitting")
        ],
        default='auto'
    )
    
    bpy.types.Scene.topology_space_uniform = bpy.props.BoolProperty(
        name="Uniform Spacing",
        default=True,
        description="Use uniform spacing along curve"
    )
    
    bpy.types.Scene.topology_flatten = bpy.props.BoolProperty(
        name="Flatten",
        default=False,
        description="Project vertices to plane before fitting"
    )
    
    bpy.types.Scene.topology_outlier_threshold = bpy.props.FloatProperty(
        name="Outlier Threshold",
        default=2.0,
        min=1.0,
        max=5.0,
        precision=1,
        description="Outlier detection threshold (std dev multiplier)"
    )
    
    bpy.types.Scene.topology_corner_angle_threshold = bpy.props.FloatProperty(
        name="Corner Angle",
        default=150.0,
        min=90.0,
        max=179.0,
        precision=1,
        description="Angle threshold for corner detection (degrees)"
    )
    
    bpy.types.Scene.topology_corner_curvature_threshold = bpy.props.FloatProperty(
        name="Corner Curvature",
        default=0.8,
        min=0.1,
        max=1.0,
        precision=2,
        description="Curvature threshold for corner detection"
    )


def unregister_properties() -> None:
    """Unregister scene properties."""
    del bpy.types.Scene.topology_iterations
    del bpy.types.Scene.topology_smooth
    del bpy.types.Scene.topology_feature_angle
    del bpy.types.Scene.topology_G0_threshold
    del bpy.types.Scene.topology_G1_threshold
    del bpy.types.Scene.topology_G2_threshold
    del bpy.types.Scene.topology_G3_threshold
    del bpy.types.Scene.topology_G4_threshold
    del bpy.types.Scene.topology_multithreading
    del bpy.types.Scene.topology_batch_size
    del bpy.types.Scene.topology_fit_method
    del bpy.types.Scene.topology_space_uniform
    del bpy.types.Scene.topology_flatten
    del bpy.types.Scene.topology_outlier_threshold
    del bpy.types.Scene.topology_corner_angle_threshold
    del bpy.types.Scene.topology_corner_curvature_threshold


# ============================================================================
# Registration
# ============================================================================

classes = [
    MESH_OT_analyze_continuity,
    MESH_OT_optimize_vertex,
    MESH_OT_optimize_region,
    MESH_OT_fit_curve,
    VIEW3D_PT_topology_pro,
    VIEW3D_PT_topology_pro_settings,
]


def register() -> None:
    """Register all classes and properties."""
    register_properties()
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            logger.error(f"Failed to register {cls.__name__}: {e}")
    
    logger.info("E Topology Smooth registered successfully")


def unregister() -> None:
    """Unregister all classes and properties."""
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            logger.error(f"Failed to unregister {cls.__name__}: {e}")
    
    unregister_properties()
    logger.info("E Topology Smooth unregistered successfully")


if __name__ == "__main__":
    register()