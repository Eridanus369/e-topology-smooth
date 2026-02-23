# Changelog / 更新日志

All notable changes to the E Topology Smooth project will be documented in this file.
本文件记录了 E Topology Smooth 项目的所有重要变更。

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.7.0] - 2026-02-23

### Changed / 变更
- **Dependency management** / **依赖管理**
  - Removed bundled numpy wheels / 移除了打包的 numpy wheel 文件
  - Now exclusively uses Blender's built-in numpy (per VFX Reference Platform) / 现在完全使用 Blender 内置的 numpy（符合 VFX 参考平台要求）
  - Retained scipy wheels as they are not included with Blender / 保留了 scipy wheels，因为 Blender 不包含它

- **Logging system** / **日志系统**
  - Disabled logging by default to reduce memory usage and unnecessary output / 默认禁用日志以减少内存占用和不必要的输出
  - Added `enable_logging()` function for manual activation when debugging / 添加了 `enable_logging()` 函数供调试时手动激活
  - Uses NullHandler by default to prevent any log output / 默认使用 NullHandler 防止任何日志输出

- **Build configuration** / **构建配置**
  - Updated blender_manifest.toml to remove numpy wheels / 更新了 blender_manifest.toml 以移除 numpy wheels
  - Ready for platform-specific builds using `--split-platforms` / 已准备好使用 `--split-platforms` 进行平台特定构建

---

## [2.6.1] - 2026-02-20

### Added / 新增
- **Multi-language support** / **多语言支持**
  - English and Chinese UI / 中英文双语界面
  - Automatic language detection based on Blender preferences / 自动检测Blender语言设置
  - Complete translation of all UI elements and messages / 所有界面元素和提示信息完整翻译

- **Comprehensive error handling** / **完善的错误处理**
  - Logging system for debugging / 日志记录系统便于调试
  - Graceful fallback when scipy is unavailable / scipy不可用时的优雅降级
  - Better exception handling in all operators / 所有操作符中更好的异常处理

- **Documentation** / **文档**
  - Detailed README with bilingual support / 详细的README双语说明
  - Full GPL-3.0 license file / 完整的GPL-3.0许可证文件
  - Version history tracking / 版本历史记录

### Changed / 变更
- **Code structure** / **代码结构**
  - Reorganized imports for better maintainability / 重新组织导入以提高可维护性
  - Added type hints throughout the codebase / 添加了完整的类型提示
  - Improved function documentation strings / 改进函数文档字符串

- **UI/UX** / **用户界面**
  - Redesigned panel layout with better organization / 重新设计的面板布局，更清晰的组织
  - Added status indicators for scipy availability / 添加scipy可用性状态指示
  - Improved parameter labels and tooltips / 改进参数标签和提示信息

### Fixed / 修复
- **Property registration** / **属性注册**
  - Fixed AttributeError for topology_fit_method / 修复topology_fit_method的属性错误
  - Ensured all scene properties are properly registered/unregistered / 确保所有场景属性正确注册/注销
  - Added missing property cleanup on unregister / 添加注销时缺失的属性清理

- **Panel display** / **面板显示**
  - Fixed advanced settings panel inheritance / 修复高级设置面板的继承关系
  - Ensured all UI elements display correctly / 确保所有UI元素正确显示
  - Added proper polling for edit mode / 添加编辑模式的正确轮询

### Removed / 移除
- **Automatic dependency installation** / **自动依赖安装**
  - Removed all scipy installation operators / 移除所有scipy安装操作符
  - Removed dependency checking UI / 移除依赖检查界面
  - Now using bundled wheels exclusively / 现在完全使用打包的wheel文件

---

## [2.6.0] - 2026-02-01

### Added / 新增
- Initial release with core functionality / 包含核心功能的初始版本
- G0-G4 continuity analysis / G0-G4连续性分析
- Multi-algorithm curve fitting / 多算法曲线拟合
  - Circle fitting (Kasa method) / 圆形拟合
  - Ellipse fitting (direct least squares) / 椭圆拟合
  - B-spline fitting / B样条拟合
  - Power function fitting / 幂函数拟合
  - Polynomial fitting / 多项式拟合
- Corner defect detection / 尖角瑕疵检测
- Outlier detection algorithm / 异常点检测算法
- Uniform spacing using arc length parameterization / 基于弧长参数化的均匀间距
- Flatten to plane functionality / 压平到平面功能

### Technical / 技术细节
- NumPy-based calculations for core algorithms / 基于NumPy的核心算法计算
- Scipy integration for advanced features / 集成Scipy实现高级功能
- Multi-threading support for performance / 多线程支持以提升性能
- Batch processing for large meshes / 大型网格的批处理

---

## Future Plans / 未来计划

### [2.7.0] - Planned / 计划中
- [ ] Unit tests for core algorithms / 核心算法的单元测试
- [ ] Performance benchmarks / 性能基准测试
- [ ] Real-time preview for curve fitting / 曲线拟合的实时预览
- [ ] Custom fitting parameter presets / 自定义拟合参数预设
- [ ] More fitting algorithms (Bezier, NURBS) / 更多拟合算法

### [3.0.0] - Roadmap / 路线图
- [ ] GPU acceleration using OpenCL / 使用OpenCL的GPU加速
- [ ] Interactive fitting with handles / 带手柄的交互式拟合
- [ ] Integration with Blender's modifier stack / 集成到Blender的修改器堆栈
- [ ] Support for vertex groups / 支持顶点组
- [ ] Support for shape keys / 支持形态键

---

## Version History / 版本历史

| Version | Date | Blender | Python | Status |
|---------|------|---------|--------|--------|
| 2.7.0 | 2026-02-23 | 4.5+ | 3.11 | ✅ Current |
| 2.6.1 | 2026-02-20 | 4.5+ | 3.11 | ⏳ Legacy |
| 2.6.0 | 2026-02-01 | 4.5+ | 3.11 | ⏳ Legacy |

---

## How to Update / 如何更新

1. Download the latest version from [Releases](https://github.com/Eridanus369/e-topology-smooth/releases)
2. In Blender, go to **Edit > Preferences > Add-ons**
3. Disable the old version
4. Click **Install...** and select the new ZIP file
5. Enable the new version

---

## Reporting Issues / 报告问题

If you encounter any problems, please report them on [GitHub Issues](https://github.com/Eridanus369/e-topology-smooth/issues) with:

- Blender version / Blender版本
- Operating system / 操作系统
- Steps to reproduce / 重现步骤
- Error messages (if any) / 错误信息

---

## Contributors / 贡献者

- **Eridanus** - Lead developer / 首席开发者

---

## License / 许可证

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
本项目采用 GPL-3.0 许可证 - 详情参见 [LICENSE](LICENSE) 文件。

---

*For older versions, please check the GitHub releases page.*
*旧版本请查看 GitHub 发布页面。*