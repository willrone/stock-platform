"""
模型兼容性验证器
检查模型依赖、环境兼容性和接口一致性
"""
import hashlib
import importlib
import json
import pickle
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import importlib.metadata
from loguru import logger
from packaging import version


class CompatibilityLevel(Enum):
    """兼容性级别"""

    COMPATIBLE = "compatible"
    WARNING = "warning"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class ValidationCategory(Enum):
    """验证类别"""

    DEPENDENCIES = "dependencies"
    PYTHON_VERSION = "python_version"
    SYSTEM_REQUIREMENTS = "system_requirements"
    MODEL_FORMAT = "model_format"
    API_INTERFACE = "api_interface"
    DATA_SCHEMA = "data_schema"
    PERFORMANCE = "performance"


@dataclass
class ValidationResult:
    """验证结果"""

    category: ValidationCategory
    level: CompatibilityLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "level": self.level.value,
            "message": self.message,
            "details": self.details,
            "suggestions": self.suggestions,
        }


@dataclass
class ModelMetadata:
    """模型元数据"""

    model_id: str
    version: str
    model_type: str
    framework: str
    framework_version: str
    python_version: str
    dependencies: Dict[str, str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    system_requirements: Dict[str, Any]
    created_at: str
    file_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "model_type": self.model_type,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "python_version": self.python_version,
            "dependencies": self.dependencies,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "system_requirements": self.system_requirements,
            "created_at": self.created_at,
            "file_hash": self.file_hash,
        }


class DependencyValidator:
    """依赖验证器"""

    def __init__(self):
        self.installed_packages = self._get_installed_packages()

    def _get_installed_packages(self) -> Dict[str, str]:
        """获取已安装的包及其版本"""
        packages = {}
        try:
            for dist in importlib.metadata.distributions():
                packages[dist.metadata["Name"].lower()] = dist.metadata["Version"]
        except Exception as e:
            logger.warning(f"获取已安装包列表失败: {e}")
        return packages

    def validate_dependencies(
        self, required_deps: Dict[str, str]
    ) -> List[ValidationResult]:
        """验证依赖包"""
        results = []

        for package_name, required_version in required_deps.items():
            package_name_lower = package_name.lower()

            if package_name_lower not in self.installed_packages:
                # 包未安装
                results.append(
                    ValidationResult(
                        category=ValidationCategory.DEPENDENCIES,
                        level=CompatibilityLevel.INCOMPATIBLE,
                        message=f"缺少依赖包: {package_name}",
                        details={
                            "package": package_name,
                            "required_version": required_version,
                            "installed_version": None,
                        },
                        suggestions=[
                            f"安装依赖包: pip install {package_name}=={required_version}"
                        ],
                    )
                )
                continue

            installed_version = self.installed_packages[package_name_lower]

            # 版本兼容性检查
            compatibility = self._check_version_compatibility(
                installed_version, required_version
            )

            if compatibility["compatible"]:
                results.append(
                    ValidationResult(
                        category=ValidationCategory.DEPENDENCIES,
                        level=CompatibilityLevel.COMPATIBLE,
                        message=f"依赖包 {package_name} 版本兼容",
                        details={
                            "package": package_name,
                            "required_version": required_version,
                            "installed_version": installed_version,
                        },
                    )
                )
            elif compatibility["warning"]:
                results.append(
                    ValidationResult(
                        category=ValidationCategory.DEPENDENCIES,
                        level=CompatibilityLevel.WARNING,
                        message=f"依赖包 {package_name} 版本可能不兼容",
                        details={
                            "package": package_name,
                            "required_version": required_version,
                            "installed_version": installed_version,
                            "reason": compatibility["reason"],
                        },
                        suggestions=[f"建议升级到版本 {required_version}"],
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        category=ValidationCategory.DEPENDENCIES,
                        level=CompatibilityLevel.INCOMPATIBLE,
                        message=f"依赖包 {package_name} 版本不兼容",
                        details={
                            "package": package_name,
                            "required_version": required_version,
                            "installed_version": installed_version,
                            "reason": compatibility["reason"],
                        },
                        suggestions=[
                            f"安装正确版本: pip install {package_name}=={required_version}"
                        ],
                    )
                )

        return results

    def _check_version_compatibility(
        self, installed: str, required: str
    ) -> Dict[str, Any]:
        """检查版本兼容性"""
        try:
            installed_ver = version.parse(installed)
            required_ver = version.parse(required)

            # 完全匹配
            if installed_ver == required_ver:
                return {"compatible": True, "warning": False}

            # 主版本不同
            if installed_ver.major != required_ver.major:
                return {
                    "compatible": False,
                    "warning": False,
                    "reason": f"主版本不匹配: {installed_ver.major} vs {required_ver.major}",
                }

            # 次版本兼容性
            if installed_ver.minor < required_ver.minor:
                return {
                    "compatible": False,
                    "warning": True,
                    "reason": f"次版本过低: {installed_ver.minor} < {required_ver.minor}",
                }
            elif installed_ver.minor > required_ver.minor:
                return {
                    "compatible": True,
                    "warning": True,
                    "reason": f"次版本较新: {installed_ver.minor} > {required_ver.minor}",
                }

            # 修订版本兼容性
            if installed_ver.micro < required_ver.micro:
                return {
                    "compatible": True,
                    "warning": True,
                    "reason": f"修订版本略低: {installed_ver.micro} < {required_ver.micro}",
                }

            return {"compatible": True, "warning": False}

        except Exception as e:
            logger.error(f"版本比较失败: {e}")
            return {"compatible": False, "warning": False, "reason": f"版本格式错误: {e}"}


class SystemValidator:
    """系统环境验证器"""

    def __init__(self):
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }

    def validate_python_version(self, required_python: str) -> ValidationResult:
        """验证Python版本"""
        current_python = platform.python_version()

        try:
            current_ver = version.parse(current_python)
            required_ver = version.parse(required_python)

            if current_ver.major != required_ver.major:
                return ValidationResult(
                    category=ValidationCategory.PYTHON_VERSION,
                    level=CompatibilityLevel.INCOMPATIBLE,
                    message=f"Python主版本不兼容: {current_python} vs {required_python}",
                    details={
                        "current_version": current_python,
                        "required_version": required_python,
                    },
                    suggestions=[f"安装Python {required_python}"],
                )

            if current_ver.minor < required_ver.minor:
                return ValidationResult(
                    category=ValidationCategory.PYTHON_VERSION,
                    level=CompatibilityLevel.WARNING,
                    message=f"Python次版本较低: {current_python} < {required_python}",
                    details={
                        "current_version": current_python,
                        "required_version": required_python,
                    },
                    suggestions=[f"建议升级到Python {required_python}"],
                )

            return ValidationResult(
                category=ValidationCategory.PYTHON_VERSION,
                level=CompatibilityLevel.COMPATIBLE,
                message=f"Python版本兼容: {current_python}",
                details={
                    "current_version": current_python,
                    "required_version": required_python,
                },
            )

        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.PYTHON_VERSION,
                level=CompatibilityLevel.UNKNOWN,
                message=f"Python版本检查失败: {e}",
                details={
                    "current_version": current_python,
                    "required_version": required_python,
                    "error": str(e),
                },
            )

    def validate_system_requirements(
        self, requirements: Dict[str, Any]
    ) -> List[ValidationResult]:
        """验证系统需求"""
        results = []

        # 检查操作系统
        if "os" in requirements:
            required_os = requirements["os"]
            current_os = platform.system().lower()

            if isinstance(required_os, str):
                required_os = [required_os.lower()]
            elif isinstance(required_os, list):
                required_os = [os.lower() for os in required_os]

            if current_os in required_os:
                results.append(
                    ValidationResult(
                        category=ValidationCategory.SYSTEM_REQUIREMENTS,
                        level=CompatibilityLevel.COMPATIBLE,
                        message=f"操作系统兼容: {platform.system()}",
                        details={
                            "current_os": platform.system(),
                            "required_os": requirements["os"],
                        },
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        category=ValidationCategory.SYSTEM_REQUIREMENTS,
                        level=CompatibilityLevel.INCOMPATIBLE,
                        message=f"操作系统不兼容: {platform.system()}",
                        details={
                            "current_os": platform.system(),
                            "required_os": requirements["os"],
                        },
                        suggestions=[f"需要运行在 {requirements['os']} 系统上"],
                    )
                )

        # 检查架构
        if "architecture" in requirements:
            required_arch = requirements["architecture"]
            current_arch = platform.machine().lower()

            if isinstance(required_arch, str):
                required_arch = [required_arch.lower()]
            elif isinstance(required_arch, list):
                required_arch = [arch.lower() for arch in required_arch]

            if any(arch in current_arch for arch in required_arch):
                results.append(
                    ValidationResult(
                        category=ValidationCategory.SYSTEM_REQUIREMENTS,
                        level=CompatibilityLevel.COMPATIBLE,
                        message=f"系统架构兼容: {platform.machine()}",
                        details={
                            "current_arch": platform.machine(),
                            "required_arch": requirements["architecture"],
                        },
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        category=ValidationCategory.SYSTEM_REQUIREMENTS,
                        level=CompatibilityLevel.INCOMPATIBLE,
                        message=f"系统架构不兼容: {platform.machine()}",
                        details={
                            "current_arch": platform.machine(),
                            "required_arch": requirements["architecture"],
                        },
                        suggestions=[f"需要 {requirements['architecture']} 架构"],
                    )
                )

        return results


class ModelFormatValidator:
    """模型格式验证器"""

    def __init__(self):
        self.supported_formats = {
            "pickle": self._validate_pickle,
            "joblib": self._validate_joblib,
            "onnx": self._validate_onnx,
            "tensorflow": self._validate_tensorflow,
            "pytorch": self._validate_pytorch,
            "qlib": self._validate_qlib,
        }

    def validate_model_format(
        self, model_path: str, expected_format: str
    ) -> ValidationResult:
        """验证模型格式"""
        model_path = Path(model_path)

        if not model_path.exists():
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"模型文件不存在: {model_path}",
                details={"model_path": str(model_path)},
                suggestions=["检查模型文件路径是否正确"],
            )

        if expected_format not in self.supported_formats:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.UNKNOWN,
                message=f"不支持的模型格式: {expected_format}",
                details={"expected_format": expected_format},
                suggestions=["使用支持的模型格式"],
            )

        try:
            validator_func = self.supported_formats[expected_format]
            return validator_func(model_path)
        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"模型格式验证失败: {e}",
                details={
                    "model_path": str(model_path),
                    "expected_format": expected_format,
                    "error": str(e),
                },
                suggestions=["检查模型文件是否损坏"],
            )

    def _validate_pickle(self, model_path: Path) -> ValidationResult:
        """验证Pickle格式"""
        try:
            with open(model_path, "rb") as f:
                pickle.load(f)

            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.COMPATIBLE,
                message="Pickle模型格式有效",
                details={"model_path": str(model_path), "format": "pickle"},
            )
        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"Pickle模型加载失败: {e}",
                details={"model_path": str(model_path), "error": str(e)},
                suggestions=["检查模型文件是否为有效的Pickle格式"],
            )

    def _validate_joblib(self, model_path: Path) -> ValidationResult:
        """验证Joblib格式"""
        try:
            import joblib

            joblib.load(model_path)

            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.COMPATIBLE,
                message="Joblib模型格式有效",
                details={"model_path": str(model_path), "format": "joblib"},
            )
        except ImportError:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message="Joblib库未安装",
                details={"model_path": str(model_path)},
                suggestions=["安装joblib: pip install joblib"],
            )
        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"Joblib模型加载失败: {e}",
                details={"model_path": str(model_path), "error": str(e)},
                suggestions=["检查模型文件是否为有效的Joblib格式"],
            )

    def _validate_onnx(self, model_path: Path) -> ValidationResult:
        """验证ONNX格式"""
        try:
            import onnx

            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)

            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.COMPATIBLE,
                message="ONNX模型格式有效",
                details={"model_path": str(model_path), "format": "onnx"},
            )
        except ImportError:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message="ONNX库未安装",
                details={"model_path": str(model_path)},
                suggestions=["安装onnx: pip install onnx"],
            )
        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"ONNX模型验证失败: {e}",
                details={"model_path": str(model_path), "error": str(e)},
                suggestions=["检查模型文件是否为有效的ONNX格式"],
            )

    def _validate_tensorflow(self, model_path: Path) -> ValidationResult:
        """验证TensorFlow格式"""
        try:
            import tensorflow as tf

            if model_path.is_dir():
                # SavedModel格式
                model = tf.saved_model.load(str(model_path))
            else:
                # H5格式
                model = tf.keras.models.load_model(str(model_path))

            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.COMPATIBLE,
                message="TensorFlow模型格式有效",
                details={"model_path": str(model_path), "format": "tensorflow"},
            )
        except ImportError:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message="TensorFlow库未安装",
                details={"model_path": str(model_path)},
                suggestions=["安装tensorflow: pip install tensorflow"],
            )
        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"TensorFlow模型加载失败: {e}",
                details={"model_path": str(model_path), "error": str(e)},
                suggestions=["检查模型文件是否为有效的TensorFlow格式"],
            )

    def _validate_pytorch(self, model_path: Path) -> ValidationResult:
        """验证PyTorch格式"""
        try:
            import torch

            model = torch.load(model_path, map_location="cpu")

            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.COMPATIBLE,
                message="PyTorch模型格式有效",
                details={"model_path": str(model_path), "format": "pytorch"},
            )
        except ImportError:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message="PyTorch库未安装",
                details={"model_path": str(model_path)},
                suggestions=["安装torch: pip install torch"],
            )
        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"PyTorch模型加载失败: {e}",
                details={"model_path": str(model_path), "error": str(e)},
                suggestions=["检查模型文件是否为有效的PyTorch格式"],
            )

    def _validate_qlib(self, model_path: Path) -> ValidationResult:
        """验证Qlib格式"""
        try:
            # Qlib模型通常是pickle格式
            return self._validate_pickle(model_path)
        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.MODEL_FORMAT,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"Qlib模型验证失败: {e}",
                details={"model_path": str(model_path), "error": str(e)},
                suggestions=["检查模型文件是否为有效的Qlib格式"],
            )


class APIInterfaceValidator:
    """API接口验证器"""

    def validate_interface_compatibility(
        self, current_schema: Dict[str, Any], new_schema: Dict[str, Any]
    ) -> List[ValidationResult]:
        """验证API接口兼容性"""
        results = []

        # 验证输入接口
        input_result = self._validate_schema_compatibility(
            current_schema.get("input", {}), new_schema.get("input", {}), "input"
        )
        results.append(input_result)

        # 验证输出接口
        output_result = self._validate_schema_compatibility(
            current_schema.get("output", {}), new_schema.get("output", {}), "output"
        )
        results.append(output_result)

        return results

    def _validate_schema_compatibility(
        self,
        current_schema: Dict[str, Any],
        new_schema: Dict[str, Any],
        schema_type: str,
    ) -> ValidationResult:
        """验证模式兼容性"""
        try:
            # 检查必需字段
            current_required = set(current_schema.get("required", []))
            new_required = set(new_schema.get("required", []))

            # 新增必需字段（破坏性变更）
            added_required = new_required - current_required
            if added_required:
                return ValidationResult(
                    category=ValidationCategory.API_INTERFACE,
                    level=CompatibilityLevel.INCOMPATIBLE,
                    message=f"{schema_type}接口不兼容: 新增必需字段",
                    details={
                        "schema_type": schema_type,
                        "added_required_fields": list(added_required),
                    },
                    suggestions=["将新增字段设为可选，或提供默认值"],
                )

            # 移除必需字段（可能的破坏性变更）
            removed_required = current_required - new_required
            if removed_required:
                return ValidationResult(
                    category=ValidationCategory.API_INTERFACE,
                    level=CompatibilityLevel.WARNING,
                    message=f"{schema_type}接口变更: 移除必需字段",
                    details={
                        "schema_type": schema_type,
                        "removed_required_fields": list(removed_required),
                    },
                    suggestions=["确认移除的字段不再被使用"],
                )

            # 检查字段类型变更
            current_properties = current_schema.get("properties", {})
            new_properties = new_schema.get("properties", {})

            type_changes = []
            for field_name in current_properties:
                if field_name in new_properties:
                    current_type = current_properties[field_name].get("type")
                    new_type = new_properties[field_name].get("type")

                    if current_type != new_type:
                        type_changes.append(
                            {
                                "field": field_name,
                                "old_type": current_type,
                                "new_type": new_type,
                            }
                        )

            if type_changes:
                return ValidationResult(
                    category=ValidationCategory.API_INTERFACE,
                    level=CompatibilityLevel.WARNING,
                    message=f"{schema_type}接口变更: 字段类型变更",
                    details={"schema_type": schema_type, "type_changes": type_changes},
                    suggestions=["确认类型变更不会影响现有客户端"],
                )

            return ValidationResult(
                category=ValidationCategory.API_INTERFACE,
                level=CompatibilityLevel.COMPATIBLE,
                message=f"{schema_type}接口兼容",
                details={"schema_type": schema_type},
            )

        except Exception as e:
            return ValidationResult(
                category=ValidationCategory.API_INTERFACE,
                level=CompatibilityLevel.UNKNOWN,
                message=f"{schema_type}接口验证失败: {e}",
                details={"schema_type": schema_type, "error": str(e)},
            )


class CompatibilityValidator:
    """模型兼容性验证器主类"""

    def __init__(self):
        self.dependency_validator = DependencyValidator()
        self.system_validator = SystemValidator()
        self.format_validator = ModelFormatValidator()
        self.api_validator = APIInterfaceValidator()

        logger.info("模型兼容性验证器初始化完成")

    def extract_model_metadata(self, model_path: str) -> ModelMetadata:
        """提取模型元数据"""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 计算文件哈希
        file_hash = self._calculate_file_hash(model_path)

        # 尝试从模型文件中提取元数据
        metadata = self._extract_metadata_from_model(model_path)

        # 如果无法从模型中提取，使用默认值
        if not metadata:
            metadata = self._create_default_metadata(model_path, file_hash)

        return metadata

    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _extract_metadata_from_model(self, model_path: Path) -> Optional[ModelMetadata]:
        """从模型文件中提取元数据"""
        try:
            # 尝试加载为pickle格式并查找元数据
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            # 如果模型包含元数据
            if isinstance(model_data, dict) and "metadata" in model_data:
                meta = model_data["metadata"]
                return ModelMetadata(
                    model_id=meta.get("model_id", "unknown"),
                    version=meta.get("version", "1.0.0"),
                    model_type=meta.get("model_type", "unknown"),
                    framework=meta.get("framework", "unknown"),
                    framework_version=meta.get("framework_version", "unknown"),
                    python_version=meta.get(
                        "python_version", platform.python_version()
                    ),
                    dependencies=meta.get("dependencies", {}),
                    input_schema=meta.get("input_schema", {}),
                    output_schema=meta.get("output_schema", {}),
                    system_requirements=meta.get("system_requirements", {}),
                    created_at=meta.get("created_at", ""),
                    file_hash=self._calculate_file_hash(model_path),
                )
        except Exception:
            pass

        return None

    def _create_default_metadata(
        self, model_path: Path, file_hash: str
    ) -> ModelMetadata:
        """创建默认元数据"""
        return ModelMetadata(
            model_id=model_path.stem,
            version="1.0.0",
            model_type="unknown",
            framework="unknown",
            framework_version="unknown",
            python_version=platform.python_version(),
            dependencies={},
            input_schema={},
            output_schema={},
            system_requirements={},
            created_at="",
            file_hash=file_hash,
        )

    def validate_compatibility(
        self, model_path: str, target_metadata: Optional[ModelMetadata] = None
    ) -> Dict[str, Any]:
        """
        验证模型兼容性

        Args:
            model_path: 模型文件路径
            target_metadata: 目标环境元数据（可选）

        Returns:
            验证结果
        """
        results = []

        try:
            # 提取模型元数据
            model_metadata = self.extract_model_metadata(model_path)

            # 1. 验证Python版本
            if model_metadata.python_version:
                python_result = self.system_validator.validate_python_version(
                    model_metadata.python_version
                )
                results.append(python_result)

            # 2. 验证依赖包
            if model_metadata.dependencies:
                dep_results = self.dependency_validator.validate_dependencies(
                    model_metadata.dependencies
                )
                results.extend(dep_results)

            # 3. 验证系统需求
            if model_metadata.system_requirements:
                sys_results = self.system_validator.validate_system_requirements(
                    model_metadata.system_requirements
                )
                results.extend(sys_results)

            # 4. 验证模型格式
            if model_metadata.framework:
                format_result = self.format_validator.validate_model_format(
                    model_path, model_metadata.framework.lower()
                )
                results.append(format_result)

            # 5. 验证API接口兼容性（如果提供了目标元数据）
            if target_metadata:
                api_results = self.api_validator.validate_interface_compatibility(
                    {
                        "input": target_metadata.input_schema,
                        "output": target_metadata.output_schema,
                    },
                    {
                        "input": model_metadata.input_schema,
                        "output": model_metadata.output_schema,
                    },
                )
                results.extend(api_results)

            # 计算总体兼容性
            overall_compatibility = self._calculate_overall_compatibility(results)

            return {
                "compatible": overall_compatibility["compatible"],
                "compatibility_level": overall_compatibility["level"],
                "summary": overall_compatibility["summary"],
                "model_metadata": model_metadata.to_dict(),
                "validation_results": [result.to_dict() for result in results],
                "recommendations": self._generate_recommendations(results),
            }

        except Exception as e:
            logger.error(f"兼容性验证失败: {e}")
            return {
                "compatible": False,
                "compatibility_level": "error",
                "summary": f"验证过程出错: {e}",
                "model_metadata": None,
                "validation_results": [],
                "recommendations": ["检查模型文件是否存在且可访问"],
            }

    def _calculate_overall_compatibility(
        self, results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """计算总体兼容性"""
        if not results:
            return {"compatible": True, "level": "unknown", "summary": "无验证结果"}

        # 统计各级别数量
        levels = [result.level for result in results]
        incompatible_count = levels.count(CompatibilityLevel.INCOMPATIBLE)
        warning_count = levels.count(CompatibilityLevel.WARNING)
        compatible_count = levels.count(CompatibilityLevel.COMPATIBLE)
        unknown_count = levels.count(CompatibilityLevel.UNKNOWN)

        # 确定总体兼容性
        if incompatible_count > 0:
            return {
                "compatible": False,
                "level": "incompatible",
                "summary": f"发现 {incompatible_count} 个不兼容问题",
            }
        elif warning_count > 0:
            return {
                "compatible": True,
                "level": "warning",
                "summary": f"发现 {warning_count} 个警告，{compatible_count} 个兼容项",
            }
        elif compatible_count > 0:
            return {
                "compatible": True,
                "level": "compatible",
                "summary": f"所有 {compatible_count} 项检查通过",
            }
        else:
            return {
                "compatible": True,
                "level": "unknown",
                "summary": f"{unknown_count} 项无法确定兼容性",
            }

    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """生成建议"""
        recommendations = []

        for result in results:
            if result.level in [
                CompatibilityLevel.INCOMPATIBLE,
                CompatibilityLevel.WARNING,
            ]:
                recommendations.extend(result.suggestions)

        # 去重并排序
        return sorted(list(set(recommendations)))


# 全局兼容性验证器实例
compatibility_validator = CompatibilityValidator()
