"""
ATLAS Core: Space Registry
空间注册表

管理所有可用的认知空间实现
"""

from typing import Dict, Type, Callable, Any
import inspect

from .space import CognitiveSpace


class SpaceRegistry:
    """
    认知空间注册表

    单例模式，全局管理所有空间类型
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._spaces: Dict[str, Type[CognitiveSpace]] = {}
            cls._instance._factories: Dict[str, Callable] = {}
        return cls._instance

    def register(self, name: str, space_class: Type[CognitiveSpace]) -> 'SpaceRegistry':
        """
        注册空间类

        Args:
            name: 空间类型的名称标识
            space_class: 继承自 CognitiveSpace 的类
        """
        if not issubclass(space_class, CognitiveSpace):
            raise TypeError(f"{space_class} must inherit from CognitiveSpace")

        self._spaces[name] = space_class
        return self

    def register_factory(self, name: str,
                        factory: Callable[..., CognitiveSpace]) -> 'SpaceRegistry':
        """
        注册空间工厂函数

        用于需要复杂初始化逻辑的空间
        """
        self._factories[name] = factory
        return self

    def create(self, name: str, width: int, height: int,
              **kwargs) -> CognitiveSpace:
        """
        创建空间实例

        Args:
            name: 注册的空间名称
            width, height: 空间尺寸
            **kwargs: 传递给构造函数的额外参数

        Returns:
            CognitiveSpace 实例
        """
        if name in self._factories:
            return self._factories[name](width, height, **kwargs)

        if name not in self._spaces:
            raise KeyError(f"Unknown space type: {name}. "
                          f"Available: {list(self._spaces.keys())}")

        space_class = self._spaces[name]
        return space_class(width, height, **kwargs)

    def list_spaces(self) -> Dict[str, str]:
        """列出所有可用的空间类型及其文档"""
        result = {}
        for name, space_class in self._spaces.items():
            doc = inspect.getdoc(space_class)
            result[name] = doc.split('\n')[0] if doc else "No documentation"
        return result

    def get_space_class(self, name: str) -> Type[CognitiveSpace]:
        """获取空间类（用于 inspection）"""
        return self._spaces[name]


# 全局注册表实例
registry = SpaceRegistry()


def register_space(name: str):
    """
    装饰器：自动注册空间类

    用法:
        @register_space("ricci")
        class RicciSpace(CognitiveSpace):
            ...
    """
    def decorator(cls: Type[CognitiveSpace]) -> Type[CognitiveSpace]:
        registry.register(name, cls)
        return cls
    return decorator


def discover_spaces():
    """
    自动发现 spaces 目录下的所有空间实现

    需要在初始化时调用一次
    """
    # 延迟导入，避免循环依赖
    try:
        from ..spaces import ricci, fisher, wasserstein, finsler, conformal
        # 导入即注册（通过装饰器）
    except ImportError as e:
        print(f"Warning: Could not import some space modules: {e}")


# 便捷函数
def create_space(name: str, width: int, height: int, **kwargs) -> CognitiveSpace:
    """便捷函数：创建空间实例"""
    return registry.create(name, width, height, **kwargs)


def list_available_spaces() -> Dict[str, str]:
    """便捷函数：列出可用空间"""
    return registry.list_spaces()
