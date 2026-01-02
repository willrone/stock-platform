"""
数据血缘追踪器
追踪从原始数据到特征的转换，记录特征计算依赖关系
"""
import logging
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import networkx as nx
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class TransformationType(Enum):
    """转换类型"""
    DATA_LOADING = "data_loading"  # 数据加载
    DATA_CLEANING = "data_cleaning"  # 数据清洗
    FEATURE_ENGINEERING = "feature_engineering"  # 特征工程
    DATA_AGGREGATION = "data_aggregation"  # 数据聚合
    DATA_FILTERING = "data_filtering"  # 数据过滤
    DATA_JOINING = "data_joining"  # 数据连接
    DATA_SPLITTING = "data_splitting"  # 数据分割
    MODEL_TRAINING = "model_training"  # 模型训练
    MODEL_PREDICTION = "model_prediction"  # 模型预测
    CUSTOM = "custom"  # 自定义转换

class NodeType(Enum):
    """节点类型"""
    DATA_SOURCE = "data_source"  # 数据源
    DATASET = "dataset"  # 数据集
    FEATURE = "feature"  # 特征
    MODEL = "model"  # 模型
    PREDICTION = "prediction"  # 预测结果
    TRANSFORMATION = "transformation"  # 转换过程

@dataclass
class LineageNode:
    """血缘节点"""
    node_id: str
    node_type: NodeType
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'name': self.name,
            'description': self.description,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'tags': self.tags
        }

@dataclass
class LineageEdge:
    """血缘边"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    transformation_type: TransformationType
    transformation_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'edge_id': self.edge_id,
            'source_node_id': self.source_node_id,
            'target_node_id': self.target_node_id,
            'transformation_type': self.transformation_type.value,
            'transformation_config': self.transformation_config,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'description': self.description
        }

@dataclass
class LineageGraph:
    """血缘图"""
    graph_id: str
    name: str
    description: str
    nodes: Dict[str, LineageNode]
    edges: Dict[str, LineageEdge]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'graph_id': self.graph_id,
            'name': self.name,
            'description': self.description,
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': {k: v.to_dict() for k, v in self.edges.items()},
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class LineageAnalyzer:
    """血缘分析器"""
    
    def __init__(self, graph: LineageGraph):
        self.graph = graph
        self.nx_graph = self._build_networkx_graph()
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """构建NetworkX图"""
        G = nx.DiGraph()
        
        # 添加节点
        for node_id, node in self.graph.nodes.items():
            G.add_node(node_id, **node.to_dict())
        
        # 添加边
        for edge in self.graph.edges.values():
            G.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                **edge.to_dict()
            )
        
        return G
    
    def get_upstream_nodes(self, node_id: str, max_depth: Optional[int] = None) -> List[str]:
        """获取上游节点"""
        if node_id not in self.nx_graph:
            return []
        
        upstream_nodes = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_node, depth = queue.pop(0)
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # 获取前驱节点
            predecessors = list(self.nx_graph.predecessors(current_node))
            for pred in predecessors:
                if pred not in visited:
                    upstream_nodes.append(pred)
                    queue.append((pred, depth + 1))
        
        return upstream_nodes
    
    def get_downstream_nodes(self, node_id: str, max_depth: Optional[int] = None) -> List[str]:
        """获取下游节点"""
        if node_id not in self.nx_graph:
            return []
        
        downstream_nodes = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_node, depth = queue.pop(0)
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # 获取后继节点
            successors = list(self.nx_graph.successors(current_node))
            for succ in successors:
                if succ not in visited:
                    downstream_nodes.append(succ)
                    queue.append((succ, depth + 1))
        
        return downstream_nodes
    
    def get_lineage_path(self, source_node_id: str, target_node_id: str) -> List[List[str]]:
        """获取血缘路径"""
        try:
            paths = list(nx.all_simple_paths(self.nx_graph, source_node_id, target_node_id))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def find_root_nodes(self) -> List[str]:
        """查找根节点（没有前驱的节点）"""
        return [node for node in self.nx_graph.nodes() if self.nx_graph.in_degree(node) == 0]
    
    def find_leaf_nodes(self) -> List[str]:
        """查找叶子节点（没有后继的节点）"""
        return [node for node in self.nx_graph.nodes() if self.nx_graph.out_degree(node) == 0]
    
    def detect_cycles(self) -> List[List[str]]:
        """检测循环依赖"""
        try:
            cycles = list(nx.simple_cycles(self.nx_graph))
            return cycles
        except:
            return []
    
    def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """获取影响分析"""
        upstream = self.get_upstream_nodes(node_id)
        downstream = self.get_downstream_nodes(node_id)
        
        return {
            'node_id': node_id,
            'upstream_count': len(upstream),
            'downstream_count': len(downstream),
            'upstream_nodes': upstream,
            'downstream_nodes': downstream,
            'impact_score': len(downstream),  # 简单的影响分数
            'dependency_score': len(upstream)  # 简单的依赖分数
        }
    
    def get_graph_metrics(self) -> Dict[str, Any]:
        """获取图指标"""
        return {
            'node_count': self.nx_graph.number_of_nodes(),
            'edge_count': self.nx_graph.number_of_edges(),
            'density': nx.density(self.nx_graph),
            'is_dag': nx.is_directed_acyclic_graph(self.nx_graph),
            'strongly_connected_components': len(list(nx.strongly_connected_components(self.nx_graph))),
            'weakly_connected_components': len(list(nx.weakly_connected_components(self.nx_graph)))
        }

class DataLineageTracker:
    """数据血缘追踪器"""
    
    def __init__(self, storage_path: str = "data/lineage"):
        """
        初始化血缘追踪器
        
        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 血缘图存储
        self.graphs: Dict[str, LineageGraph] = {}
        self.default_graph_id = "default"
        
        # 节点和边的全局索引
        self.all_nodes: Dict[str, LineageNode] = {}
        self.all_edges: Dict[str, LineageEdge] = {}
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 初始化默认图
        self._initialize_default_graph()
        
        # 加载现有数据
        self._load_lineage_data()
        
        logger.info(f"数据血缘追踪器初始化完成，存储路径: {self.storage_path}")
    
    def _initialize_default_graph(self):
        """初始化默认血缘图"""
        default_graph = LineageGraph(
            graph_id=self.default_graph_id,
            name="默认血缘图",
            description="系统默认的数据血缘图",
            nodes={},
            edges={}
        )
        self.graphs[self.default_graph_id] = default_graph
    
    def create_node(
        self,
        node_id: str,
        node_type: NodeType,
        name: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        created_by: str = "",
        tags: Optional[List[str]] = None,
        graph_id: str = None
    ) -> str:
        """
        创建血缘节点
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            name: 节点名称
            description: 描述
            properties: 属性
            created_by: 创建者
            tags: 标签
            graph_id: 图ID
            
        Returns:
            节点ID
        """
        if graph_id is None:
            graph_id = self.default_graph_id
        
        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            description=description,
            properties=properties or {},
            created_by=created_by,
            tags=tags or []
        )
        
        with self.lock:
            # 添加到全局索引
            self.all_nodes[node_id] = node
            
            # 添加到指定图
            if graph_id in self.graphs:
                self.graphs[graph_id].nodes[node_id] = node
                self.graphs[graph_id].updated_at = datetime.now()
            
            # 保存数据
            self._save_lineage_data()
        
        logger.info(f"创建血缘节点: {name} ({node_id})")
        return node_id
    
    def create_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        transformation_type: TransformationType,
        transformation_config: Optional[Dict[str, Any]] = None,
        created_by: str = "",
        description: str = "",
        graph_id: str = None
    ) -> str:
        """
        创建血缘边
        
        Args:
            source_node_id: 源节点ID
            target_node_id: 目标节点ID
            transformation_type: 转换类型
            transformation_config: 转换配置
            created_by: 创建者
            description: 描述
            graph_id: 图ID
            
        Returns:
            边ID
        """
        if graph_id is None:
            graph_id = self.default_graph_id
        
        edge_id = f"edge_{source_node_id}_{target_node_id}_{int(datetime.now().timestamp())}"
        
        edge = LineageEdge(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transformation_type=transformation_type,
            transformation_config=transformation_config or {},
            created_by=created_by,
            description=description
        )
        
        with self.lock:
            # 验证节点存在
            if source_node_id not in self.all_nodes:
                raise ValueError(f"源节点不存在: {source_node_id}")
            if target_node_id not in self.all_nodes:
                raise ValueError(f"目标节点不存在: {target_node_id}")
            
            # 添加到全局索引
            self.all_edges[edge_id] = edge
            
            # 添加到指定图
            if graph_id in self.graphs:
                self.graphs[graph_id].edges[edge_id] = edge
                self.graphs[graph_id].updated_at = datetime.now()
            
            # 保存数据
            self._save_lineage_data()
        
        logger.info(f"创建血缘边: {source_node_id} -> {target_node_id}")
        return edge_id
    
    def track_data_transformation(
        self,
        source_data_id: str,
        target_data_id: str,
        transformation_type: TransformationType,
        transformation_config: Dict[str, Any],
        created_by: str = "",
        description: str = ""
    ) -> str:
        """
        追踪数据转换
        
        Args:
            source_data_id: 源数据ID
            target_data_id: 目标数据ID
            transformation_type: 转换类型
            transformation_config: 转换配置
            created_by: 创建者
            description: 描述
            
        Returns:
            边ID
        """
        # 确保源节点和目标节点存在
        if source_data_id not in self.all_nodes:
            self.create_node(
                node_id=source_data_id,
                node_type=NodeType.DATASET,
                name=f"数据集_{source_data_id}",
                created_by=created_by
            )
        
        if target_data_id not in self.all_nodes:
            self.create_node(
                node_id=target_data_id,
                node_type=NodeType.DATASET,
                name=f"数据集_{target_data_id}",
                created_by=created_by
            )
        
        # 创建转换边
        return self.create_edge(
            source_node_id=source_data_id,
            target_node_id=target_data_id,
            transformation_type=transformation_type,
            transformation_config=transformation_config,
            created_by=created_by,
            description=description
        )
    
    def track_feature_computation(
        self,
        source_data_ids: List[str],
        feature_id: str,
        feature_name: str,
        computation_config: Dict[str, Any],
        created_by: str = ""
    ) -> List[str]:
        """
        追踪特征计算
        
        Args:
            source_data_ids: 源数据ID列表
            feature_id: 特征ID
            feature_name: 特征名称
            computation_config: 计算配置
            created_by: 创建者
            
        Returns:
            边ID列表
        """
        # 创建特征节点
        if feature_id not in self.all_nodes:
            self.create_node(
                node_id=feature_id,
                node_type=NodeType.FEATURE,
                name=feature_name,
                properties={
                    'computation_config': computation_config,
                    'feature_type': computation_config.get('feature_type', 'unknown')
                },
                created_by=created_by
            )
        
        # 为每个源数据创建边
        edge_ids = []
        for source_data_id in source_data_ids:
            # 确保源数据节点存在
            if source_data_id not in self.all_nodes:
                self.create_node(
                    node_id=source_data_id,
                    node_type=NodeType.DATASET,
                    name=f"数据集_{source_data_id}",
                    created_by=created_by
                )
            
            edge_id = self.create_edge(
                source_node_id=source_data_id,
                target_node_id=feature_id,
                transformation_type=TransformationType.FEATURE_ENGINEERING,
                transformation_config=computation_config,
                created_by=created_by,
                description=f"计算特征: {feature_name}"
            )
            edge_ids.append(edge_id)
        
        return edge_ids
    
    def track_model_training(
        self,
        training_data_ids: List[str],
        feature_ids: List[str],
        model_id: str,
        model_name: str,
        training_config: Dict[str, Any],
        created_by: str = ""
    ) -> List[str]:
        """
        追踪模型训练
        
        Args:
            training_data_ids: 训练数据ID列表
            feature_ids: 特征ID列表
            model_id: 模型ID
            model_name: 模型名称
            training_config: 训练配置
            created_by: 创建者
            
        Returns:
            边ID列表
        """
        # 创建模型节点
        if model_id not in self.all_nodes:
            self.create_node(
                node_id=model_id,
                node_type=NodeType.MODEL,
                name=model_name,
                properties={
                    'training_config': training_config,
                    'model_type': training_config.get('model_type', 'unknown')
                },
                created_by=created_by
            )
        
        edge_ids = []
        
        # 从训练数据到模型的边
        for data_id in training_data_ids:
            if data_id not in self.all_nodes:
                self.create_node(
                    node_id=data_id,
                    node_type=NodeType.DATASET,
                    name=f"训练数据_{data_id}",
                    created_by=created_by
                )
            
            edge_id = self.create_edge(
                source_node_id=data_id,
                target_node_id=model_id,
                transformation_type=TransformationType.MODEL_TRAINING,
                transformation_config=training_config,
                created_by=created_by,
                description=f"训练模型: {model_name}"
            )
            edge_ids.append(edge_id)
        
        # 从特征到模型的边
        for feature_id in feature_ids:
            if feature_id in self.all_nodes:
                edge_id = self.create_edge(
                    source_node_id=feature_id,
                    target_node_id=model_id,
                    transformation_type=TransformationType.MODEL_TRAINING,
                    transformation_config=training_config,
                    created_by=created_by,
                    description=f"使用特征训练模型: {model_name}"
                )
                edge_ids.append(edge_id)
        
        return edge_ids
    
    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """获取节点"""
        return self.all_nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[LineageEdge]:
        """获取边"""
        return self.all_edges.get(edge_id)
    
    def get_graph(self, graph_id: str = None) -> Optional[LineageGraph]:
        """获取血缘图"""
        if graph_id is None:
            graph_id = self.default_graph_id
        return self.graphs.get(graph_id)
    
    def get_analyzer(self, graph_id: str = None) -> LineageAnalyzer:
        """获取血缘分析器"""
        if graph_id is None:
            graph_id = self.default_graph_id
        
        graph = self.graphs.get(graph_id)
        if not graph:
            raise ValueError(f"图不存在: {graph_id}")
        
        return LineageAnalyzer(graph)
    
    def get_feature_lineage(self, feature_id: str) -> Dict[str, Any]:
        """获取特征血缘"""
        analyzer = self.get_analyzer()
        
        if feature_id not in self.all_nodes:
            return {}
        
        feature_node = self.all_nodes[feature_id]
        upstream_nodes = analyzer.get_upstream_nodes(feature_id)
        downstream_nodes = analyzer.get_downstream_nodes(feature_id)
        
        # 获取上游数据源
        data_sources = []
        for node_id in upstream_nodes:
            node = self.all_nodes.get(node_id)
            if node and node.node_type in [NodeType.DATA_SOURCE, NodeType.DATASET]:
                data_sources.append(node.to_dict())
        
        # 获取下游模型
        models = []
        for node_id in downstream_nodes:
            node = self.all_nodes.get(node_id)
            if node and node.node_type == NodeType.MODEL:
                models.append(node.to_dict())
        
        return {
            'feature': feature_node.to_dict(),
            'data_sources': data_sources,
            'models': models,
            'upstream_count': len(upstream_nodes),
            'downstream_count': len(downstream_nodes)
        }
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """获取模型血缘"""
        analyzer = self.get_analyzer()
        
        if model_id not in self.all_nodes:
            return {}
        
        model_node = self.all_nodes[model_id]
        upstream_nodes = analyzer.get_upstream_nodes(model_id)
        
        # 获取训练数据
        training_data = []
        features = []
        
        for node_id in upstream_nodes:
            node = self.all_nodes.get(node_id)
            if node:
                if node.node_type in [NodeType.DATA_SOURCE, NodeType.DATASET]:
                    training_data.append(node.to_dict())
                elif node.node_type == NodeType.FEATURE:
                    features.append(node.to_dict())
        
        return {
            'model': model_node.to_dict(),
            'training_data': training_data,
            'features': features,
            'total_dependencies': len(upstream_nodes)
        }
    
    def search_nodes(
        self,
        node_type: Optional[NodeType] = None,
        name_pattern: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> List[LineageNode]:
        """搜索节点"""
        nodes = list(self.all_nodes.values())
        
        # 过滤条件
        if node_type:
            nodes = [n for n in nodes if n.node_type == node_type]
        
        if name_pattern:
            nodes = [n for n in nodes if name_pattern.lower() in n.name.lower()]
        
        if tags:
            nodes = [n for n in nodes if any(tag in n.tags for tag in tags)]
        
        if created_by:
            nodes = [n for n in nodes if n.created_by == created_by]
        
        return nodes
    
    def get_lineage_summary(self, graph_id: str = None) -> Dict[str, Any]:
        """获取血缘摘要"""
        if graph_id is None:
            graph_id = self.default_graph_id
        
        analyzer = self.get_analyzer(graph_id)
        metrics = analyzer.get_graph_metrics()
        
        # 按类型统计节点
        node_type_counts = {}
        for node in self.all_nodes.values():
            node_type = node.node_type.value
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        # 按类型统计转换
        transformation_type_counts = {}
        for edge in self.all_edges.values():
            trans_type = edge.transformation_type.value
            transformation_type_counts[trans_type] = transformation_type_counts.get(trans_type, 0) + 1
        
        return {
            'graph_metrics': metrics,
            'node_type_counts': node_type_counts,
            'transformation_type_counts': transformation_type_counts,
            'total_nodes': len(self.all_nodes),
            'total_edges': len(self.all_edges)
        }
    
    def export_lineage_graph(self, graph_id: str = None, format: str = "json") -> str:
        """导出血缘图"""
        if graph_id is None:
            graph_id = self.default_graph_id
        
        graph = self.graphs.get(graph_id)
        if not graph:
            raise ValueError(f"图不存在: {graph_id}")
        
        if format == "json":
            return json.dumps(graph.to_dict(), indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _load_lineage_data(self):
        """加载血缘数据"""
        try:
            lineage_file = self.storage_path / "lineage.json"
            if lineage_file.exists():
                with open(lineage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 加载节点
                for node_data in data.get('nodes', []):
                    node = LineageNode(
                        node_id=node_data['node_id'],
                        node_type=NodeType(node_data['node_type']),
                        name=node_data['name'],
                        description=node_data.get('description', ''),
                        properties=node_data.get('properties', {}),
                        created_at=datetime.fromisoformat(node_data['created_at']),
                        created_by=node_data.get('created_by', ''),
                        tags=node_data.get('tags', [])
                    )
                    self.all_nodes[node.node_id] = node
                
                # 加载边
                for edge_data in data.get('edges', []):
                    edge = LineageEdge(
                        edge_id=edge_data['edge_id'],
                        source_node_id=edge_data['source_node_id'],
                        target_node_id=edge_data['target_node_id'],
                        transformation_type=TransformationType(edge_data['transformation_type']),
                        transformation_config=edge_data.get('transformation_config', {}),
                        created_at=datetime.fromisoformat(edge_data['created_at']),
                        created_by=edge_data.get('created_by', ''),
                        description=edge_data.get('description', '')
                    )
                    self.all_edges[edge.edge_id] = edge
                
                # 重建默认图
                default_graph = self.graphs[self.default_graph_id]
                default_graph.nodes = self.all_nodes.copy()
                default_graph.edges = self.all_edges.copy()
                
                logger.info(f"加载了 {len(self.all_nodes)} 个节点，{len(self.all_edges)} 条边")
        
        except Exception as e:
            logger.error(f"加载血缘数据失败: {e}")
    
    def _save_lineage_data(self):
        """保存血缘数据"""
        try:
            data = {
                'nodes': [node.to_dict() for node in self.all_nodes.values()],
                'edges': [edge.to_dict() for edge in self.all_edges.values()],
                'last_updated': datetime.now().isoformat()
            }
            
            lineage_file = self.storage_path / "lineage.json"
            with open(lineage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"保存血缘数据失败: {e}")

# 全局数据血缘追踪器实例
data_lineage_tracker = DataLineageTracker()